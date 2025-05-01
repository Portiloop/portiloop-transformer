import csv
import os
import random

import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from wandb.sdk import Config

from transformiloop.src.data.pretraining import read_pretraining_dataset

def divide_subjects_into_sets(labels:dict[str, list[str]])->tuple[list[str], list[str]]:
    """
    Divide the subjects into train and test sets.

    Args:
        labels (dict[str, list[str]]): A dictionary with the subjects id as keys and a list of sleep stages as values.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists: the first list contains the subjects in the train set, and the second list contains the subjects in the test set.
    """
    subjects = list(labels.keys())
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    test_subjects = subjects[int(len(subjects) * 0.8):]
    return train_subjects, test_subjects

def get_dataloaders_sleep_stage(MASS_dir:str, ds_dir:str, config:Config)->tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders

    Args:
        MASS_dir (str): The path to the MASS directory.
        ds_dir (str): The path to the dataset directory.
        config (Config): The configuration object.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing two dataloaders: the first one for the train set, and the second one for the test set.
    """
    # Read all the subjects available in the dataset
    labels = read_sleep_staging_labels(ds_dir) 

    train_subjects, test_subjects = divide_subjects_into_sets(labels)

    # Read the pretraining dataset
    data = read_pretraining_dataset(MASS_dir)

    # Create the train and test datasets
    train_dataset = SleepStageDataset(train_subjects, data, labels, config)
    test_dataset = SleepStageDataset(test_subjects, data, labels, config)

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SleepStageSampler(train_dataset, config),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=SleepStageSampler(test_dataset, config),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


def read_sleep_staging_labels(MASS_dir:str)->dict[str, list[str]]:
    """
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary

    Args:
        MASS_dir (str): The path to the MASS directory

    Returns:
        dict[str, list[str]]: A dictionary with the subjects id as keys and a list of sleep stages as values.
    """
    # Read the sleep_staging.csv file 
    sleep_staging_file = os.path.join(MASS_dir, 'sleep_staging.csv')
    with open(sleep_staging_file, 'r') as f:
        reader = csv.reader(f)
        # Remove the header line from the information
        sleep_staging = list(reader)[1:]

    sleep_stages = {}
    for i in range(len(sleep_staging)):
        subject = sleep_staging[i][0]
        sleep_stages[subject] = [stage for stage in sleep_staging[i][1:] if stage != '']

    return sleep_stages

class SleepStageDataset(Dataset):
    """
    This class represents a dataset for sleep stage prediction.

    :ivar config: Configuration options containing parameters such as `window_size`, `seq_len`,
        and `seq_stride`.
    :type config: Config
    :ivar window_size: Size of the window for extracting signal patches.
    :type window_size: int
    :ivar seq_len: Length of the sequence used in preprocessing.
    :type seq_len: int
    :ivar seq_stride: Stride length for advancing the sequence window.
    :type seq_stride: int
    :ivar past_signal_len: Number of past signals required before the last window in a sequence.
    :type past_signal_len: int
    :ivar full_signal: Concatenated tensor of all signals from the specified subjects.
    :type full_signal: Tensor
    :ivar full_labels: Concatenated tensor of all labels associated with the signals.
    :type full_labels: Tensor
    """
    def __init__(self, subjects:list[str], data:dict[str, dict[str, int | str | ndarray | Tensor]], labels:dict[str, list[str]], config:Config):
        """
        This class takes in a list of subject, a path to the MASS directory
        and reads the files associated with the given subjects as well as the sleep stage annotations

        Args:
            subjects (list[str]): List of subject IDs
            data (dict[str, dict[str, int | str | ndarray | Tensor]]): Dictionary containing the pretraining dataset
            labels (dict[str, list[str]]): Dictionary containing the sleep stage labels for each subject
            config (Config): Configuration dictionary containing required settings.
        """
        super().__init__()

        self.config = config
        self.window_size = config['window_size']
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride

        # Get the sleep stage labels
        full_signal = []
        full_labels = []

        for subject in subjects:
            if subject not in data.keys():
                print(f"Subject {subject} not found in the pretraining dataset")
                continue
            # assert subject in data.keys(), f"Subject {subject} not found in the pretraining dataset" 
            signal = torch.tensor(
                data[subject]['signal'], dtype=torch.float)
            # Get all the labels for the given subject
            label = []
            for lab in labels[subject]:
                label += [SleepStageDataset.get_labels().index(lab)] * self.config['fe']
            
            # Add some '?' padding at the end to make sure the length of signal and label match
            label += [SleepStageDataset.get_labels().index('?')] * (len(signal) - len(label))

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            full_labels.append(torch.tensor(label, dtype=torch.uint8))
            full_signal.append(signal)
            del data[subject], signal, label
        
        self.full_signal = torch.cat(full_signal)
        self.full_labels = torch.cat(full_labels)

    @staticmethod
    def get_labels():
        """
        Return the list of sleep stage labels.

        Returns:
            list[str]: List of sleep stage labels.
        """
        return ['1', '2', '3', 'R', 'W', '?']

    def __getitem__(self, index:int) -> tuple[Tensor, Tensor]:
        """
        Retrieve a sequence of signal data and corresponding label based on the provided index.

        :param index: The index of the data to retrieve. It is adjusted internally
            by adding the length of the past signal.

        :return: A tuple containing:
            - A tensor representing the unfolded signal data within the specified
              window size and sequence stride.
            - A tensor containing the label corresponding to the end of the unfolded
              signal window, converted to a long integer type.

        :raises AssertionError: Raised if the label at the specified index is equal
            to 5, indicating that the label is invalid (represented as '?').
        """
        # Get the signal and label at the given index
        index += self.past_signal_len

        # Get data
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        label = self.full_labels[index + self.window_size - 1]

        assert label != 5, "Label is '?'"

        return signal, label.type(torch.long)

    def __len__(self):
        """
        Calculate the number of elements in the `full_signal`.

        :return: The length of the `full_signal` sequence.
        :rtype: int
        """
        return len(self.full_signal)

class SleepStageSampler(Sampler):
    """
    A data sampler class for sleep stage datasets.

    :ivar dataset: The dataset from which the sampler draws samples.
    :type dataset: SleepStageDataset
    :ivar window_size: The size of the data window to sample, as specified in the configuration.
    :type window_size: int
    :ivar max_len: The maximum index for sampling, adjusted for past signals and window size.
    :type max_len: int
    """
    def __init__(self, dataset:SleepStageDataset, config:Config):
        """
        Initializes the instance of the class.

        Args:
            dataset (SleepStageDataset): The dataset object containing sleep stage data.
            config (Config): Configuration dictionary containing required settings
        """
        super().__init__()
        self.dataset = dataset
        self.window_size = config['window_size']
        self.max_len = len(dataset) - self.dataset.past_signal_len - self.window_size

    def __iter__(self):
        """
        Provides an iterator interface for random sampling over a dataset that ensures
        specific conditions are met for the sampled data. The method generates indices
        to access the dataset while verifying that the label at a particular position
        is valid and not a placeholder.

        :return: Yields indices of entries in the dataset that meet the specified
            conditions.
        :rtype: Iterator[int]
        """
        while True:
            index = random.randint(0, self.max_len - 1)
            # Make sure that the label at the end of the window is not '?'
            label = self.dataset.full_labels[index + self.dataset.past_signal_len + self.window_size - 1]
            if label != SleepStageDataset.get_labels().index('?'):
                yield index

    def __len__(self):
        """
        Represents the length functionality for the object containing a dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.dataset)