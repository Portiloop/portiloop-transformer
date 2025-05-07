import json
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from wandb.sdk import Config

from transformiloop.src.data.pretraining import read_pretraining_dataset
from transformiloop.src.data.sleep_stage import divide_subjects_into_sets


def generate_spindle_trains_dataset(raw_dataset_path:str, output_file:str, electrode:str='Cz'):
    """
    Constructs a dataset of spindle trains from the MASS dataset
    Args:
        raw_dataset_path (str): The path to the raw dataset directory.
        output_file (str): The path to the output file.
        electrode (str): The electrode to use. Defaults to 'Cz'.
    """
    data = {}

    spindle_infos = os.listdir(os.path.join(raw_dataset_path, "spindle_info"))

    # List all files in the subject directory
    for subject_dir in os.listdir(os.path.join(raw_dataset_path, "subject_info")):
        subset = subject_dir[5:8]
        # Get the spindle info file where the subset is in the filename
        spindle_info_file = [f for f in spindle_infos if subset in f and electrode in f][0]

        # Read the spindle info file
        train_ds_ss = read_spindle_train_info(
            os.path.join(raw_dataset_path, "subject_info", subject_dir),
            os.path.join(raw_dataset_path, "spindle_info", spindle_info_file))
        
        # Append the data
        data.update(train_ds_ss)

    # Write the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f)


def read_spindle_train_info(subject_dir:str, spindle_info_file:str)->dict[str, dict[str, list[int]]]:
    """
    Read the spindle train info from the given subject directory and spindle info file
    Args:
        subject_dir (str): The path to the subject directory.
        spindle_info_file (str): The path to the spindle info file.

    Returns:
        data (dict[str, dict[str, list[int]]]): A dictionary containing the spindle train info for each subject.
    """
    subject_names = pd.read_csv(subject_dir, header=None).to_numpy()[:, 0]
    spindle_info = pd.read_csv(spindle_info_file)
    headers = list(spindle_info.columns)[:-1]

    data = {}
    for subj in subject_names:
        data[subj] = {
            headers[0]: [],
            headers[1]: [],
            headers[2]: [],
        }
    subject_counter = 1
    for index, row in spindle_info.iterrows():
        if index != 0 and row['onsets'] < spindle_info.iloc[index-1]['onsets']:
            subject_counter += 1
    
    assert subject_counter == len(subject_names), \
        f"The number of subjects in the subject_info file and the spindle_info file should be the same, \
            found {len(subject_names)} and {subject_counter} respectively"

    def convert_row_to_250hz(row):
        "Convert the row to 250hz"
        row['onsets'] = int((row['onsets'] / 256) * 250)
        row['offsets'] = int((row['offsets'] / 256) * 250)
        if row['onsets'] == row['offsets']:
            return None
        assert row['onsets'] < row['offsets'], "The onset should be smaller than the offset"
        return row

    subject_names_counter = 0
    for index, row in spindle_info.iterrows():
        if index != 0 and row['onsets'] < spindle_info.iloc[index-1]['onsets']:
            subject_names_counter += 1
        row = convert_row_to_250hz(row)
        if row is None:
            continue
        for h in headers:
            data[subject_names[subject_names_counter]][h].append(row[h])

    for subj in subject_names:
        assert len(data[subj][headers[0]]) == len(data[subj][headers[1]]) == len(data[subj][headers[2]]), "The number of onsets, offsets and labels should be the same"

    return data

def get_dataloaders_spindle_trains(MASS_dir:str, ds_dir:str, config:Config)->tuple[DataLoader, DataLoader]:
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
    labels = read_spindle_trains_labels(ds_dir)

    train_subjects, test_subjects = divide_subjects_into_sets(labels)

    # Read the pretraining dataset
    data = read_pretraining_dataset(MASS_dir)

    # Create the train and test datasets
    train_dataset = SpindleTrainDataset(train_subjects, data, labels, config)
    test_dataset = SpindleTrainDataset(test_subjects, data, labels, config)

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=EquiRandomSampler(train_dataset, sample_list=[1, 2]),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=EquiRandomSampler(test_dataset, sample_list=[1, 2]),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


def read_spindle_trains_labels(ds_dir):
    '''
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    '''
    spindle_trains_file = os.path.join(ds_dir, 'spindle_trains_annots.json')
    # Read the json file
    with open(spindle_trains_file, 'r') as f:
        labels = json.load(f)
    return labels


class SpindleTrainDataset(Dataset):
    def __init__(self, subjects, data, labels, config):
        '''
        This class takes in a list of subjects, a path to the MASS directory
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

        self.config = config
        self.window_size = config['window_size']
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + self.window_size

        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []
        self.spindle_labels_iso = []
        self.spindle_labels_first = []
        self.spindle_labels_train = []

        accumulator = 0
        for subject in subjects:
            if subject not in data.keys():
                print(f"Subject {subject} not found in the pretraining dataset")
                continue

            # Get the signal for the given subject
            signal = torch.tensor(
                data[subject]['signal'], dtype=torch.float)

            # Get all the labels for the given subject
            label = torch.zeros_like(signal, dtype=torch.uint8)
            for index, (onset, offset, l) in enumerate(zip(labels[subject]['onsets'], labels[subject]['offsets'], labels[subject]['labels_num'])):
                
                # Some of the spindles in the dataset overlap with the previous spindle
                # If that is the case, we need to make sure that the onset is at least the offset of the previous spindle
                if onset < labels[subject]['offsets'][index - 1]:
                    onset = labels[subject]['offsets'][index - 1]

                label[onset:offset] = l
                # Make a separate list with the indexes of all the spindle labels so that sampling is easier
                to_add = list(range(accumulator + onset, accumulator + offset))
                assert offset < len(signal), f"Offset {offset} is greater than the length of the signal {len(signal)} for subject {subject}"
                if l == 1:
                    self.spindle_labels_iso += to_add
                elif l == 2:
                    self.spindle_labels_first += to_add
                elif l == 3:
                    self.spindle_labels_train += to_add
                else:
                    raise ValueError(f"Unknown label {l} for subject {subject}")
            # increment the accumulator
            accumulator += len(signal)

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels.append(label)
            self.full_signal.append(signal)
            del data[subject], signal, label
        
        # Concatenate the full signal and the full labels into one continuous tensor
        self.full_signal = torch.cat(self.full_signal)
        self.full_labels = torch.cat(self.full_labels)

        # Shuffle the spindle labels
        start = time.time()
        random.shuffle(self.spindle_labels_iso)
        random.shuffle(self.spindle_labels_first)
        random.shuffle(self.spindle_labels_train)
        end = time.time()
        print(f"Shuffling took {end - start} seconds")
        print(f"Number of spindle labels: {len(self.spindle_labels_iso) + len(self.spindle_labels_first) + len(self.spindle_labels_train)}")


    @staticmethod
    def get_labels():
        return ['non-spindle', 'isolated', 'first', 'train']

    def __getitem__(self, index):

        # Get data
        index = index + self.past_signal_len
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)
        label = self.full_labels[index + self.window_size - 1]

        # Make sure that the last index of the signal is the same as the label
        # assert signal[-1, -1] == self.full_signal[index + self.window_size - 1], "Issue with the data and the labels"
        label = label.type(torch.LongTensor)

        assert label in [1, 2], f"Label {label} is not 1 or 2"

        return signal, label-1

    def __len__(self):
        return len(self.full_signal) - self.window_size


if __name__ == "__main__":
    # Get the path to the dataset directory
    dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
    generate_spindle_trains_dataset(dataset_path / 'SpindleTrains_raw_data', dataset_path / 'spindle_trains_annots.json')
