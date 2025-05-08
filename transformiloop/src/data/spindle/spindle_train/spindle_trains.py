import json
import os
import pathlib

import pandas as pd
from torch.utils.data import DataLoader

from transformiloop.src.data.pretraining import read_pretraining_dataset
from transformiloop.src.data.sleep_stage import divide_subjects_into_sets
from transformiloop.src.data.spindle.spindle_train.samplers.equirandom_sampler import EquiRandomSampler
from transformiloop.src.data.spindle.spindle_train.datasets.spindle_train_dataset import SpindleTrainDataset


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

    def convert_row_to_250hz(row:pd.Series)->pd.Series | None:
        """
        Convert the row to 250 Hz
        Args:
            row (pd.Series): The row to convert.
        Returns:
            pd.Series | None: The converted row or None if the row is invalid.
        """
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

def get_dataloaders_spindle_trains(MASS_dir:str, ds_dir:str, config:dict)->tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders
    Args:
        MASS_dir (str): The path to the MASS directory.
        ds_dir (str): The path to the dataset directory.
        config (dict): The configuration object.

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


def read_spindle_trains_labels(ds_dir:str)->dict[str, dict[str, list[int|str]]]:
    """
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    Args:
        ds_dir (str): The path to the dataset directory.

    Returns:
        dict[str, dict[str, list[int]]]: A dictionary with the subjects id as keys and a dictionary with the onsets, offsets and labels as values.
    """
    spindle_trains_file = os.path.join(ds_dir, 'spindle_trains_annots.json')
    # Read the JSON file
    with open(spindle_trains_file, 'r') as f:
        labels = json.load(f)
    return labels

if __name__ == "__main__":
    # Get the path to the dataset directory
    dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
    generate_spindle_trains_dataset(str(dataset_path / 'SpindleTrains_raw_data'),
                                    str(dataset_path / 'spindle_trains_annots.json'))