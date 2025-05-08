import logging
import os
import time

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from transformiloop.src.data.spindle.spindle_detection.datasets.abstract_spindle_dataset import AbstractSpindleDataset
from transformiloop.src.data.spindle.spindle_detection.datasets.finetune_dataset import FinetuneDataset
from transformiloop.src.data.spindle.spindle_detection.samplers.random_sampler import RandomSampler
from transformiloop.src.data.spindle.spindle_detection.samplers.validation_sampler import ValidationSampler

DATASET_FILE = 'dataset_classification_full_big_250_matlab_standardized_envelope_pf.txt'

def get_subject_list(dataset_path:str)->tuple[ ndarray, ndarray, ndarray]:
    """
    Split the dataset into train, validation and test subject list.

    Args:
        dataset_path (str): path to the dataset folder.

    Returns:
        tuple[ ndarray, ndarray, ndarray ]: train_subject, validation_subject, test_subject lists.
    """
    # Load all subject files
    all_subject = pd.read_csv(os.path.join(dataset_path, "subject_sequence_full_big.txt"), header=None, delim_whitespace=True).to_numpy()
    p1_subject = pd.read_csv(os.path.join(dataset_path, 'subject_sequence_p1_big.txt'), header=None, delim_whitespace=True).to_numpy()
    p2_subject = pd.read_csv(os.path.join(dataset_path, 'subject_sequence_p2_big.txt'), header=None, delim_whitespace=True).to_numpy()

    # Get splits for train, validation and test
    train_subject_p1, validation_subject_p1 = train_test_split(p1_subject, train_size=0.8, random_state=None)
    test_subject_p1, validation_subject_p1 = train_test_split(validation_subject_p1, train_size=0.5, random_state=None)
    train_subject_p2, validation_subject_p2 = train_test_split(p2_subject, train_size=0.8, random_state=None)
    test_subject_p2, validation_subject_p2 = train_test_split(validation_subject_p2, train_size=0.5, random_state=None)

    # Get subject list depending on split
    train_subject = np.array([s for s in all_subject if s[0] in train_subject_p1[:, 0] or s[0] in train_subject_p2[:, 0]]).squeeze()
    test_subject = np.array([s for s in all_subject if s[0] in test_subject_p1[:, 0] or s[0] in test_subject_p2[:, 0]]).squeeze()
    validation_subject = np.array(
        [s for s in all_subject if s[0] in validation_subject_p1[:, 0] or s[0] in validation_subject_p2[:, 0]]).squeeze()

    print(f"Subjects in training : {train_subject[:, 0]}")
    print(f"Subjects in validation : {validation_subject[:, 0]}")
    print(f"Subjects in test : {test_subject[:, 0]}")
    
    return train_subject, validation_subject, test_subject


def get_data(dataset_path:str)->ndarray:
    """
    Load the dataset from the dataset_path.

    Args:
        dataset_path (str): path to the dataset folder.

    Returns:
        ndarray: data array.
    """
    start = time.time()
    data = pd.read_csv(os.path.join(dataset_path, DATASET_FILE), header=None).to_numpy()
    end = time.time()
    logging.info(f"Loaded data in {(end-start)} seconds...")
    return data


def get_class_idxs(dataset:AbstractSpindleDataset, distribution_mode:int)->tuple[ndarray, ndarray]:
    """
    Directly outputs idx_true and idx_false arrays

    Args:
        dataset (AbstractSpindleDataset): dataset object.
        distribution_mode (int): distribution mode (0: no modification, 1: only spindles).
    """
    length_dataset = len(dataset)

    nb_true = 0
    nb_false = 0

    idx_true = []
    idx_false = []

    for i in range(length_dataset):
        is_spindle = dataset.is_spindle(i)
        if is_spindle or distribution_mode == 1:
            nb_true += 1
            idx_true.append(i)
        else:
            nb_false += 1
            idx_false.append(i)

    assert len(dataset) == nb_true + nb_false, f"Bad length dataset"

    return np.array(idx_true), np.array(idx_false)


def get_info_subject(subjects:ndarray, config:dict)->tuple[int, int]:
    """
    Returns the number of segments and the batch size for a given subject list.
    Args:
        subjects (ndarray): subject list.
        config (dict): config dictionary.

    Returns:
        tuple[int, int]: nb_segment, batch_size.
    """
    nb_segment = len(np.hstack([range(int(s[1]), int(s[2])) for s in subjects]))
    batch_size = len(list(range(0, (config['seq_stride'] // config['network_stride']) * config['network_stride'], config['network_stride']))) * nb_segment
    return nb_segment, batch_size


def get_dataloaders(config:dict, dataset_path:str)->tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns the dataloaders for the train, validation and test sets.

    Args:
        config (dict): config dictionary.
        dataset_path (str): path to the dataset folder.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train_dl, val_dl, test_dl dataloaders.
    """
    subs_train, subs_val, subs_test = get_subject_list(dataset_path)
    data = get_data(dataset_path)

    train_ds = FinetuneDataset(subs_train, config, data, config['full_transformer'], augmentation_config=None, device=config['device'])
    val_ds = FinetuneDataset(subs_val, config, data, False, augmentation_config=None, device=config['device'])
    test_ds = FinetuneDataset(subs_test, config, data, False, augmentation_config=None, device=config['device'])

    idx_true, idx_false = get_class_idxs(train_ds, 0)

    train_sampler = RandomSampler(idx_true, idx_false, config)

    nb_segment_val, batch_size_val = get_info_subject(subs_val, config)
    nb_segment_test, batch_size_test = get_info_subject(subs_test, config)
    config['batch_size_validation'] = batch_size_val
    config['batch_size_test'] = batch_size_test

    print(f"Batch Size validation: {batch_size_val}")
    print(f"Batch Size test: {batch_size_test}")

    val_sampler = ValidationSampler(config['seq_stride'], nb_segment_val, config['network_stride'])
    test_sampler = ValidationSampler(config['seq_stride'], nb_segment_test, config['network_stride'])
    
    if config['pretraining']:
        train_dl = DataLoader(
            train_ds, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True)
    else:
        train_dl = DataLoader(
            train_ds, 
            batch_size=config['batch_size'],
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True)
        
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size_val,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        shuffle=False)

    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size_test,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=True)

    return train_dl, val_dl, test_dl
