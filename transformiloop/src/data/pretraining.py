import csv
import logging
import os
import traceback

import pyedflib
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from wandb.sdk import Config


def read_patients_info(dataset_path:str)->dict[str,dict[str,int|str|ndarray|Tensor]]:
    """
    Read the patient info from a patient_info file and initialize it in a dictionary

    Args:
        dataset_path (str): Path to the dataset directory (NOT THE INFO FILE).

    Returns:
        dict[str,dict[str, int|str|ndarray]]: Dictionary containing the patient info (age, gender) for each subject in the dataset.

    """
    patient_info_file = os.path.join(dataset_path, 'patient_info.csv')
    with open(patient_info_file, 'r') as patient_info_f:
        # Skip the header if present
        has_header = csv.Sniffer().has_header(patient_info_f.read(1024))
        patient_info_f.seek(0)  # Rewind.
        reader = csv.reader(patient_info_f)
        if has_header:
            next(reader)  # Skip the header row.

        patients_info = {
            line[0]: {
                'age': int(line[1]),
                'gender': line[2]
            } for line in reader
        }
    return patients_info


def read_pretraining_dataset(dataset_path:str, patients_to_keep:list[str]=None)->dict[str,dict[str,int|str|ndarray|Tensor]]:
    """
    Load all dataset files into a dictionary to be ready for a Pytorch Dataset.
    Note that this will only read the first signal even if the EDF file contains more.

    Args:
        dataset_path (str): Path to the dataset directory.
        patients_to_keep (list[str]): List of patient IDs to keep in the dataset. If None, all patients are kept.

    Returns:
        dict[str,dict[str, int|str|ndarray]]: Dictionary containing the patient info (age, gender) for each subject in the dataset.
    """
    patients_info = read_patients_info(dataset_path)

    for patient_id in patients_info.keys():
        if patients_to_keep is not None and patient_id not in patients_to_keep:
            continue
        try:
            filename = os.path.join(dataset_path, patient_id + ".edf")
        except FileNotFoundError as e:
            print("FileNotFoundError:", e)
            traceback.print_exc()
            raise
        try:
            with pyedflib.EdfReader(filename) as edf_file:
                patients_info[patient_id]['signal'] = edf_file.readSignal(0)
        except FileNotFoundError:
            logging.debug(f"Skipping file {filename} as it is not in dataset.")

    # Remove all patients whose signal is not in the dataset
    dataset = {patient_id: patient_details for (patient_id, patient_details) in patients_info.items()
        if 'signal' in patients_info[patient_id].keys()}

    return dataset


class PretrainingDataset(Dataset):
    """
    A dataset class designed for pretraining models on sequential data.

    This class is tailored for processing large datasets containing signals, associated
    genders, and ages. It prepares the data for masked sequence modeling tasks. The class
    takes into account masking probabilities and ensures consistent handling during
    sampling. The class also includes mechanisms for sorting and organizing data
    by gender and age.

    :ivar device: The device on which data should reside (CPU or GPU).
    :type device: Optional[torch.device]
    :ivar window_size: The size of the sliding window used for sequences.
    :type window_size: int
    :ivar subjects: A sorted list of subjects based on age and gender.
    :type subjects: List[str]
    :ivar nb_subjects: Number of subjects in the dataset.
    :type nb_subjects: int
    :ivar seq_len: The length of each sequence fed into the model.
    :type seq_len: int
    :ivar seq_stride: The stride of the sliding window for sequences.
    :type seq_stride: int
    :ivar past_signal_len: Signal length required before the last window in a sequence.
    :type past_signal_len: int
    :ivar min_signal_len: Minimum signal length required to generate a valid sample.
    :type min_signal_len: int
    :ivar full_signal: A concatenated tensor of all subject signals (float32).
    :type full_signal: Tensor
    :ivar genders: A concatenated tensor of genders corresponding to all signals (uint8).
    :type genders: Tensor
    :ivar ages: A concatenated tensor of ages corresponding to all signals (uint8).
    :type ages: Tensor
    :ivar samplable_len: The number of sampleable signal windows from the dataset.
    :type samplable_len: int
    :ivar mask_probs: Probabilities for each masking type (not masked, masked, replaced, kept).
    :type mask_probs: torch.Tensor
    :ivar mask_cum_probs: Cumulative probabilities derived from mask_probs.
    :type mask_cum_probs: torch.Tensor
    """
    def __init__(self, dataset_path:str, config:Config, device:torch.device=None):
        self.device = device
        self.window_size = config['window_size']

        data = read_pretraining_dataset(dataset_path)

        def compute_subject_sorting_weight(subject:str)->int:
            """
            Compute a weight for a subject based on its gender and age in purpose of sorting.

            Args:
                subject (str): Subject ID.

            Returns:
                int: Weight for sorting.
            """
            res = 0
            assert data[subject]['age'] < 255, f"{data[subject]['age']} years is a bit old."
            if data[subject]['gender'] == 'M':
                res += 1000
            res += data[subject]['age']
            return res
        self.subjects = sorted(data.keys(), key=compute_subject_sorting_weight)
        self.nb_subjects = len(self.subjects)

        logging.debug(f"DEBUG: {self.nb_subjects} subjects:")
        for subject in self.subjects:
            logging.debug(
                f"DEBUG: {subject}, {data[subject]['gender']}, {data[subject]['age']} yo")

        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + \
            self.window_size  # signal needed for one sample

        full_signal = []
        genders = []
        ages = []

        for subject in self.subjects:
            assert self.min_signal_len <= len(
                data[subject]['signal']), f"Signal {subject} is too short."
            data[subject]['signal'] = torch.tensor(
                data[subject]['signal'], dtype=torch.float)
            full_signal.append(data[subject]['signal'])
            gender = 1 if data[subject]['gender'] == 'M' else 0
            age = data[subject]['age']
            ones = torch.ones_like(data[subject]['signal'], dtype=torch.uint8)
            gender_tensor = ones * gender
            age_tensor = ones * age
            genders.append(gender_tensor)
            ages.append(age_tensor)
            del data[subject]  # we delete this as it is not necessary anymore

        # all signals concatenated (float32)
        self.full_signal = torch.cat(full_signal)
        # all corresponding genders (uint8)
        self.genders = torch.cat(genders)
        # all corresponding ages (uint8)
        self.ages = torch.cat(ages)
        assert len(self.full_signal) == len(self.genders) == len(self.ages)

        self.samplable_len = len(self.full_signal) - self.min_signal_len + 1

        # Masking probabilities
        prob_not_masked = 1 - config['ratio_masked']
        prob_masked = config['ratio_masked'] * (1 - (config['ratio_replaced'] + config['ratio_kept']))
        prob_replaced = config['ratio_masked'] * config['ratio_replaced']
        prob_kept = config['ratio_masked'] * config['ratio_kept']
        self.mask_probs = torch.tensor([prob_not_masked, prob_masked, prob_replaced, prob_kept])
        self.mask_cum_probs = self.mask_probs.cumsum(0)

    def __len__(self)->int:
        """
        Calculate the number of samplable elements.

        Returns:
          int: Number of samplable elements.
        """
        return self.samplable_len

    def __getitem__(self, idx:int)->tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Get every attribute of a sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            x_data (Tensor): Signal data.
            x_gender (Tensor): Gender data.
            x_age (Tensor): Age data.
            mask (Tensor): Mask of the sequence.
            masked_seq (Tensor): Masked sequence.
        """
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."

        idx += self.past_signal_len

        # Get data
        x_data = self.full_signal[idx - self.past_signal_len:idx + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        x_gender = self.genders[idx + self.window_size - 1]  # TODO: double-check
        x_age = self.ages[idx + self.window_size - 1]  # TODO: double-check
        
        # Get a random mask from given probabilities:
        mask = torch.searchsorted(self.mask_cum_probs, torch.rand(self.seq_len))

        # Get the sequence for masked sequence modeling
        masked_seq = x_data.clone()
        for seq_idx, mask_token in enumerate(mask):
            # No mask or skip mask or MASK token (which is done later)
            if mask_token in [0, 1, 3]: 
                continue
            elif mask_token == 2:
                # Replace token with replacement
                random_idx = int(torch.randint(high=len(self)-self.window_size, size=(1, )))
                masked_seq[seq_idx] = self.full_signal[random_idx: random_idx+self.window_size]
            else:
                raise RuntimeError("Issue with masks, shouldn't get a value not in {0, 1, 2, 3}")

        return x_data, x_gender, x_age, mask, masked_seq
