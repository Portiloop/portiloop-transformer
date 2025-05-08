import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import ndarray

from transformiloop.src.data.spindle.spindle_detection.datasets.abstract_spindle_dataset import AbstractSpindleDataset


class SignalDataset(AbstractSpindleDataset):
    """
    Handles signal dataset preparation for a sleep-related machine learning task.

    :ivar fe: Sampling frequency of the signal data.
    :vartype fe: int
    :ivar window_size: The size of the window used for slicing data sequences.
    :vartype window_size: int
    :ivar path_file: Complete file path to the dataset file.
    :vartype path_file: pathlib.Path
    :ivar data: Preprocessed signal data loaded from the file.
    :vartype data: numpy.ndarray
    :ivar full_signal: Tensor of the full signal data.
    :vartype full_signal: torch.Tensor
    :ivar full_envelope: Tensor of the full envelope data.
    :vartype full_envelope: torch.Tensor
    :ivar seq_len: Length of signal sequences used for data samples.
    :vartype seq_len: int
    :ivar idx_stride: Stride used for generating data sequences.
    :vartype idx_stride: int
    :ivar past_signal_len: Computed length of the sequence stride for the dataset.
    :vartype past_signal_len: int
    :ivar indices: List of indices indicating valid data samples in the dataset.
    :vartype indices: list
    """
    def __init__(self, filename:str, path:str, window_size:int, fe:int, seq_len:int, seq_stride:int, list_subject:ndarray, len_segment:int):
        """
        Constructor for the SignalDataset class.

        Args:
            filename (str): Name of the dataset file.
            path (str): Path to the dataset file.
            window_size (int): Size of the window used for slicing data sequences.
            fe (int): Sampling frequency of the signal data.
            seq_len (int): Length of signal sequences used for data samples.
            seq_stride (int): Stride used for generating data sequences.
            list_subject (ndarray): List of subjects to include in the dataset.
            len_segment (int): Length of the signal segments used for training.
        """
        self.fe = fe
        self.window_size = window_size
        self.path_file = Path(path) / filename

        self.data = pd.read_csv(self.path_file, header=None).to_numpy()
        assert list_subject is not None
        used_sequence = np.hstack([range(int(s[1]), int(s[2])) for s in list_subject])
        split_data = np.array(np.split(self.data, int(len(self.data) / (len_segment + 30 * fe))))  # 115+30 = nb seconds per sequence in the dataset
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))

        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.full_envelope = torch.tensor(self.data[1], dtype=torch.float)
        self.seq_len = seq_len  # 1 means single sample / no sequence ?
        self.idx_stride = seq_stride
        self.past_signal_len = self.seq_len * self.idx_stride

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                or idx < self.past_signal_len)]  # and far enough from the beginning to build a sequence up to here
        total_spindles = np.sum(self.data[3] > 0.2)
        logging.debug(f"total number of spindles in this dataset : {total_spindles}")

    def __len__(self)->int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx:int)->tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data and label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the data and label tensors.
        """
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        assert self.data[3][idx + self.window_size - 1] >= 0, f"Bad index: {idx}."

        signal_seq = self.full_signal[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0, self.window_size, self.idx_stride)
        self.full_envelope[idx - (self.past_signal_len - self.idx_stride):idx + self.window_size].unfold(0,
                                                                                                         self.window_size,
                                                                                                         self.idx_stride)

        torch.tensor(self.data[2][idx + self.window_size - 1], dtype=torch.float)
        label = torch.tensor(self.data[3][idx + self.window_size - 1], dtype=torch.float)

        return signal_seq, label
