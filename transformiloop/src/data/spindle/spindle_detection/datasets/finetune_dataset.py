import random
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
from numpy import ndarray


from transformiloop.src.data.spindle.spindle_detection.datasets.abstract_spindle_dataset import AbstractSpindleDataset


def default_modif(signal: torch.Tensor) -> torch.Tensor:
    """
    Return a modified signal by flipping the sign of a random sample.

    Args:
        signal (torch.Tensor): Input signal tensor.

    Returns:
        torch.Tensor: Modified signal tensor.
    """
    # Get one random sequence
    modified_index = random.randint(0, signal.size(0) - 1)
    new_sig = deepcopy(signal)
    new_sig[modified_index] = -signal[modified_index]
    return new_sig


class FinetuneDataset(AbstractSpindleDataset):
    """
    Custom dataset class tailored for fine-tuning a deep learning model, specifically for spindle
    recognition and related signal processing tasks.

    :ivar fe: Sampling frequency derived from the configuration.
    :type fe: int
    :ivar device: Device for computations, e.g., 'cpu' or 'cuda'.
    :type device: str
    :ivar window_size: Time window size for the processed sequences.
    :type window_size: int
    :ivar augmentation_config: dicturation for data augmentation, if any.
    :type augmentation_config: dict or None
    :ivar data: Input raw signal data.
    :type data: numpy.ndarray
    :ivar seq_len: Sequence length for the training/validation data.
    :type seq_len: int
    :ivar seq_stride: Stride for sequence generation.
    :type seq_stride: int
    :ivar past_signal_len: Length of signal history for sequence generation.
    :type past_signal_len: int
    :ivar threshold: Threshold value for data labeling.
    :type threshold: float
    :ivar label_history: Whether to include label history in samples.
    :type label_history: bool
    :ivar pretraining: Indicates if the model is in pretraining mode.
    :type pretraining: bool
    :ivar modif_ratio: Ratio of signal modifications applied during pretraining.
    :type modif_ratio: float
    :ivar signal_modif: Custom function for modifying signals during pretraining.
    :type signal_modif: Callable or None
    :ivar full_signal: Full processed signal data for indexing.
    :type full_signal: torch.Tensor
    :ivar full_labels: Full processed labels corresponding to the signal data.
    :type full_labels: torch.Tensor
    :ivar indices: List of indices of the valid samples in the dataset, based on specified
        constraints.
    :type indices: list[int]
    """


    def __init__(self, list_subject: ndarray, config: dict, data: ndarray, history: bool, augmentation_config: dict = None, device: str = None, signal_modif: Callable = None):
        """
        Constructor for the FinetuneDataset class.

        Args:
            list_subject (ndarray): List of subjects to include in the dataset.
            config (dict): dicturation object for the dataset.
            data (ndarray): Signal data.
            history (bool): Whether to include label history in samples.
            augmentation_config (dict, optional): dicturation for data augmentation, if any. Defaults to None.
            device (str, optional): Device for computations, e.g., 'cpu' or 'cuda'. Defaults to None.
            signal_modif (Callable, optional): Custom function for modifying signals during pretraining. Defaults to None.
        """
        self.fe = config['fe']
        self.device = device
        self.window_size = config['window_size']
        self.augmentation_config = augmentation_config
        self.data = data
        assert list_subject is not None
        used_sequence = np.hstack([range(int(s[1]), int(s[2])) for s in list_subject])
        split_data = np.array(np.split(self.data, int(len(self.data) / (
                    config['len_segment'] + 30 * self.fe))))  # 115+30 = nb seconds per sequence in the dataset
        split_data = split_data[used_sequence]
        self.data = np.transpose(split_data.reshape((split_data.shape[0] * split_data.shape[1], 4)))

        assert self.window_size <= len(self.data[0]), "Dataset smaller than window size."
        self.full_signal = torch.tensor(self.data[0], dtype=torch.float)
        self.full_labels = torch.tensor(self.data[3], dtype=torch.float)
        self.seq_len = 1 if config['full_transformer'] and not history else config[
            'seq_len']  # want a single sample if full transformer and not training (aka validating), else we use seq len
        self.seq_stride = config['seq_stride']
        self.past_signal_len = self.seq_len * self.seq_stride
        self.threshold = config['data_threshold']
        self.label_history = history

        # Check if we are pretrining the model
        self.pretraining = config['pretraining']
        self.modif_ratio = config['modif_ratio']
        self.signal_modif = signal_modif

        # list of indices that can be sampled:
        self.indices = [idx for idx in range(len(self.data[0]) - self.window_size)  # all possible idxs in the dataset
                        if not (self.data[3][idx + self.window_size - 1] < 0  # that are not ending in an unlabeled zone
                                or idx < self.past_signal_len  # and far enough from the beginning to build a sequence up to here
                                or (self.label_history and self.data[3][
                        idx - (self.past_signal_len - self.seq_stride) + self.window_size - 1] < 0))
                        # and not beginning in an unlabeled zone
                        ]

        total_spindles = np.sum(self.data[3] > self.threshold)
        print(f"total number of spindles in this dataset : {total_spindles}")


    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.indices)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather the data and label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the data and label tensors.
        """
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        assert self.data[3][idx + self.window_size - 1] >= 0, f"Bad index: {idx}."

        # Get data
        x_data = self.full_signal[idx - (self.past_signal_len - self.seq_stride):idx + self.window_size].unfold(0,
                                                                                                                self.window_size,
                                                                                                                self.seq_stride)

        if self.pretraining:
            if random.uniform(0, 1) < self.modif_ratio:
                x_data = self.signal_modif(x_data) if self.signal_modif is not None else default_modif(x_data)
                label = torch.tensor(1, dtype=torch.float)
            else:
                label = torch.tensor(0, dtype=torch.float)
        else:
            # Get label for the spindle recognition task
            label_unique = torch.tensor(self.data[3][idx + self.window_size - 1], dtype=torch.float)
            label = label_unique
            if self.label_history:
                # Get the label history if we want to learn from that as well.
                label_history = self.full_labels[idx - (
                            self.past_signal_len - self.seq_stride) + self.window_size - 1:idx + self.window_size].unfold(0,
                                                                                                                          1,
                                                                                                                          self.seq_stride)
                assert len(label_history) == len(x_data), f"len(label):{len(label_history)} != len(x_data):{len(x_data)}"
                assert -1 not in label_history, f"invalid label: {label_history}"
                assert label_unique == label_history[-1], f"bad label: {label_unique} != {label_history[-1]}"
                label = label_history

        assert label in [0, 1], f"Invalid label: {label}"
        label = label.type(torch.long)
        return x_data, label


