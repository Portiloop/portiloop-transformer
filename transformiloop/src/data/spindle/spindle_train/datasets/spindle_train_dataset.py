import random
import time

import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset


class SpindleTrainDataset(Dataset):
    """
    Represents a PyTorch Dataset for spindle train data.
    :ivar config: dicturation dictionary containing key details such as ``window_size``,
        ``seq_len``, and ``seq_stride`` that determine signal processing and windowing behavior.
    :type config: dict
    :ivar window_size: Size of the sliding window used for segmenting signals into smaller chunks.
    :type window_size: int
    :ivar seq_len: Length of the sequence, representing how many windows are concatenated.
    :type seq_len: int
    :ivar seq_stride: Stride value determining the overlap between consecutive sequence windows.
    :type seq_stride: int
    :ivar past_signal_len: The length of past signal required before the last window in a sequence.
    :type past_signal_len: int
    :ivar min_signal_len: The minimum signal length required to process any sequence, based
        on the ``past_signal_len`` and ``window_size``.
    :type min_signal_len: int
    :ivar full_signal: Concatenated tensor of all signals from every subject, prepared for processing.
    :type full_signal: torch.Tensor
    :ivar full_labels: Concatenated tensor of all labels corresponding to the signals, with indices
        indicating where specific labels appear.
    :type full_labels: torch.Tensor
    :ivar spindle_labels_iso: List of indices for spindle labels categorized as "isolated."
    :type spindle_labels_iso: list[int]
    :ivar spindle_labels_first: List of indices for spindle labels categorized as "first."
    :type spindle_labels_first: list[int]
    :ivar spindle_labels_train: List of indices for spindle labels categorized as "train."
    :type spindle_labels_train: list[int]
    """

    def __init__(self, subjects: list[str], data: dict[str, dict[str, int | str | ndarray | Tensor]],
                 labels: dict[str, dict[str, list[int | str]]], config: dict):
        """
        This class takes in a list of subjects, a path to the MASS directory
        and reads the files associated with the given subjects as well as the sleep stage annotations
        Args:
            subjects (list[str]): A list of subjects to be used for training.
            data (dict[str, dict[str, int | str | ndarray | Tensor]]): A dictionary containing the pretraining dataset.
            labels (dict[str, dict[str, list[int]]]): A dictionary containing the sleep stage annotations.
            config (dict): A configuration object containing details such as window size, sequence length, and stride.
        """
        super().__init__()
        self.config = config
        self.window_size = config['window_size']
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']

        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + self.window_size
        # Get the sleep stage labels
        full_signal = []
        full_labels = []
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
            for index, (onset, offset, l) in enumerate(
                    zip(labels[subject]['onsets'], labels[subject]['offsets'], labels[subject]['labels_num'])):

                # Some of the spindles in the dataset overlap with the previous spindle
                # If that is the case, we need to make sure that the onset is at least the offset of the previous spindle
                if onset < labels[subject]['offsets'][index - 1]:
                    onset = labels[subject]['offsets'][index - 1]
                label[onset:offset] = l
                # Make a separate list with the indexes of all the spindle labels so that sampling is easier
                to_add = list(range(accumulator + onset, accumulator + offset))
                assert offset < len(
                    signal), f"Offset {offset} is greater than the length of the signal {len(signal)} for subject {subject}"
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
            full_labels.append(label)
            full_signal.append(signal)
            del data[subject], signal, label

        # Concatenate the full signal and the full labels into one continuous tensor
        self.full_signal = torch.cat(full_signal)
        self.full_labels = torch.cat(full_labels)
        # Shuffle the spindle labels
        start = time.time()
        random.shuffle(self.spindle_labels_iso)
        random.shuffle(self.spindle_labels_first)
        random.shuffle(self.spindle_labels_train)
        end = time.time()
        print(f"Shuffling took {end - start} seconds")
        print(
            f"Number of spindle labels: {len(self.spindle_labels_iso) + len(self.spindle_labels_first) + len(self.spindle_labels_train)}")

    @staticmethod
    def get_labels() -> list[str]:
        """
        Get the labels for the spindle train dataset.
        Returns:
            list[str]: A list of labels for the spindle train dataset.
        """
        return ['non-spindle', 'isolated', 'first', 'train']

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """
        Returns a tuple containing the signal and the label for the given index.
        Args:
            index (int): The index of the signal and label to return.
        Returns:
            tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]: A tuple containing the signal and the label for the given index.
        """
        # Get data
        index = index + self.past_signal_len
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)
        label = self.full_labels[index + self.window_size - 1]
        # Make sure that the last index of the signal is the same as the label
        label = label.type(torch.long)
        assert label in [1, 2], f"Label {label} is not 1 or 2"
        return signal, label - 1

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.full_signal) - self.window_size