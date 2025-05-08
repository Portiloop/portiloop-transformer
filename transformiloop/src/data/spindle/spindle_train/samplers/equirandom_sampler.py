import random
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from transformiloop.src.data.spindle.spindle_train.datasets.spindle_train_dataset import SpindleTrainDataset


class EquiRandomSampler(Sampler):
    """
    Implements a sampling strategy for SpindleTrainDataset, enabling random
    selection of data samples based on specified class distributions. The
    class is particularly designed to support four types of sample classes
    such as non-spindle, isolated, first, and train, providing an effective
    mechanism to iterate over dataset indices in accordance with predefined
    sample class ratios.

    :ivar sample_list: A list specifying the classes to sample from,
        with default being [0, 1, 2, 3].
    :type sample_list: list[int]
    :ivar dataset: The dataset object representing SpindleTrainDataset that
        contains input signals and class-specific spindle labels.
    :type dataset: SpindleTrainDataset
    :ivar isolated_index: The current index pointer for the isolated spindle
        class.
    :type isolated_index: int
    :ivar first_index: The current index pointer for the first spindle class.
    :type first_index: int
    :ivar train_index: The current index pointer for the train spindle class.
    :type train_index: int
    :ivar max_isolated_index: The maximum index for the isolated spindle class,
        corresponding to the total available samples in this class.
    :type max_isolated_index: int
    :ivar max_first_index: The maximum index for the first spindle class,
        corresponding to the total available samples in this class.
    :type max_first_index: int
    :ivar max_train_index: The maximum index for the train spindle class,
        corresponding to the total available samples in this class.
    :type max_train_index: int
    """
    def __init__(self, dataset:SpindleTrainDataset, sample_list:list[int]=[0, 1, 2, 3]):
        """
        Constructor for the EquiRandomSampler class.
        ratio: list of ratios for each class [non-spindle, isolated, first, train]
        Args:
            dataset (SpindleTrainDataset): The dataset to sample from.
            sample_list (list[int], optional): A list of integers representing the classes to sample from. Defaults to [0, 1, 2, 3].
        """
        super().__init__()
        self.sample_list = sample_list
        self.dataset = dataset

        self.isolated_index = 0
        self.first_index = 0
        self.train_index = 0

        # Come up with a good maximum number of samples to take from the dataset
        self.max_isolated_index = len(dataset.spindle_labels_iso)
        self.max_first_index = len(dataset.spindle_labels_first)
        self.max_train_index = len(dataset.spindle_labels_train)

    def __iter__(self)->Iterator[int]:
        """
        Iterate over the dataset and yield the indices at the specified interval.

        Yield:
            int: The next index in the dataset.
        """
        while True:
            # Select a random class
            class_choice = np.random.choice(self.sample_list)

            def treat_index(specific_index: int, max_index: int) -> tuple[int, int]:
                """
                Get the next index for the given class.
                Args:
                    specific_index (int): The current index for the given class.
                    max_index (int): The maximum index for the given class.

                Returns:
                    tuple[int, int]: A tuple containing the next index for the given class and the updated index.
                """
                i = self.dataset.spindle_labels_iso[specific_index]
                specific_index += 1
                if specific_index >= max_index:
                    specific_index = 0
                assert i in self.dataset.spindle_labels_iso, "Spindle index not found in list"
                return i, specific_index

            if class_choice == 0:
                # Sample from the rest of the signal
                yield random.randint(0, len(self.dataset.full_signal) - self.dataset.min_signal_len - self.dataset.window_size)
                continue
            elif class_choice == 1:
                index, self.isolated_index = treat_index(self.isolated_index, self.max_isolated_index)
            elif class_choice == 2:
                index, self.first_index = treat_index(self.first_index, self.max_first_index)
            elif class_choice == 3:
                index, self.train_index = treat_index(self.train_index, self.max_train_index)
            else:
                continue
            yield index - self.dataset.past_signal_len - self.dataset.window_size + 1

    def __len__(self)->int:
        """
        Returns the total number of iterations/samples the sampler will generate.
        Returns:
            int: Total number of iterations/samples the sampler will generate.
        """
        return len(self.dataset.full_signal)