import random

from torch.utils.data import Sampler


class ValidationSampler(Sampler):
    """
    ValidationSampler is a custom sampler class.

    :ivar network_stride: Defines the stride within each segment, determining how data is skipped
        while iterating through the dataset. Must always be greater than or equal to 1.
    :type network_stride: int
    :ivar seq_stride: Length of each sequence stride, defining the step size between two
        sequences within the segment.
    :type seq_stride: int
    :ivar nb_segment: Number of segments in the dataset, dictating the dataset's logical division
        into segments.
    :type nb_segment: int
    :ivar len_segment: Fixed length of segment derived from 115 seconds of data sampled at
        250 Hz. Represents the number of records each segment holds.
    :type len_segment: int
    """

    def __init__(self, seq_stride: int, nb_segment: int, network_stride: int):
        """
        Constructor for the ValidationSampler class.

        Args:
            network_stride (int): (>= 1, default: 1) divides the size of the dataset (and of the batch) by striding further than 1.
            seq_stride (int): length of each sequence stride, defining the step size between two sequences within the segment.
            nb_segment (int): number of segments in the dataset.
        """
        super().__init__()
        network_stride = int(network_stride)
        assert network_stride >= 1
        self.network_stride = network_stride
        self.seq_stride = seq_stride
        self.nb_segment = nb_segment
        self.len_segment = 115 * 250  # 115 seconds x 250 Hz

    def __iter__(self):
        """
        Iterate over the dataset and yield the indices at the specified interval.
        Yield:
            int: The next index in the dataset.
        """
        random.seed()
        batches_per_segment = self.len_segment // self.seq_stride  # len sequence = 115 s + add the 15 first s?
        cursor_segment = 0
        while cursor_segment < batches_per_segment:
            for i in range(self.nb_segment):
                for j in range(0, (self.seq_stride // self.network_stride) * self.network_stride, self.network_stride):
                    cur_idx = i * self.len_segment + j + cursor_segment * self.seq_stride
                    yield cur_idx
            cursor_segment += 1