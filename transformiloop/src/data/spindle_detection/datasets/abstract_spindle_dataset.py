from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class AbstractSpindleDataset(ABC, Dataset):
    """
    Abstract base class for spindle datasets.

    :ivar data: The data storage containing samples and spindle
        information.
    :type data: Any
    :ivar indices: A mapping or list of indices corresponding to
        available dataset samples.
    :type indices: Any
    :ivar threshold: The threshold value used to determine if a
        sample qualifies as a spindle.
    :type threshold: float
    :ivar window_size: Size of the sample window used in spindle
        evaluations.
    :type window_size: int
    """
    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def is_spindle(self, idx: int) -> bool:
        """
        Verify if a sample is a spindle.

        Args:
            idx (int): Index of the sample to check.

        Returns:
            bool: True if the sample is a spindle, False otherwise.
        """
        assert 0 <= idx <= len(self), f"Index out of range ({idx}/{len(self)})."
        idx = self.indices[idx]
        return True if (self.data[3][idx + self.window_size - 1] > self.threshold) else False

