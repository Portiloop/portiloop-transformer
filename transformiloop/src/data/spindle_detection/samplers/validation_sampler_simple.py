from numpy import ndarray
from torch.utils.data import Sampler

class ValidationSamplerSimple(Sampler):
    """
    ValidationSamplerSimple is a custom sampler class.

    :ivar len_max: Total number of elements in the data source.
    :type len_max: int
    :ivar data: The data source being sampled.
    :type data: Any
    :ivar dividing_factor: The interval between sampled indices.
    :type dividing_factor: int
    """
    def __init__(self, data_source:ndarray, dividing_factor:int):
        """
        Constructor for the ValidationSamplerSimple class.
        Args:
            data_source (ndarray): The data source being sampled.
            dividing_factor (int): The interval between sampled indices.
        """
        super().__init__(data_source)
        self.len_max = len(data_source)
        self.data = data_source
        self.dividing_factor = dividing_factor

    def __iter__(self):
        """
        Iterate over the data source and yield the indices at the specified interval.
        Yield:
            int: The next index in the data source.
        """
        for idx in range(0, self.len_max, self.dividing_factor):
            yield idx

    def __len__(self)->int:
        """
        Returns the number of iterations/samples the sampler will generate.

        Returns:
            int: The number of iterations/samples the sampler will generate.
        """
        return self.len_max // self.dividing_factor