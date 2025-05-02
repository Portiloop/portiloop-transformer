import random

import numpy as np
from torch.utils.data import Sampler
from wandb.sdk import Config


class RandomSampler(Sampler):
    """
    Samples elements randomly and evenly between the two classes.
    The sampling happens WITH replacement.
    __iter__ stops after an arbitrary number of iterations = batch_size_list * nb_batch

    :ivar idx_true: Indices corresponding to the 'true' group of data.
    :type idx_true: np.ndarray
    :ivar idx_false: Indices corresponding to the 'false' group of data.
    :type idx_false: np.ndarray
    :ivar nb_true: Total number of items in the 'true' group.
    :type nb_true: int
    :ivar nb_false: Total number of items in the 'false' group.
    :type nb_false: int
    :ivar length: Total number of iterations/samples the sampler will generate.
    :type length: int
    """

    def __init__(self, idx_true:np.ndarray, idx_false:np.ndarray, config:Config):
        """
        Constructor for the RandomSampler class.
        Args:
            idx_true (np.ndarray): Indices corresponding to the 'true' group of data.
            idx_false (np.ndarray): Indices corresponding to the 'false' group of data.
            config (Config): Configuration dictionary containing required settings.
        """
        super().__init__()
        self.idx_true = idx_true
        self.idx_false = idx_false
        self.nb_true = self.idx_true.size
        self.nb_false = self.idx_false.size
        self.length = config['batches_per_epoch'] * config['batch_size']

    def __iter__(self):
        """
        Iterates over a sequence of sampled indices based on probabilities and predetermined classification.

        :yield: int
            The next sampled index from either the true or false class, depending on
            the sampled classification.
        """
        cur_iter = 0
        proba = 0.5

        while cur_iter < self.length:
            cur_iter += 1
            sample_class = np.random.choice([0, 1], p=[1 - proba, proba])
            if sample_class:  # sample true
                idx_file = random.randint(0, self.nb_true - 1)
                idx_res = self.idx_true[idx_file]
            else:  # sample false
                idx_file = random.randint(0, self.nb_false - 1)
                idx_res = self.idx_false[idx_file]

            # print('Sampled at index {}'.format(idx_res))
            yield idx_res

    def __len__(self)->int:
        """
        Returns the total number of iterations/samples the sampler will generate.

        Returns:
            int: Total number of iterations/samples the sampler will generate.
        """
        return self.length