import pathlib
import traceback
import unittest

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from transformiloop.src.data.pretraining import PretrainingDataset
from transformiloop.src.utils.configs import initialize_config, validate_config

MAX_ITER_TEST = 6

class TestPretrainingDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST")
        if not validate_config(self.config):
            raise AttributeError(
                "Error when initializing test config, check your config")

        # Get the subject list for each dataset and the data for all of them
        self.dataset_path = str(pathlib.Path(__file__).parents[1].resolve(
        ) / 'transformiloop' / 'dataset' / 'MASS_preds')

    # Test pretraining dataset
    def test_pretraining_dataset(self):
        try:
            pre_dataset = PretrainingDataset(self.dataset_path, self.config)
        except FileNotFoundError as e:
            print("FileNotFoundError:", e)
            traceback.print_exc()
            raise
        print(len(pre_dataset))
        print("Done loading dataset")
        sampler = RandomSampler(data_source=pre_dataset, replacement=True)
        pre_dl = DataLoader(
            pre_dataset, 
            batch_size=self.config['batch_size'],
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )
        print("Done building dataloader")
        counter = torch.zeros(4)

        # for index, batch in enumerate(pre_dl):
        #     signal, gender, age, mask, reps = batch
        #
        #     if index == 0:
        #         self.assertEqual(signal.shape, torch.Size([self.config['batch_size'], self.config['seq_len'], self.config['window_size']]))
        #         self.assertTrue(gender[0] == 0 or gender[0] == 1)
        #         self.assertTrue(18 <= age[0] <= 76)
        #         self.assertEqual(mask.shape, torch.Size([self.config['batch_size'], self.config['seq_len']]))
        #     elif index > MAX_ITER_TEST:
        #         break
        #     else:
        #         elems = []
        #         for elem in mask:
        #             elems.append(torch.bincount(elem, minlength=4))
        #         add = torch.stack(elems, dim=0).sum(dim=0)
        #         counter += add
        #
        # sampled = counter / sum(counter)
        # for (samp, expected) in zip(sampled.tolist(), pre_dataset.mask_probs.tolist()):
        #     self.assertAlmostEqual(samp, expected, places=1)
        print("Done whole thing")
    def tearDown(self):
        return


if __name__ == '__main__':
    unittest.main()
