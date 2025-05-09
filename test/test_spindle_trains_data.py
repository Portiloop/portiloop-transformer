import pathlib
import unittest

from torch.utils.data import DataLoader

from transformiloop.src.data.pretraining import read_pretraining_dataset
from transformiloop.src.data.spindle.spindle_train.datasets.spindle_train_dataset import SpindleTrainDataset
from transformiloop.src.data.spindle.spindle_train.samplers.equirandom_sampler import EquiRandomSampler
from transformiloop.src.data.spindle.spindle_train.spindle_trains import read_spindle_trains_labels
from transformiloop.src.utils.configs import initialize_config, validate_config


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST")
        if not validate_config(self.config):
            raise AttributeError("Error when initializing test config, check your config")

        self.subject_list = ['01-01-0004']
        self.MASS_dir = str(pathlib.Path(__file__).parents[1].resolve() / 'transformiloop' / 'dataset')
        self.sleep_stages = read_spindle_trains_labels(self.MASS_dir)
        self.data = read_pretraining_dataset(str(self.MASS_dir + "/" + 'MASS_preds'), patients_to_keep=self.subject_list)

    def test_reader_sleep_staging_labels(self):
        labels = read_spindle_trains_labels(self.MASS_dir)
        self.assertEqual(len(labels.keys()), 138)

    # Test pretraining dataset
    def test_sleep_Staging_dataset(self):
        dataset = SpindleTrainDataset(self.subject_list, self.data, self.sleep_stages, self.config)

        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            sampler=EquiRandomSampler(dataset),
            pin_memory=True,
            drop_last=True,
            shuffle=False
        )

        # Check that the dataloader is working and the ratio of labels is correct
        num_spindles = 0
        totals = 0
        for index, batch in enumerate(dataloader):
            if index > 1000:
                break

            signal, label = batch
            num_spindles += sum([1 for i in label if i != 0])
            totals += len(label)
        print(num_spindles/totals)
        self.assertAlmostEqual(num_spindles / totals, 0.25, delta=0.1)


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
