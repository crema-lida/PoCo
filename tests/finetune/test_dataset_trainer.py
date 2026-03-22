import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'finetune'))

from trainer import MultiTaskTrainer, PolymerDataset


class PolymerDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmpdir.name) / 'ExampleDataset'
        self.dataset_dir.mkdir(parents=True)

        (self.dataset_dir / 'data.csv').write_text(
            'smiles,prop_a,prop_b,prop_c\n'
            'C,1.0,10.0,100.0\n'
            'CC,2.0,,200.0\n'
            'CCC,3.0,30.0,300.0\n'
        )
        (self.dataset_dir / 'dataset.yaml').write_text(
            'name: Example\n'
            'data_file: data.csv\n'
            'smiles_column: smiles\n'
            'categories:\n'
            '  Group 1:\n'
            '    - prop_a\n'
            '    - prop_b\n'
            '  Group 2:\n'
            '    - prop_c\n'
            'transforms:\n'
            '  log10:\n'
            '    - prop_b\n'
            '  logm1: []\n'
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def build_dataset(self):
        with patch('trainer.dataset.parallel_canonicalize', side_effect=lambda values: values.tolist()), \
             patch('trainer.dataset.get_encoder') as mock_get_encoder:
            mock_get_encoder.return_value = lambda smiles, **kwargs: np.arange(
                len(smiles) * 2, dtype=np.float32
            ).reshape(len(smiles), 2)
            return PolymerDataset.from_dir(
                str(self.dataset_dir),
                encoder_path='dummy-encoder',
                pooling='cls',
                concat_last_layers=2,
            )

    def test_from_dir_loads_yaml_metadata_and_encodes_once(self):
        dataset = self.build_dataset()

        self.assertEqual(dataset.name, 'Example')
        self.assertEqual(dataset.all_properties, ['prop_a', 'prop_b', 'prop_c'])
        self.assertEqual(dataset.properties, ['prop_a', 'prop_b', 'prop_c'])
        self.assertEqual(len(dataset), 2)
        np.testing.assert_array_equal(dataset.row_indices, np.array([0, 2]))
        np.testing.assert_array_equal(dataset.log10_idx, np.array([False, True, False]))
        np.testing.assert_array_equal(dataset.fingerprints, np.array([[0, 1], [4, 5]], dtype=np.float32))

    def test_subset_filters_rows_and_tasks_without_reencoding(self):
        dataset = self.build_dataset()

        task_subset = dataset.subset(tasks=['prop_a'])
        self.assertEqual(task_subset.properties, ['prop_a'])
        self.assertEqual(len(task_subset), 3)
        np.testing.assert_array_equal(task_subset.log10_idx, np.array([False]))
        np.testing.assert_array_equal(task_subset.fingerprints, np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32))

        row_subset = task_subset.subset(indices=[1])
        self.assertEqual(len(row_subset), 1)
        np.testing.assert_array_equal(row_subset.row_indices, np.array([1]))
        np.testing.assert_array_equal(row_subset.fingerprints, np.array([[2, 3]], dtype=np.float32))
        np.testing.assert_array_equal(row_subset.values, np.array([[2.0]]))

class MultiTaskTrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmpdir.name) / 'ExampleDataset'
        self.dataset_dir.mkdir(parents=True)

        (self.dataset_dir / 'data.csv').write_text(
            'smiles,prop_a,prop_b\n'
            'C,1.0,10.0\n'
            'CC,2.0,20.0\n'
            'CCC,3.0,30.0\n'
        )
        (self.dataset_dir / 'dataset.yaml').write_text(
            'name: Example\n'
            'data_file: data.csv\n'
            'smiles_column: smiles\n'
            'categories:\n'
            '  Group 1:\n'
            '    - prop_a\n'
            '    - prop_b\n'
            'transforms:\n'
            '  log10: []\n'
            '  logm1: []\n'
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def build_dataset(self):
        with patch('trainer.dataset.parallel_canonicalize', side_effect=lambda values: values.tolist()), \
             patch('trainer.dataset.get_encoder') as mock_get_encoder:
            mock_get_encoder.return_value = lambda smiles, **kwargs: np.ones((len(smiles), 2), dtype=np.float32)
            return PolymerDataset.from_dir(str(self.dataset_dir), encoder_path='dummy-encoder')

    def test_train_serial_does_not_mutate_base_dataset_properties(self):
        dataset = self.build_dataset()
        output_dir = Path(self.tmpdir.name) / 'output'
        trainer = MultiTaskTrainer(
            model_config={'input_dim': 2, 'hidden_dim': 4, 'num_hidden_layers': 1, 'dropout': 0.1},
            dataset=dataset,
            tasks=['prop_a', 'prop_b'],
            serial=True,
            output_dir=str(output_dir),
        )

        calls = []

        def fake_train_impl(current_output_dir, current_tasks):
            calls.append((current_output_dir, list(current_tasks), list(trainer.dataset.properties)))
            task = current_tasks[0]
            trainer.results['R2'][task].append(0.1)
            trainer.results['MAE'][task].append(0.2)

        trainer._train_impl = fake_train_impl

        with patch.object(MultiTaskTrainer, 'save_results') as mock_save_results:
            trainer.train()

        self.assertEqual(dataset.properties, ['prop_a', 'prop_b'])
        self.assertEqual(calls[0][1], ['prop_a'])
        self.assertEqual(calls[1][1], ['prop_b'])
        self.assertEqual(calls[0][2], ['prop_a', 'prop_b'])
        self.assertTrue((output_dir / 'config.json').is_file())
        config = json.loads((output_dir / 'config.json').read_text())
        self.assertEqual(config['dataset']['name'], 'Example')
        self.assertEqual(config['tasks'], ['prop_a', 'prop_b'])
        mock_save_results.assert_called_once()


if __name__ == '__main__':
    unittest.main()
