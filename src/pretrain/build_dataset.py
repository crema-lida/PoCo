import sys
from pathlib import Path
from datasets import load_dataset

path = Path(__file__).resolve().parent
sys.path.append(path.parent.as_posix())

from utils import augment_smiles


def build_dataset(
    dataset_path: str,
    tokenizer,
    n_views: int = 2,
):
    dataset = load_dataset(
        path='csv',
        data_files={'train': dataset_path},
        column_names=['SMILES'],
    )['train']

    config = dict(
        truncation=True,
        max_length=384,
        return_token_type_ids=False,
        return_attention_mask=False,
    )

    def transform(batch):
        raw_smiles = batch['SMILES']
        views = [[] for _ in range(n_views)]
        for raw in raw_smiles:
            anchors = []
            for i in range(n_views):
                aug = augment_smiles(
                    base=raw,
                    anchor=anchors,
                    translate=True,
                    multiply=True,
                    permute=True,
                )
                anchors.append(aug)
                views[i].append(aug)

        return {
            f'input_ids_{i}': tokenizer(view, **config)['input_ids']
            for i, view in enumerate(views)
        }

    dataset.set_transform(transform)
    return dataset
