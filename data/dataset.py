import torch
from typing import Tuple

from datasets import load_dataset
from pathlib import Path


def get_dataset():
    # TODO: need to process this into smaller chunks (of about 1024, suppose)
    # whilst keeping each chunk valid
    # TODO: remove any extra columns
    # TODO: also return position ids and mask, but that can potentially be done elsewhere
    # see https://huggingface.co/docs/datasets/v1.2.0/processing.html#augmenting-the-dataset
    ds = load_dataset("json", data_files="tokenized_dataset/merged.json")
    ds = ds.set_format('pt')
    ds = ds.remove_columns("ids_bpe_encoded")
    return ds


def generate_splits(dataset, splits: Tuple[float, float]):
    return dataset.train_test_split(
        test_size=splits[0], train_size=splits[1], shuffle=False
    )

def get_dataloader(dataset, **dataloader_kwargs):
    return torch.utils.data.DataLoader(
        dataset, **dataloader_kwargs
    )
