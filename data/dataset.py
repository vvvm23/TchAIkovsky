import torch
from typing import Tuple

from datasets import load_dataset
from pathlib import Path


def get_dataset():
    return load_dataset("json", data_files="tokenized_dataset/merged.json")


def generate_splits(dataset, splits: Tuple[float, float]):
    return dataset.train_test_split(
        test_size=splits[0], train_size=splits[1], shuffle=False
    )

def get_dataloader(dataset, **dataloader_kwargs):
    return torch.utils.data.DataLoader(
        dataset, **dataloader_kwargs
    )
