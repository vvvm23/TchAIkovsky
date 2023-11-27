from pathlib import Path
from typing import Tuple

import torch
from miditok.pytorch_data.collators import DataCollator
from miditok.pytorch_data.datasets import DatasetTok


def get_dataset(
    dataset_root: str = "tokenized_dataset",
    min_sequence_length=128,
    max_sequence_length=1024,
    subset: float = 1.0,
):
    files = list(Path(dataset_root).glob("**/*.json"))
    files = files[: int(len(files) * subset)]
    ds = DatasetTok(
        files,
        min_seq_len=min_sequence_length,
        max_seq_len=max_sequence_length,
        one_token_stream=False,
    )
    return ds


def generate_splits(dataset, splits: Tuple[float, float]):
    length = len(dataset)
    split_size = int(splits[0] * length)

    return torch.utils.data.Subset(dataset, range(split_size)), torch.utils.data.Subset(
        dataset, range(split_size, length)
    )


def get_dataloader(dataset, **dataloader_kwargs):
    collator = DataCollator(pad_token_id=0, shift_labels=False)
    return torch.utils.data.DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)
