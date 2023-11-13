#!/usr/bin/env python

import shutil
from pathlib import Path

from .tokenizer import get_tokenizer, get_tokenizer_config, get_pretrained_tokenizer

# TODO: later, load a config to specify options
# for now, we hard-code
if __name__ == "__main__":
    tokenizer_config = get_tokenizer_config()
    tokenizer = get_tokenizer(tokenizer_config)

    midi_paths = list(
        Path("/home/alex/datasets/GiantMIDI-PIano/midis").glob("**/*.mid")
    )
    assert len(midi_paths)

    data_augmentation_offsets = [2, 1, 1]

    tokenizer.tokenize_midi_dataset(midi_paths, "tokenized_dataset_no_bpe", data_augment_offsets=data_augmentation_offsets, save_programs=False)

    tokenizer.learn_bpe(
        vocab_size=10_000,
        tokens_paths=list(Path("tokenized_dataset_no_bpe").glob("**/*.json")),
        start_from_empty_voc=False,
    )

    tokenizer.save_params("tokenizer.json")

    # tokenizer = get_pretrained_tokenizer()
    tokenizer.apply_bpe_to_dataset("tokenized_dataset_no_bpe", "tokenized_dataset")

    concat_files = list(Path("tokenized_dataset").glob("**/*.json"))
    with open(Path("tokenized_dataset", "merged.json"), mode="w") as outfile:
        for f in concat_files:
            with open(f, "r") as infile:
                shutil.copyfileobj(infile, outfile)
