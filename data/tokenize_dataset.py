#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path

from .tokenizer import get_pretrained_tokenizer, get_tokenizer, get_tokenizer_config


# TODO: later, load a config to specify options
# for now, we hard-code
def main(args):
    tokenizer_config = get_tokenizer_config(
        num_velocities=args.num_velocities,
        use_chords=args.use_chords,
        use_tempos=args.use_tempos,
        use_sustain_pedal=args.use_sustain_pedal,
    )
    tokenizer = get_tokenizer(tokenizer_config)

    midi_paths = list(Path(args.midis_dir).glob("**/*.mid"))
    assert len(midi_paths)

    data_augmentation_offsets = [2, 1, 1]
    no_bpe_out_dir = args.out_dir + "_no_bpe"

    # TODO: add back data augmentation, currently crashes without it
    # likely due to version mismatch
    tokenizer.tokenize_midi_dataset(midi_paths, no_bpe_out_dir, save_programs=False)
    # tokenizer.tokenize_midi_dataset(midi_paths, "tokenized_dataset_no_bpe", data_augment_offsets=data_augmentation_offsets, save_programs=False)

    tokenizer.learn_bpe(
        vocab_size=args.vocab_size,
        tokens_paths=list(Path(no_bpe_out_dir).glob("**/*.json")),
        start_from_empty_voc=False,
    )

    tokenizer.save_params(args.out_tokenizer)
    tokenizer.apply_bpe_to_dataset(no_bpe_out_dir, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midis_dir", type=str, default="/home/alex/datasets/GiantMIDI-PIano/midis")
    parser.add_argument("--out_dir", type=str, default="tokenized_dataset")
    parser.add_argument("--out_tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--num_velocities", type=int, default=16)
    parser.add_argument("--use_chords", action="store_true")
    parser.add_argument("--use_tempos", action="store_true")
    parser.add_argument("--use_sustain_pedal", action="store_true")
    args = parser.parse_args()
    main(args)
