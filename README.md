# TchAIkovsky
Using [JAX](https://github.com/google/jax) +
[Equinox](https://github.com/patrick-kidger/equinox) to generate expressive
piano performances in a MIDI format.

## Installation

Simply create a Python environment and install requirements:
```shell
# tested in python 3.11
python -m venv venv
. venv/bin/activate

pip install -r requirements.txt
```

## Usage

Before training, we need to prepare a dataset of MIDI files. For my
experiments, I used the [GiantMIDI dataset from Bytedance
research](https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md).
Extract this and the MIDI files held therein to a directory. Then, execute the
following on the directory containing the extracted MIDI files:
```shell
python -m data.tokenize_dataset \
    --midis_dir <midi_dir> \
    --out_dir <output_dir> \
    --out_tokenizer <output_tokenizer_path>
```
See `python -m data.tokenize_dataset --help` for other configuration options.

The above script will output a tokenized dataset to `output_dir` and a BPE
trained tokenizer to `output_tokenizer_path`

To begin training with default options, run `python train.py --dataset
<output_dir>`. See `python train.py` for other configuration options.

During training, checkpoints will be saved to `checkpoints` along with a
configuration file. After training, the checkpoint and configuration can be
used to sample from the model like so:

```shell
python generate.py \
    --config <path_to_config_json> \
    --checkpoint <path_to_checkpoint_dir> \
    --tokenizer <output_tokenizer_path> \
    --prompt_midi <path_to_prompt_midi>
```
Where `path_to_prompt_midi` is the path to a single-track MIDI file that is
used as a prompt.

For long prompt files, you may want to slice the file to limit the amount of
prompt the model receives. To do so, add the flag `--prompt_midi_slice
<num_tokens>`.

To limit the length of the generation, add the flag `--max_to_generate
<num_tokens>`.

To output the generated content only (dropping the prompt) add the flag
`--output_generated_only`.

To see further configuration options, execute `python generate.py --help`

Generated files will be saved to `generated_midis`.

### Architecture

The architecture of this model is a relatively simple Transformer architecture
using learned positional embeddings (up to 1024 positions) and parallel
(GPTJ-style) attention blocks.

This results in a relatively small model of approximately 35M parameters.

### Tokenizer

The MIDI tokenizer in use is the [REMI](https://dl.acm.org/doi/10.1145/3394171.3413671) tokenizer implemented in the [MidiTok](https://github.com/Natooz/MidiTok) library.


### Pretrained Models

`TODO: add this`
