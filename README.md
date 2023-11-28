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
following on the directory containing the extracted MIDI files to tokenise the
dataset using [MidiTok](https://github.com/Natooz/MidiTok):
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
<output_dir>`. See `python train.py --help` for other configuration options.

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

You can also use an unconditional prompt (simply the `BOS` token) by passing
`--prompt_mode unconditional`.

To see further configuration options, execute `python generate.py --help`

Generated files will be saved to `generated_midis`.

### How to prompt your model

There are three ways I recommend prompting your model:
- Completely unconditionally, which seems to produce decent results but lacks
  any control over the outputs.
- From a simple prompt file: a MIDI file that has some basic starting elements
  you want in your generated piece. This could include things like tempo
  markings, key signatures, or even chords and chord progressions.
    - For the chords and chord progressions, you can find a bunch of MIDIs for
      these [here](https://github.com/ldrolez/free-midi-chords)
- From an existing piece, in which case the model will attempt to continue the
  piece (either from the end or from a slice specified by
  `--prompt_midi_slice`). This produces the most coherent results, but is more
  prone to directly copying from the piece, particularly when
  `--prompt_midi_slice` is high relative to the context size of the model (1024
  tokens). However, it can be interesting to see how a model continues a piece
  you are already familiar with, such as by pulling MIDI files off
  [Musescore](https://musescore.com). Be aware that the model only works with
  single program MIDI files, so if there are multiple programs these will need
  to be merged.

Like most generative models on discrete tokens, you can also control the
sampling temperature using `--temperature <float>`. A general rule of thumb is
that higher values result in "more creative" outputs whereas values close to 0
are increasingly likely to just select the most probably next token each
generation step. Suitable values vary greatly depending on the prompt and the
result you want, but would recommend in the range of `0.5 - 2.0` and maybe
trying out `0.0` to see what happens.

### Architecture

The architecture of this model is a relatively simple Transformer architecture
using learned positional embeddings (up to 1024 positions) and parallel
(GPTJ-style) attention blocks.

### Tokenizer

The MIDI tokenizer in use is the [REMI](https://dl.acm.org/doi/10.1145/3394171.3413671) tokenizer implemented in the [MidiTok](https://github.com/Natooz/MidiTok) library.

### Pretrained Models

Some pretrained checkpoints trained on the GiantMIDI dataset.
- [101M Parameter Model, No Chord Identification in Tokenizer](https://drive.google.com/drive/folders/1RnoJFdwBQazTxQOmtTSGQpyt2ooE4x3a?usp=drive_link)

GiantMIDI is a classical music, single-channel MIDI dataset. Results on out of
distribution prompts may vary.
