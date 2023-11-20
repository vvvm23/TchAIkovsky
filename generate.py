import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tqdm
from loguru import logger
from miditoolkit import MidiFile

from data.tokenizer import get_pretrained_tokenizer
from model import TchAIkovskyModel
from utils import seed_others


def load_config(config_path):
    with open(config_path, mode="r") as f:
        data = f.read()

    json_dict = json.loads(data)
    return SimpleNamespace(**json_dict)


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def generate_step(model, inputs, length, key, temperature):
    logits = model(**inputs)
    logits = jnp.take(logits, length - 1, axis=0)
    # logits = logits.at[0].set(-jnp.inf)
    if temperature == 0:
        # argmax sampling
        raise NotImplementedError()

    logits = logits / temperature
    return jax.random.categorical(key, logits, axis=-1)


def generate_loop(
    model,
    initial_input,
    temperature,
    key,
    max_to_generate: Optional[int] = None,
    model_max_positions: int = 1024,
    output_generated_only: bool = False,
) -> np.array:
    real_length = initial_input.shape[0]  # TODO: rename this variable (sample_idx?)

    if output_generated_only:
        output = []
    else:
        output = initial_input.tolist()

    if max_to_generate is None:
        DEFAULT_MAX = 1000
        max_to_generate = DEFAULT_MAX

    input_length = real_length + max_to_generate
    if input_length > model_max_positions - 1:
        input_length = model_max_positions - 1

    position_ids = np.arange(input_length)
    mask = np.concatenate(
        [
            np.ones((real_length,), dtype=bool),
            np.zeros((input_length - real_length,), dtype=bool),
        ],
        axis=-1,
        dtype=bool,
    )
    input_ids = np.pad(initial_input, ((0, input_length - real_length),))

    # TODO: replace with jax loop for faster generation
    for _ in tqdm.trange(max_to_generate):
        key, subkey = jax.random.split(key)
        inputs = dict(input_ids=input_ids, position_ids=position_ids, mask=mask)
        token = generate_step(model, inputs, np.array(real_length), subkey, temperature).item()
        output.append(token)

        if real_length < input_length:
            input_ids[real_length] = token
            mask[real_length] = True
        else:
            input_ids = np.concatenate([input_ids[1:], np.array([token])], axis=-1)

        real_length = min(input_length - 1, real_length + 1)

    return np.array(output)


# tokenizes initial prompt
def tokenize_prompt(midi, tokenizer):
    return tokenizer(midi)


# returns midi version of uncondtional prompt
def unconditional_prompt():
    pass


# return midi version of chord prompt
def chord_prompt(chord):
    pass


# loads prompt MIDI file
def file_prompt(path):
    midi = MidiFile(path)
    return midi


def main(args):
    logger.info("Beginning generation script.")
    key = jax.random.PRNGKey(args.seed)
    logger.info(f"Using PRNG key {args.seed}")
    seed_others(args.seed)

    logger.info("Loading config.")
    config = load_config(args.config)

    logger.info(f"Loading tokenizer from '{args.tokenizer}'")
    tokenizer = get_pretrained_tokenizer(args.tokenizer)

    logger.info("Initialising model.")
    model = TchAIkovskyModel(
        dim=config.dim,
        num_heads=config.heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        max_positions=config.max_sequence_length,
        head_dim=config.head_dim,
        dropout=config.dropout,
        key=key,  # don't bother splitting here, as we will load from checkpoint anyway
        dtype=jnp.bfloat16 if config.use_bf16 else jnp.float32,
    )

    if args.checkpoint is None:
        logger.warning("Did not specify checkpoint! Using randomly initialised weights.")
        logger.warning(
            "If you do not intend to use random weights, please specifiy --checkpoint when excecuting script."
        )
    else:
        # TODO: this does not restore things like activation functions and such
        logger.info(f"Loading model from '{args.checkpoint}'")
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        # model = checkpointer.restore(Path(args.checkpoint).resolve(), item=eqx.filter(model, eqx.is_inexact_array))
        loaded_model = checkpointer.restore(
            Path(args.checkpoint).resolve(),
            item=eqx.filter([model], eqx.is_inexact_array),
        )[0]

        # hack to deal with optax not serialising some parameters
        # TODO: change to use eqx serialisation
        model = jax.tree_map(lambda x, y: x if (y is None) else y, model, loaded_model)
        del loaded_model

        logger.info("Model loaded!")

    num_parameters = jax.tree_util.tree_reduce(lambda s, p: s + (p.size if eqx.is_inexact_array(p) else 0), model, 0)
    logger.info(f"Model has {num_parameters:,} parameters.")

    if args.prompt_mode == "unconditional":
        raise NotImplementedError("Unconditional generation not implemented yet.")
    elif args.prompt_mode == "chord":
        raise NotImplementedError("Chord-conditioned generation not implemented yet.")
    elif args.prompt_mode == "file":
        logger.info(f"Loading prompt file '{args.prompt_midi}'")
        midi = file_prompt(args.prompt_midi)
        logger.info(midi)

    logger.info("Tokenising prompt.")
    start_tokens = np.array(tokenize_prompt(midi, tokenizer))[0]
    logger.info(f"Tokenised prompt is of length {start_tokens.shape[0]}")

    if args.prompt_midi_slice is not None:
        logger.info(f"Slicing starting prompt to {args.prompt_midi_slice} tokens")
        start_tokens = start_tokens[: args.prompt_midi_slice]

    if start_tokens.shape[0] >= config.max_sequence_length:
        logger.warning("Tokenised prompt provided is longer than maximum length supported by model.")
        logger.warning("Terminating")
        return

    logger.info("Beginning generation loop")
    generated_tokens = generate_loop(
        model,
        start_tokens,
        args.temperature,
        key,
        max_to_generate=args.max_to_generate,
        model_max_positions=config.max_sequence_length - 1,
        output_generated_only=args.output_generated_only,
    )

    logger.info(f"Generated MIDI has {len(generated_tokens)} tokens.")
    logger.info("Decoding generated MIDI")
    generated_midi = tokenizer(np.expand_dims(generated_tokens, axis=0))

    if args.output_file is None:
        output_dir = Path("generated_midis")
        output_dir.mkdir(exist_ok=True)
        args.output_file = output_dir / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mid")

    logger.info(f"Saving generated MIDI to '{args.output_file}'")
    generated_midi.dump(args.output_file)
    logger.info("Done")


def validate_args(args):
    if args.config is None:
        raise ValueError("Must specify --config!")
    if args.prompt_mode not in ["unconditional", "chord", "file"]:
        raise ValueError(f"Invalid prompt mode 'args.prompt_mode'!")
    if args.prompt_mode == "file" and args.prompt_midi is None:
        raise ValueError("Must specify --prompt_midi if `--prompt_mode file` specified!")
    if args.prompt_mode == "chord" and args.prompt_chord is None:
        raise ValueError("Must specify --prompt_chord if `--prompt_mode chord` specified!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0xFF,
        help="Random seed used for PRNG key initialisation.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to JSON config file generated by train.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to directory containing trained model parameters. If not specified, generate from random weights.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="file",
        help="Specifies the prompting mode. Currently just 'file' is supported.",
    )
    parser.add_argument(
        "--prompt_midi",
        type=str,
        default=None,
        help="Path to the MIDI file to use as a prompt.",
    )
    parser.add_argument(
        "--prompt_midi_slice",
        type=int,
        default=None,
        help="Specifies the number of tokens to take from the start of the prompt file for use as a prompt.",
    )
    parser.add_argument(
        "--prompt_chord",
        type=str,
        default=None,
        help="Specifies a starting chord or chord progression.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to BPE tokenizer to use when tokenizing the prompt and detokenizing the sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. This scales the logits and can be used to influence 'how creative' the model is when generating.",
    )
    parser.add_argument(
        "--max_to_generate",
        type=int,
        default=None,
        help="Max number of tokens to generate.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save result to. Defaults to creating a new file in generated_midis.",
    )
    parser.add_argument(
        "--output_generated_only",
        action="store_true",
        help="Only save the generated content to file, do not prepend the prompt.",
    )
    args = parser.parse_args()
    validate_args(args)
    main(args)
