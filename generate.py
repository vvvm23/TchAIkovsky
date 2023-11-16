import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import tqdm
from loguru import logger

from data.tokenizer import get_pretrained_tokenizer
from model import TchAIkovskyModel
from utils import seed_others


def load_config(config_path):
    with open(config_path, mode="r") as f:
        data = f.read()

    json_dict = json.loads(data)
    return SimpleNamespace(**json_dict)


def generate_step(model, inputs, key, temperature):
    pass


def generate_loop(model, initial_input, temperature, max_to_generate):
    pass


# tokenizes initial prompt
def tokenize_prompt(midi_path):
    pass


# returns string version of uncondtional prompt
def unconditional_prompt():
    pass


# return string version of chord prompt
def chord_prompt(chord):
    pass


# loads prompt MIDI file
def file_prompt(path):
    pass


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
        logger.info(f"Loading model from '{args.checkpoint}'")
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        # model = checkpointer.restore(Path(args.checkpoint).resolve(), item=eqx.filter(model, eqx.is_inexact_array))
        model = checkpointer.restore(Path(args.checkpoint).resolve(), item=eqx.filter([model], eqx.is_inexact_array))[0]
        logger.info("Model loaded!")
        num_parameters = jax.tree_util.tree_reduce(
            lambda s, p: s + (p.size if eqx.is_inexact_array(p) else 0), model, 0
        )
        logger.info(f"Model has {num_parameters:,} parameters.")


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
    parser.add_argument("--seed", type=int, default=0xFF)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, default="unconditional")
    parser.add_argument("--prompt_midi", type=str, default=None)
    parser.add_argument("--prompt_chord", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    args = parser.parse_args()
    validate_args(args)
    main(args)
