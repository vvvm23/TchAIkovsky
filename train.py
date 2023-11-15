import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tqdm
import wandb
from loguru import logger

from data import generate_splits, get_dataloader, get_dataset
from model import TchAIkovskyModel
from utils import seed_others


def prepare_batch(batch, key=None):
    input_ids = jnp.copy(batch["input_ids"][:, :-1])
    labels = jnp.copy(batch["input_ids"][:, 1:])

    labels = jnp.where(labels == 0, -100, labels)
    position_ids = jnp.expand_dims(jnp.arange(labels.shape[-1]), 0).repeat(labels.shape[0], 0)
    mask = jnp.asarray(batch["attention_mask"][:, :-1], dtype=bool)

    keys = jax.random.split(key, input_ids.shape[0]) if key is not None else None
    return dict(input_ids=input_ids, position_ids=position_ids, mask=mask), labels, keys


def loss_fn(model, batch, labels, keys=None):
    if keys is None:
        logits = jax.vmap(model[0])(**batch)
    else:
        logits = jax.vmap(model[0])(**batch, key=keys)

    num_tokens = (labels != -100).sum()
    accuracy = jnp.argmax(logits, axis=-1) == labels
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    accuracy = jnp.where(labels == -100, 0, accuracy).sum() / num_tokens
    loss = jnp.where(labels == -100, 0, loss).sum() / num_tokens

    return loss, accuracy


def create_train_step(model, optimiser):
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

    # @eqx.debug.assert_max_traces(max_traces=1)
    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        batch, labels, keys = prepare_batch(batch, key)
        (loss, _), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, labels, keys)

        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    # @eqx.debug.assert_max_traces(max_traces=1)
    @eqx.filter_jit
    def eval_step(model, batch):
        batch, labels, _ = prepare_batch(batch)
        loss, accuracy = loss_fn(model, batch, labels)

        return loss, accuracy

    return train_step, eval_step, opt_state


def wandb_init(args):
    return wandb.init(
        project="tchaikovsky",
        config=vars(args),
        mode=None if args.wandb else "disabled",
    )


PRINT_INTERVAL = 10


def main(args):
    logger.info("Beginning training script.")
    key = jax.random.PRNGKey(args.seed)
    seed_others(args.seed)
    logger.info(f"Using PRNG key {args.seed}")

    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size

    assert args.batch_size % args.micro_batch_size == 0

    model_key, key = jax.random.split(key)
    logger.info("Initialising model.")
    model = TchAIkovskyModel(
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        max_positions=args.max_sequence_length,
        head_dim=args.head_dim,
        dropout=args.dropout,
        key=model_key,
        dtype=jnp.bfloat16 if args.use_bf16 else jnp.float32,
    )

    num_parameters = jax.tree_util.tree_reduce(lambda s, p: s + (p.size if eqx.is_inexact_array(p) else 0), model, 0)
    logger.info(f"Model has {num_parameters:,} parameters.")

    if args.use_bf16:
        # map all params to bf16
        logger.info("Training with bfloat16.")
        model = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if eqx.is_inexact_array(p) else p, model)

    logger.info("Initialising dataset.")
    dataset = get_dataset(
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        subset=args.subset_proportion,
    )
    val_dataset, train_dataset = generate_splits(dataset, (args.val_proportion, 1.0 - args.val_proportion))
    logger.info(f"Training set size: {len(train_dataset):,} Validation set size: {len(val_dataset):,}")

    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    logger.info("Initialising optimiser.")
    # optimiser = optax.adamw(learning_rate=args.learning_rate)
    lr = args.learning_rate
    if args.use_lr_scheduler:
        WARMUP_START_LR = 1e-7
        logger.info("Using learning rate scheduler")
        steps = args.epochs * len(train_dataset) // args.batch_size
        warmup_steps = int(steps * args.warmup_proportion)
        logger.info(f"{WARMUP_START_LR} -> {lr} (for {warmup_steps:,} steps)")
        logger.info(f"{lr} -> {args.end_learning_rate} (for {steps - warmup_steps:,} steps)")
        lr = optax.join_schedules(
            [
                optax.linear_schedule(
                    WARMUP_START_LR,
                    lr,
                    warmup_steps,
                ),
                optax.linear_schedule(
                    lr,
                    args.end_learning_rate,
                    steps - warmup_steps,
                ),
            ],
            [warmup_steps],
        )

    model = [model]
    decay_spec = jax.tree_map(lambda _: "no_decay", eqx.filter(model, eqx.is_inexact_array))
    is_decay_weight = lambda p: hasattr(p, "weight") and not hasattr(p, "num_embeddings")
    where_decay_weight = lambda m: tuple(
        p.weight for p in jax.tree_util.tree_leaves(m, is_leaf=is_decay_weight) if is_decay_weight(p)
    )
    decay_spec = eqx.tree_at(where_decay_weight, decay_spec, replace_fn=lambda _: "decay")

    optimiser = optax.chain(
        optax.clip_by_global_norm(args.global_norm),
        # optax.adamw(learning_rate=lr, weight_decay=args.weight_decay),
        optax.multi_transform(
            {
                "decay": optax.adamw(learning_rate=lr, weight_decay=args.weight_decay),
                "no_decay": optax.adamw(learning_rate=lr, weight_decay=0.0),
            },
            decay_spec,
        ),
    )

    # TODO: fix bug when adding multi step
    # optimiser = optax.MultiSteps(optimiser, args.batch_size // args.micro_batch_size)
    train_step, eval_step, opt_state = create_train_step(model, optimiser)

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint_root = Path("checkpoints")
    exp_root = checkpoint_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    exp_root.mkdir(parents=True)
    logger.info(f"Saving checkpoints and config to {exp_root}")

    with open(exp_root / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.wandb:
        logger.info("Initialising W&B")
    run = wandb_init(args)
    num_steps = 0

    try:
        for ei in range(args.epochs):
            pb = tqdm.tqdm(train_loader)
            pb.set_description(f"[Epoch {ei+1}/{args.epochs}] TRAINING | Loss: ??????")
            total_loss = 0.0
            for i, batch in enumerate(pb):
                key, subkey = jax.random.split(key)
                batch = {k: v.numpy() for k, v in batch.items()}
                model, opt_state, loss = train_step(model, opt_state, batch, subkey)

                num_steps += 1
                total_loss += loss.item()

                if i > 0 and i % PRINT_INTERVAL == 0:
                    pb.set_description(
                        f"[Epoch {ei+1}/{args.epochs}] TRAINING | Loss: {total_loss / PRINT_INTERVAL:.4f}"
                    )

                    wandb.log({"train": {"loss": total_loss / PRINT_INTERVAL}}, step=num_steps)
                    total_loss = 0.0

            pb = tqdm.tqdm(val_loader)
            total_val_loss = 0.0
            total_val_accuracy = 0.0
            for i, batch in enumerate(pb):
                batch = {k: v.numpy() for k, v in batch.items()}
                loss, accuracy = eval_step(model, batch)

                total_val_loss += loss.item()
                total_val_accuracy += accuracy.item()

                pb.set_description(
                    f"[Epoch {ei+1}/{args.epochs}] VALIDATION | Loss: {total_val_loss / (i+1):.4f}, Accuracy: {100*total_val_accuracy / (i+1):.2f}"
                )

            wandb.log(
                {
                    "val": {
                        "loss": total_val_loss / (i + 1),
                        "accuracy": 100 * total_val_accuracy / (i + 1),
                    }
                },
                step=num_steps,
            )

            logger.info(f"Saving checkpoint for epoch {ei+1}")
            ckptr.save(
                (exp_root / f"checkpoint-{ei+1:03}.eqx").resolve(),
                eqx.filter(model, eqx.is_inexact_array),
            )
    except BaseException as e:
        logger.warning("Caught exception.. waiting for checkpointer to finish..")
        ckptr.wait_until_finished()
        raise e

    logger.info("Training complete.. waiting for checkpointer to finish")
    ckptr.wait_until_finished()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0xFF)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--end_learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--global_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--subset_proportion", type=float, default=1.0)
    parser.add_argument("--val_proportion", type=float, default=0.1)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--min_sequence_length", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=None)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    main(args)
