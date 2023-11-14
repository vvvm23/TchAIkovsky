from argparse import ArgumentParser

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
import wandb
from pathlib import Path
from datetime import datetime
import json

from data import generate_splits, get_dataloader, get_dataset
from model import TchAIkovskyModel

import orbax.checkpoint as ocp

def prepare_batch(batch, key=None):
    input_ids = jnp.copy(batch["input_ids"][:, :-1])
    labels = jnp.copy(batch["input_ids"][:, 1:])
    position_ids = jnp.expand_dims(jnp.arange(labels.shape[-1]), 0).repeat(
        labels.shape[0], 0
    )
    mask = jnp.asarray(batch["attention_mask"][:, :-1], dtype=bool)

    keys = jax.random.split(key, input_ids.shape[0]) if key is not None else None
    return dict(input_ids=input_ids, position_ids=position_ids, mask=mask), labels, keys


def loss_fn(model, batch, labels, keys=None):
    if keys is None:
        logits = jax.vmap(model)(**batch)
    else:
        logits = jax.vmap(model)(**batch, key=keys)
    accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()
    return (
        optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean(),
        accuracy,
    )


def create_train_step(model, optimiser):
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        batch, labels, keys = prepare_batch(batch, key)
        (loss, _), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, batch, labels, keys
        )

        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    @eqx.filter_jit
    def eval_step(model, batch):
        batch, labels, _ = prepare_batch(batch)
        loss, accuracy = loss_fn(model, batch, labels)

        return loss, accuracy

    return train_step, eval_step, opt_state


def wandb_init(args):
    return wandb.init(project="tchaikovsky", config=vars(args))


PRINT_INTERVAL = 10


def main(args):
    key = jax.random.PRNGKey(args.seed)

    model_key, key = jax.random.split(key)
    model = TchAIkovskyModel(
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        max_positions=args.max_sequence_length,
        head_dim=args.head_dim,
        dropout=args.dropout,
        key=model_key,
    )

    optimiser = optax.adamw(learning_rate=args.learning_rate)
    train_step, eval_step, opt_state = create_train_step(model, optimiser)

    dataset = get_dataset(
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        subset=args.subset_proportion,
    )
    val_dataset, train_dataset = generate_splits(
        dataset, (args.val_proportion, 1.0 - args.val_proportion)
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint_root = Path('checkpoints')
    exp_root = checkpoint_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    exp_root.mkdir(parents=True)

    with open(exp_root / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

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

                    wandb.log(
                        {"train": {"loss": total_loss / PRINT_INTERVAL}}, step=num_steps
                    )
                    total_loss = 0.0

            pb = tqdm.tqdm(train_loader)
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
                        "loss": total_val_loss / (i+1),
                        "accuracy": 100 * total_val_accuracy / (i+1),
                    }
                },
                step=num_steps,
            )

            ckptr.save((exp_root / f"checkpoint-{ei+1:03}.eqx").resolve(), eqx.filter(model, eqx.is_inexact_array))
    except BaseException as e:
        print("Caught exception.. waiting for checkpointer to finish..")
        ckptr.wait_until_finished()
        raise e

    print("Training complete.. waiting for checkpointer to finish")
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
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--subset_proportion", type=float, default=1.0)
    parser.add_argument("--val_proportion", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--min_sequence_length", type=int, default=128)
    args = parser.parse_args()

    main(args)
