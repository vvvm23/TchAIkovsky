from argparse import ArgumentParser

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

from data import generate_splits, get_dataloader, get_dataset
from model import TchAIkovskyModel


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

    for ei in range(args.epochs):
        pb = tqdm.tqdm(train_loader)
        pb.set_description(f"[Epoch {ei+1}/{args.epochs}] TRAINING | Loss: ??????")
        total_loss = 0.0
        for i, batch in enumerate(pb):
            key, subkey = jax.random.split(key)
            batch = {k: v.numpy() for k, v in batch.items()}
            model, opt_state, loss = train_step(model, opt_state, batch, subkey)

            total_loss += loss.item()

            if i > 0 and i % PRINT_INTERVAL == 0:
                pb.set_description(
                    f"[Epoch {ei+1}/{args.epochs}] TRAINING | Loss: {total_loss / PRINT_INTERVAL:.4f}"
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
