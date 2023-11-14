import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from argparse import ArgumentParser

from model import TchAIkovskyModel
from data import get_dataset, get_dataloader, generate_splits
import tqdm

def prepare_batch(batch, key = None):
    input_ids = jnp.copy(batch['input_ids'][:, :-1])
    labels = jnp.copy(batch['input_ids'][:, 1:])
    position_ids = jnp.expand_dims(jnp.arange(labels.shape[-1]), 0).repeat(labels.shape[0], 0)
    mask = jnp.asarray(batch['attention_mask'][:, :-1], dtype=bool)

    keys = jax.random.split(key, input_ids.shape[0]) if key is not None else None
    return dict(
        input_ids = input_ids,
        position_ids = position_ids,
        mask = mask
    ), labels, keys

def loss_fn(model, batch, labels, keys = None):
    if keys is None:
        logits = jax.vmap(model)(**batch)
    else:
        logits = jax.vmap(model)(**batch, key=keys)
    accuracy = (jnp.argmax(logits, axis=-1) == labels).mean()
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean(), accuracy

def create_train_step(model, optimiser):
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        batch, labels, keys = prepare_batch(batch, key)
        (loss, _), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, labels, keys)

        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    @eqx.filter_jit
    def eval_step(model, batch):
        batch, labels, _ = prepare_batch(batch)
        loss, accuracy = loss_fn(model, batch, labels)

        return loss, accuracy

    return train_step, eval_step, opt_state

def main(args):
    key = jax.random.PRNGKey(0xff)

    model_key, key = jax.random.split(key)
    model = TchAIkovskyModel(
        dim=1024, 
        num_heads=16, 
        num_layers=8,
        vocab_size=10_000,
        max_positions=1024,
        head_dim=64,
        dropout=0.1,
        key=model_key
    )

    optimiser = optax.adamw(learning_rate=1e-3)
    train_step, eval_step, opt_state = create_train_step(model, optimiser)

    dataset = get_dataset(subset=0.05)
    val_dataset, train_dataset = generate_splits(dataset, (0.1, 0.9))

    train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

    epochs = 5
    for ei in range(epochs):
        pb = tqdm.tqdm(train_loader)
        for batch in pb:
            key, subkey = jax.random.split(key)
            batch = {k: v.numpy() for k, v in batch.items()}
            model, opt_state, loss = train_step(model, opt_state, batch, subkey)

            pb.set_description(f"[Epoch {ei+1}/{epochs}] TRAINING | Loss: {loss:.4f}")

        pb = tqdm.tqdm(train_loader)
        total_val_loss = 0.0
        total_val_accuracy = 0.0
        for i, batch in enumerate(pb):
            batch = {k: v.numpy() for k, v in batch.items()}
            loss, accuracy = eval_step(model, batch)

            total_val_loss += loss
            total_val_accuracy += accuracy

            pb.set_description(f"[Epoch {ei+1}/{epochs}] VALIDATION | Loss: {loss / (i+1):.4f}, Accuracy: {100*accuracy / (i+1):.2f}")

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
