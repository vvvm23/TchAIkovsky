import jax
import optax
from argparse import ArgumentParser

from model import TchAIkovskyModel
from data import get_dataset, get_dataloader, generate_splits


def loss_fn(model, batch, key):
    input_ids = jax.numpy.copy(batch['ids'][:-1])
    targets = jax.numpy.copy(batch['ids'][1:])
    logits = jax.vmap(model)(input_ids, batch['position_ids'], batch['attention_mask'])
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

def create_train_step(model, optimiser):
    opt_state = optimiser.init(model)

    @jax.jit
    def train_step(model, opt_state, batch, key):
        loss, grads = jax.value_and_grad(loss_fn)

        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)

        return model, opt_state, loss

    return train_step, opt_state

def main(args):
    key = jax.random.PRNGKey(0xff)
    dataset = get_dataset()
    val_dataset, train_dataset = generate_splits(dataset, (0.1, 0.9))

    train_loader = get_loader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

    model_key, key = jax.random.split(key)
    model = TchAIkovskyModel(model_key, 
        dim=1024, 
        num_heads=16, 
        num_layers=8,
        vocab_size=10_000,
        max_positions=1024
        head_dim=64,
        dropout=0.1,
    )

    optimiser = optax.adamw(learning_rate=1e-3)
    train_step, opt_state = create_train_step(model, optimiser)

    for batch in train_loader:
        key, subkey = jax.random.split(key)
        batch = {k: v.numpy() for k, v in batch.items()}
        model, opt_state, loss = train_step(model, opt_state, batch, subkey)
        print(loss)

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
