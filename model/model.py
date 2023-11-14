from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from .utils import make_causal_mask, make_attention_mask

class DecoderLayer(eqx.Module):
    attn: eqx.Module
    fc: eqx.Module
    norm1: eqx.Module
    norm2: eqx.Module

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        num_heads: int,
        mult: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        attn_key, fc_key = jax.random.split(key)

        if head_dim is None:
            assert dim % num_heads == 0
            head_dim = dim // num_heads

        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=head_dim * num_heads,
            output_size=dim,
            use_output_bias=True,
            dropout_p=dropout,
            key=attn_key,
        )
        self.norm1 = eqx.nn.LayerNorm(dim)

        self.fc = eqx.nn.MLP(
            dim,
            dim,
            width_size=dim * mult,
            depth=1,
            activation=jax.nn.silu,
            key=fc_key,
        )
        self.norm2 = eqx.nn.LayerNorm(dim)

    def __call__(self, x, mask, key = None):
        attn_norm = jax.vmap(self.norm1)(x)

        attn_output = self.attn(attn_norm, attn_norm, attn_norm, mask, key=key, inference=key is None)
        fc_output = jax.vmap(self.fc)(jax.vmap(self.norm2)(x))

        return x + attn_output + fc_output


class Decoder(eqx.Module):
    layers: List[eqx.Module]

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        num_heads: int,
        num_layers: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        keys = jax.random.split(key, num_layers)

        self.layers = [
            DecoderLayer(k, dim, num_heads, head_dim=head_dim, dropout=dropout)
            for k in keys
        ]

    def __call__(self, x, mask, key = None):
        for layer in self.layers:
            key, subkey = jax.random.split(key) if (key is not None) else (None, None)
            x = layer(x, mask, subkey)

        return x


class TchAIkovskyModel(eqx.Module):
    id_embeddings: eqx.Module
    pos_embeddings: eqx.Module
    decoder: eqx.Module
    norm_out: eqx.Module
    out_head: eqx.Module

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        max_positions: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        key: PRNGKey = None,
    ):
        id_embeddings_key, pos_embeddings_key, decoder_key, out_key = jax.random.split(
            key, 4
        )

        self.id_embeddings = eqx.nn.Embedding(vocab_size, dim, key=id_embeddings_key)
        self.pos_embeddings = eqx.nn.Embedding(max_positions, dim, key=pos_embeddings_key)

        self.decoder = Decoder(
            decoder_key, dim, num_heads, num_layers, head_dim=head_dim, dropout=dropout
        )

        self.norm_out = eqx.nn.LayerNorm(dim)
        self.out_head = eqx.nn.Linear(dim, vocab_size, use_bias=True, key=out_key)

    def __call__(self, input_ids, position_ids, mask, key = None):
        causal_mask = make_causal_mask(input_ids)[0]
        mask = jnp.where(~mask, 0, causal_mask)

        x = jax.vmap(self.id_embeddings)(input_ids) + jax.vmap(self.pos_embeddings)(position_ids)
        x = self.decoder(x, mask, key)

        x = jax.vmap(self.norm_out)(x)
        logits = jax.vmap(self.out_head)(x)
        return logits


if __name__ == "__main__":
    key = jax.random.PRNGKey(0xFF)

    x_key, model_key, key = jax.random.split(key, 3)
    x = jax.random.randint(x_key, (2, 8), 0, 10)
    position_ids = jnp.expand_dims(jnp.arange(8, dtype=int), axis=0).repeat(
        2, axis=0
    )
    mask = jnp.ones((2, 8, 8), dtype=bool)
    model = TchAIkovskyModel(model_key, 512, 8, 4, 10, 8, head_dim=64, dropout=0.1)
    model = jax.vmap(model, (0, 0, 0, None))

    y = model(x, position_ids, mask, key)

    print(y.shape)
