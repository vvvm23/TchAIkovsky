from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from .attention import MultiheadAttention
from .utils import make_attention_mask, make_causal_mask


class DecoderLayer(eqx.Module):
    attn: eqx.Module
    fc: eqx.Module
    norm1: eqx.Module
    norm2: eqx.Module
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        num_heads: int,
        mult: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype
        attn_key, fc_key = jax.random.split(key)

        if head_dim is None:
            assert dim % num_heads == 0
            head_dim = dim // num_heads

        self.attn = MultiheadAttention(
            num_heads=num_heads,
            query_size=head_dim * num_heads,
            output_size=dim,
            use_output_bias=True,
            dropout_p=dropout,
            key=attn_key,
            dtype=self.dtype,
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

    def __call__(self, x, mask, key=None):
        x = x.astype(self.dtype)
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
        dtype: jnp.dtype = jnp.float32,
    ):
        keys = jax.random.split(key, num_layers)

        self.layers = [DecoderLayer(k, dim, num_heads, head_dim=head_dim, dropout=dropout, dtype=dtype) for k in keys]

    def __call__(self, x, mask, key=None):
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

    dtype: jnp.dtype = eqx.field(static=True)
    output_dtype: jnp.dtype = eqx.field(static=True)

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
        dtype: jnp.dtype = jnp.float32,
        output_dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype
        self.output_dtype = output_dtype
        id_embeddings_key, pos_embeddings_key, decoder_key, out_key = jax.random.split(key, 4)

        self.id_embeddings = eqx.nn.Embedding(vocab_size, dim, key=id_embeddings_key)
        self.pos_embeddings = eqx.nn.Embedding(max_positions, dim, key=pos_embeddings_key)

        self.decoder = Decoder(
            decoder_key,
            dim,
            num_heads,
            num_layers,
            head_dim=head_dim,
            dropout=dropout,
            dtype=dtype,
        )

        self.norm_out = eqx.nn.LayerNorm(dim)
        self.out_head = eqx.nn.Linear(dim, vocab_size, use_bias=True, key=out_key)

    def __call__(self, input_ids, position_ids, mask, key=None):
        causal_mask = make_causal_mask(input_ids)[0]
        mask = jnp.where(mask, causal_mask, 0)

        x = jax.vmap(self.id_embeddings)(input_ids) + jax.vmap(self.pos_embeddings)(position_ids)
        x = self.decoder(x, mask, key)

        x = jax.vmap(self.norm_out)(x)
        logits = jax.vmap(self.out_head)(x)
        logits = logits.astype(self.output_dtype)
        return logits


if __name__ == "__main__":
    key = jax.random.PRNGKey(0xFF)

    x_key, model_key, key = jax.random.split(key, 3)
    x = jax.random.randint(x_key, (2, 8), 0, 10)
    position_ids = jnp.expand_dims(jnp.arange(8, dtype=int), axis=0).repeat(2, axis=0)
    mask = jnp.ones((2, 8, 8), dtype=bool)
    model = TchAIkovskyModel(model_key, 512, 8, 4, 10, 8, head_dim=64, dropout=0.1)
    model = jax.vmap(model, (0, 0, 0, None))

    y = model(x, position_ids, mask, key)

    print(y.shape)
