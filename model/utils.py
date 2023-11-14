# copied from https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html#make_causal_mask
# mask-making utility functions

from typing import Any, Callable

import jax.numpy as jnp


def make_attention_mask(
    query_input,
    key_input,
    pairwise_fn: Callable[..., Any] = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype=jnp.float32,
):
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      pairwise_fn: broadcasting elementwise comparison function
      extra_batch_dims: number of extra batch dims to add singleton axes for, none
        by default
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = pairwise_fn(
        jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2)
    )
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(x, extra_batch_dims: int = 0, dtype=jnp.float32):
    """Make a causal mask for self-attention.

    In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
    will be `[batch..., heads, len, len]` and this function will produce a
    causal mask of shape `[batch..., 1, len, len]`.

    Args:
      x: input array of shape `[batch..., len]`
      extra_batch_dims: number of batch dims to add singleton axes for, none by
        default
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        jnp.greater_equal,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype,
    )
