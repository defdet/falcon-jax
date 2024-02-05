from functools import partial
import math
from typing import Any, NamedTuple

import einops as op
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand

from .ModelConfig import ModelConfig
from .kv_cache import KVCache
from .rotary_embedding import RotaryValues, forward_rotary_embedding

class Attention(NamedTuple):
    q_proj: Any  # Array
    k_proj: Any  # Array
    v_proj: Any  # Array
    out_proj: Any  # Array

def check_attention(params: Attention, *, model_config: ModelConfig) -> None:
    assert isinstance(params.q_proj, Array)
    assert isinstance(params.k_proj, Array)
    assert isinstance(params.v_proj, Array)
    assert isinstance(params.out_proj, Array)

    assert params.q_proj.shape == (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k)
    assert params.k_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_k)
    assert params.v_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_v)
    assert params.out_proj.shape == (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model)

def init_attention(*, key: Array, model_config: ModelConfig) -> Attention:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    q_proj = rand.truncated_normal(key0, -upper, upper, (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k))
    k_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_k))
    v_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_v))
    out_proj = rand.truncated_normal(key3, -upper, upper, (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model))
    return Attention(q_proj, k_proj, v_proj, out_proj)

# Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
def split_heads(fused_qkv: Array, config: ModelConfig) -> Tuple[Array, Array, Array]:
    """
        Split the last dimension into (num_heads, head_dim)

        Args:
            fused_qkv (`Array`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    fused_qkv = fused_qkv.reshape(batch_size, seq_length, config.num_heads + 2, config.head_dim)
    return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

@partial(jax.jit, static_argnames=('model_config',))
def forward_attention(params: Attention, seq: Array, qk_mask: Array, rotary_values: RotaryValues, *, kv_cache: KVCache | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    batch_size, seq_len, d_model = seq.shape[0], seq.shape[1], seq.shape[2]

    head_dim, num_heads = model_config.head_dim, model_config.num_heads
    fused_qkv = op.einsum(seq, params.query_key_value, 'B S D, D E -> B S E') # Parallel attention
    q, k, v = split_heads(fused_qkv, config=model_config)

    q = q.transpose(0, 2, 1, 3) # [B, H, S, M]
    k = k.transpose(0, 2, 1, 3) # [B, 1, S, M]
    v = v.transpose(0, 2, 1, 3) # [B, 1, S, M]
    
    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    if kv_cache is not None:
        assert seq.shape[1] == 1
        k_cache, v_cache = kv_cache

        k = k_cache.at[:, :, -1:].set(k)
        v = v_cache.at[:, :, -1:].set(v)
        max_len = k.shape[2]
        
    qk = op.einsum(q, k.squeeze(1), 'B H S M, B D M -> B H S D')
    qk_before_sqrt = qk.copy()
    qk /= math.sqrt(model_config.head_dim)
    qk = jnp.where(qk_mask, qk, -100000000)
    qk = nn.softmax(qk, axis=-1)  
    qkv = op.einsum(qk, v.squeeze(1), 'B H S D, B D V -> B H S V')
    out = op.einsum(qkv, params.dense, 'B H S V, H V M -> B S M')
    kv_cache = KVCache(k, v)
    return out, kv_cache
