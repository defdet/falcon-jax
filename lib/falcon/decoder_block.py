from functools import partial
import math
import einops as op
from typing import Any, NamedTuple

import jax
from jax import Array
import jax.random as rand

from ..rand_utils import split_key_nullable
from .ModelConfig import ModelConfig
from .attention import Attention, forward_attention
from .dropout import forward_dropout
from .kv_cache import KVCache
from .layer_norm import LayerNorm, forward_layer_norm
from .rotary_embedding import RotaryValues

class DecoderBlock(NamedTuple):
    layer_norm: LayerNorm
    attention: Attention
    dense_h_to_4h: Array
    dense_4h_to_h: Array

@partial(jax.jit, static_argnames=('model_config',))
def forward_decoder_block(params: DecoderBlock, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    key0, key1, key2 = split_key_nullable(key, num=3)
    residual = seq

    attention_layernorm_out = forward_layer_norm(params.layer_norm, seq, model_config=model_config)
    attention_output, kv_cache = forward_attention(params.attention, attention_layernorm_out, qk_mask=qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, model_config=model_config)
    mlp_layernorm_out = attention_layernorm_out

    ff = jax.nn.gelu(op.einsum(mlp_layernorm_out, params.dense_h_to_4h, 'B S D, D E -> B S E'), approximate=False)
    mlp_output = op.einsum(ff, params.dense_4h_to_h, 'B S E, E D -> B S D')
    mlp_output = forward_dropout(mlp_output, key=key0, model_config=model_config)
    
    output = mlp_output + attention_output + residual
    return output, kv_cache

