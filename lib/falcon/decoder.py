from functools import partial

import jax
from jax import Array
import jax.random as rand

from ..rand_utils import split_key_nullable
from ..tree_utils import stack_leaves
from .ModelConfig import ModelConfig
from .decoder_block import DecoderBlock, DecoderBlock as Decoder, forward_decoder_block
from .kv_cache import KVCache
from .rotary_embedding import RotaryValues

@partial(jax.jit, static_argnames=('model_config',))
def forward_decoder(params: Decoder, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    def inner(state, input_):
        key, seq = state
        params, kv_cache = input_
        key, subkey = split_key_nullable(key)
        seq, kv_cache = forward_decoder_block(params, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=subkey, model_config=model_config)
        return (key, seq), kv_cache

    (key, seq), kv_cache = jax.lax.scan(inner, (key, seq), (params, kv_cache))
    return seq, kv_cache
