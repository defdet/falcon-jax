from functools import partial
from typing import Any, NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand

from .ModelConfig import ModelConfig
from .decoder import Decoder, check_decoder, forward_decoder, init_decoder
from .embedding import check_embedding, forward_embedding, init_embedding
from .kv_cache import KVCache
from .rms_norm import check_rms_norm, forward_rms_norm, init_rms_norm
from .rotary_embedding import RotaryValues
from .layer_norm import LayerNorm

class FalconModel(NamedTuple):
    embedding: Any  # Array
    decoder: Decoder
    norm: LayerNorm  # Array

@partial(jax.jit, static_argnames=('model_config'))
def forward_falcon_model(params: FalconModel, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    assert isinstance(seq, Array)
    assert isinstance(qk_mask, Array)
    assert seq.dtype == jnp.uint16
    assert qk_mask.dtype == jnp.bool_
    assert model_config.d_k % 2 == 0
    assert key is None or model_config.dropout_rate is not None

    seq = forward_embedding(params.embedding, seq)
    seq, kv_cache = forward_decoder(params.decoder, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=key, model_config=model_config)
    seq = forward_rms_norm(params.norm, seq, model_config=model_config)
    return seq, kv_cache
