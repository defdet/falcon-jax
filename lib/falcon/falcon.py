from functools import partial
import math
from typing import Any, NamedTuple

import jax
from jax import Array
import jax.random as rand

from .ModelConfig import ModelConfig
from .kv_cache import KVCache
from .falcon_model import FalconModel, forward_falcon_model
from .rotary_embedding import RotaryValues

class Falcon(NamedTuple):
    model: FalconModel
    lm_head: Any  # Array

@partial(jax.jit, static_argnames=('model_config'))
def forward_falcon(params: Falcon, seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    outputs, kv_cache = forward_falcon_model(params.model, seq, qk_mask, rotary_values=rotary_values, kv_cache=kv_cache, key=key, model_config=model_config)
    logits = outputs @ params.lm_head
    return logits, kv_cache
