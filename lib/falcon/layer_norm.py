from functools import partial
from typing import NamedTuple
import jax
from jax import Array
import jax.numpy as jnp

from .ModelConfig import ModelConfig

class LayerNorm(NamedTuple):
    input_norm: Array
    input_norm_bias: Array

def check_layer_norm(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.d_model,)

@partial(jax.jit, static_argnames=('model_config',))
def forward_layer_norm(params: Array, x: Array, *, model_config: ModelConfig) -> Array:
    return (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + model_config.layer_norm_epsilon) * params.input_norm + params.input_norm_bias