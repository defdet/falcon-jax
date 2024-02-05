import jax

from ..falcon import Falcon, FalconModel
from ..falcon.attention import Attention
from ..falcon.decoder import Decoder
from .shard_array import shard_array

sharding_mp = Falcon(
    model=FalconModel(
        embedding=...,
        decoder=DecoderBlock(
            layer_norm=LayerNorm(input_norm=..., input_norm_bias=...),
            attention=Attention(query_key_value=2, dense=2),
            dense_h_to_4h=2,
            dense_4h_to_h=1,
        ),
        norm=LayerNorm(input_norm=..., input_norm_bias=...),
    ),
    lm_head=...,
)

def shard_model_params(params: Falcon) -> Falcon:
    return jax.tree_map(shard_array, params, sharding_mp)
