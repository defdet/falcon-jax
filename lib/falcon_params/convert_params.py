from jax import Array
import torch
import torch.nn as tnn
from transformers import FalconForCausalLM, FalconModel as FalconModelPt
from transformers.models.falcon.modeling_falcon import FalconAttention, FalconDecoderLayer

from ..array_utils import pt2jax
from ..falcon import Falcon, FalconModel, ModelConfig
from ..falcon.attention import Attention
from ..falcon.decoder_block import DecoderBlock
from ..tree_utils import stack_leaves

def convert_proj(x: tnn.Linear) -> Array:
    return pt2jax(x.weight.T)

def convert_attention(x: Any, *, model_config: ModelConfig) -> Attention:
    query_key_value = convert_proj(x.query_key_value)
    dense = convert_proj(x.dense)
    return Attention(query_key_value=query_key_value, dense=dense)

def convert_decoder_block(x: Any, *, model_config: ModelConfig) -> DecoderBlock:
    input_norm = pt2jax(x.input_layernorm.weight)
    attention = convert_attention(x.self_attention, model_config=model_config)
    dense_h_to_4h = convert_proj(x.mlp.dense_h_to_4h)
    dense_4h_to_h = convert_proj(x.mlp.dense_4h_to_h)
    return DecoderBlock(input_norm=input_norm, attention=attention, dense_h_to_4h=dense_h_to_4h, dense_4h_to_h=dense_4h_to_h)

def convert_falcon_model(model: FalconModelPt, *, model_config: ModelConfig) -> FalconModelPt:
    embedding = pt2jax(model.word_embeddings.weight)
    decoder = stack_leaves([convert_decoder_block(model.h[i], model_config=model_config) for i in tqdm(range(model_config.n_layers))])
    norm = pt2jax(model.ln_f.weight)
    return FalconModel(embedding=embedding, decoder=decoder, norm=norm)

with torch.no_grad():
    model_jax = convert_falcon_model(model.transformer, model_config=falcon_config)
    lm_head = convert_proj(model.lm_head)
    params = Falcon(model=model_jax, lm_head=lm_head)

def convert_falcon(model_pt: FalconForCausalLM, *, model_config: ModelConfig) -> Falcon:
    with torch.no_grad():
        model_jax = convert_falcon_model(model.transformer, model_config=falcon_config)
        lm_head = convert_proj(model.lm_head)
        params = Falcon(model=model_jax, lm_head=lm_head)
