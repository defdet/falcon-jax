from typing import NamedTuple

class ModelConfig(NamedTuple):
    d_ff: int
    head_dim: int
    d_model: int
    num_heads: int
    n_layers: int
    layer_norm_epsilon: float
    token_id_bos: int
    token_id_eos: int
    token_id_pad: int
    vocab_size: int
    dropout_rate: float | None
    return_kv_cache: bool

model_config_falcon_7B = ModelConfig(
    d_ff=18176,
    head_dim=64,
    d_model=4544,
    num_heads=71,
    n_layers=32,
    layer_norm_epsilon=1e-5,
    token_id_bos=11,
    token_id_eos=11,
    token_id_pad=11,
    vocab_size=65024,
    dropout_rate=0.1,
    return_kv_cache=False,
)

