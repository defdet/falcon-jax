# JAX implementation of the Falcon model.
This is an implementation of [Falcon model](https://arxiv.org/abs/2311.16867) in JAX using functional approach for improved perfomance.
The project is inspired by https://github.com/ayaka14732/llama-2-jax. A very large amount of that project's code has been reused. Newely implemented features are:

- [x] [Model architecture](lib/falcon/)
    - [x] [Layer Norm](lib/falcon/layer_norm.py)
    - [x] [Parallel Attention and Multi-Query Attention](lib/falcon/attention.py)
- [x] Training
    - [x] [Parameter freezing](lib/train.ipynb)
- [ ] Generation
    - [ ] Early stopping
