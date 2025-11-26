# nanorlhf
This project aims to perform RLHF training from scratch, implementing almost all core components manually except for PyTorch. Each module is a minimal, educational reimplementation of large-scale systems focusing on clarity and core concepts rather than production readiness. This includes SFT and RL training pipeline with evaluation, for training a small Qwen3 model on open-source math datasets.

## Contents

| Packages                                                                          | Description                                                  | Reference                                                                                              |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| [`nanosets`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/nanosets) | Scratch implementation of zero-copy dataset library          | [arrow](https://github.com/apache/arrow), [datasets](https://github.com/huggingface/datasets)          |
| [`nanotron`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/nanotron) | Scratch implementation of various parallelization algorithms | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [oslo](https://github.com/EleutherAI/oslo)       |
| `nanovllm`                                                                        | Scratch implementation of high performance inference engine  | [vllm](https://github.com/vllm-project/vllm), [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) |
| [`nanoray`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/nanoray)   | Scratch implementation of distributed computing engine       | [ray](https://github.com/ray-project/ray)                                                              |
| `nanoverl`                                                                        | Scratch implementation of SFT and PPO trainers               | [verl](https://github.com/volcengine/verl)                                                             |
| [`kernels`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/kernels)   | Scratch implementation of various triton kernels             | [trident](https://github.com/kakaobrain/trident)                                                       |
| `models`                                                                          | Pytorch model implementations                                | [transformers](https://github.com/huggingface/transformers)                                            |
