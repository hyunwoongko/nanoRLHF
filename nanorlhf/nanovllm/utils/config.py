from dataclasses import dataclass

from transformers import AutoConfig, AutoTokenizer

from nanorlhf.kernels.flash_attn_decode.ops import KVCACHE_BLOCK_SIZE


MIN_TEMPERATURE = 1e-6


@dataclass
class NanoVLLMConfig:
    # vllm options
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 1024
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    eos: int = -1
    kvcache_block_size: int = KVCACHE_BLOCK_SIZE
    num_kvcache_blocks: int = -1
    enforce_eager: bool = False

    # distributed options
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    host: str = "127.0.0.1"
    port: int = 23333
    backend: str = "nccl"
    seed: int = 42

    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.eos = tokenizer.eos_token_id
        except Exception:
            print("[NanoVLLM] Failed to load tokenizer to get `eos_token_id`. You may need to set it manually.")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
