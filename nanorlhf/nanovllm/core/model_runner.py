import torch
from transformers import AutoModelForCausalLM

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.kernels.utils.vllm import set_context, reset_context, get_context
from nanorlhf.nanotron import TensorParallel, MPU
from nanorlhf.nanovllm.core.sampler import Sampler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


@nanoray.actor
class ModelRunner:
    def __init__(self, config: NanoVLLMConfig, rank: int):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.device = torch.device("cuda")
        self.kv_cache = None

        model = AutoModelForCausalLM.from_pretrained(
            config.model, torch_dtype=getattr(config.hf_config, "torch_dtype", torch.float16)
        )
        if config.tensor_parallel_size > 1:
            mpu = MPU(
                rank=rank,
                local_rank=rank,
                world_size=config.tensor_parallel_size,
                local_world_size=config.tensor_parallel_size,
                host=config.host,
                port=config.port,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                tensor_parallel_size=config.tensor_parallel_size,
                backend=config.backend,
                seed=config.seed,
            )
            self.model = TensorParallel(model, mpu)
            self.model.parallelize()
        else:
            self.model = model.to(self.device)

        self.model = patch_kernel(self.model, use_paged_attention=True)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()

    def get_config(self):
        # This is necessary to pass `config` to Scheduler.
        # https://discuss.ray.io/t/how-can-i-get-attribute-of-a-actor/7153
        return self.config

    def warmup_model(self):
        dtype = getattr(self.config.hf_config, "torch_dtype", torch.float16)
        for module in self.model.modules():
            if "Attention" in module.__class__.__qualname__:
                if not (hasattr(module, "key_cache") and hasattr(module, "value_cache")):
                    key_cache = value_cache = torch.tensor([], device=self.device, dtype=dtype)
                    setattr(module, "key_cache", key_cache)
                    setattr(module, "value_cache", value_cache)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.max_memory_allocated()
        current = torch.cuda.memory_allocated()

        num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = hf_config.num_attention_heads
        num_kv_heads = num_kv_heads // config.tensor_parallel_size

        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        dtype = getattr(hf_config, "torch_dtype", torch.float16)
        itemsize = torch.tensor([], dtype=dtype).dtype.itemsize
        block_bytes = (
            2  # key, value
            * hf_config.num_hidden_layers
            * config.kvcache_block_size
            * num_kv_heads
            * head_dim
            * itemsize
        )

        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "Not enough GPU memory for KV cache."
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=dtype,
        )

        layer_id = 0
        for module in self.model.modules():
            if "Attention" in module.__class__.__qualname__:
                module.key_cache = self.kv_cache[0, layer_id]
                module.value_cache = self.kv_cache[1, layer_id]
                layer_id += 1
                if layer_id >= hf_config.num_hidden_layers:
                    break

    def prepare_block_tables(self, seqs):
        if any(len(seq.block_table) == 0 for seq in seqs):
            return None

        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs):
        lengths = [len(s) for s in seqs]
        seqlens_q = []
        seqlens_k = []
        packed_ids = []
        packed_pos = []
        slot_mapping = []
        block_tables = self.prepare_block_tables(seqs)

        for seq in seqs:
            length = len(seq)
            prefix = int(seq.num_cached_tokens)
            assert 0 <= prefix <= length

            suffix_ids = list(seq.token_ids[prefix:length])
            suffix_len = len(suffix_ids)
            assert suffix_len > 0, "suffix_len must be > 0 for prefill"

            packed_ids.extend(suffix_ids)
            packed_pos.extend(range(prefix, length))
            seqlens_q.append(suffix_len)
            seqlens_k.append(length)

            if block_tables is not None:
                assert len(seq.block_table) > 0, "block_tables is not None but seq.block_table is empty"
                for pos in range(prefix, length):
                    block_idx = pos // self.block_size
                    assert block_idx < len(seq.block_table), (
                        f"block_table too short: block_idx={block_idx}, len(block_table)={len(seq.block_table)}, "
                        f"pos={pos}, length={length}, block_size={self.block_size}"
                    )
                    page_id = seq.block_table[block_idx]
                    offset = pos % self.block_size
                    slot_mapping.append(page_id * self.block_size + offset)

        if block_tables is None:
            seqlens_k = seqlens_q[:]

        cu_q = [0]
        cu_k = [0]
        for lq in seqlens_q:
            cu_q.append(cu_q[-1] + lq)
        for lk in seqlens_k:
            cu_k.append(cu_k[-1] + lk)

        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_seqlen_q = int(max(seqlens_q))
        max_seqlen_k = int(max(seqlens_k))

        input_ids = torch.tensor([packed_ids], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        position_ids = torch.tensor([packed_pos], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        attention_mask = None

        if len(slot_mapping) == 0:
            slot_mapping = torch.empty((0,), dtype=torch.int32, device=self.device)
        else:
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(lengths, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            True,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        return input_ids, position_ids, attention_mask

    def prepare_decode(self, seqs):
        input_ids, position_ids = [], []
        slot_mapping, context_lens = [], []

        for seq in seqs:
            length = len(seq)
            input_ids.append(seq.last_token)
            position_ids.append(length - 1)
            context_lens.append(length)
            offset_in_block = (length - 1) % self.block_size
            slot_mapping.append(seq.block_table[-1] * self.block_size + offset_in_block)

        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        position_ids = torch.tensor(position_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping, context_lens, block_tables)
        return input_ids, position_ids, attention_mask

    @torch.inference_mode()
    def run(self, seqs, is_prefill):
        if is_prefill:
            input_ids, position_ids, attention_mask = self.prepare_prefill(seqs)
        else:
            input_ids, position_ids, attention_mask = self.prepare_decode(seqs)
            input_ids = input_ids.unsqueeze(1)
            position_ids = position_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits

        if is_prefill:
            context = get_context()
            last_pos = context.cu_seqlens_q[1:].to(logits.device, dtype=torch.int64) - 1
            logits_for_sampling = logits[0, last_pos, :]
        else:
            logits_for_sampling = logits[:, -1, :]

        next_tokens = self.sampler.sample(seqs, logits_for_sampling)
        reset_context()
        return next_tokens

