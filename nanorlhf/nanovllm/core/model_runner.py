import torch
from transformers import AutoModelForCausalLM, modeling_utils

from nanorlhf import nanoray
from nanorlhf.kernels.utils.vllm import (
    set_context,
    reset_context,
    get_context,
    paged_flash_attention_forward,
)
from nanorlhf.nanotron import TensorParallel, MPU
from nanorlhf.nanovllm.core.sampler import Sampler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import Config


@nanoray.actor
class ModelRunner:
    def __init__(self, config: Config, rank: int):
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

        if "nanoRLHF_paged" not in modeling_utils.ALL_ATTENTION_FUNCTIONS:
            modeling_utils.ALL_ATTENTION_FUNCTIONS["nanoRLHF_paged"] = paged_flash_attention_forward
        model.config._attn_implementation = "nanoRLHF_paged"

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
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs):
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)

        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.config.eos

        input_ids, attention_masks, position_ids = [], [], []
        cu_seqlens_q, cu_seqlens_k, slot_mapping = [0], [0], []
        block_tables = None
        for seq in seqs:
            length = len(seq)
            input_ids.append(list(seq.token_ids) + [pad_id] * (max_len - length))
            attention_masks.append([1] * length + [0] * (max_len - length))
            position_ids.append(list(range(max_len)))
            seqlen_q = length - seq.num_cached_tokens
            seqlen_k = length
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            if not seq.block_table:  # warmup
                continue
            for pos in range(length):
                block_idx = pos // self.block_size
                page_id = seq.block_table[block_idx]
                offset = pos % self.block_size
                slot = page_id * self.block_size + offset
                slot_mapping.append(slot)

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        attention_mask = torch.tensor(attention_masks, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        position_ids = torch.tensor(position_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(lengths, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, slot_mapping, context_lens, block_tables)
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
            lengths = context.context_lens
            bsz, seqlen, vocab_size = logits.shape
            assert lengths.numel() == bsz
            last_logits = []
            for _b in range(bsz):
                length = int(lengths[_b].item())
                last_logits.append(logits[_b, length - 1])
            logits_for_sampling = torch.stack(last_logits, dim=0)
        else:
            logits_for_sampling = logits[:, -1, :]

        next_tokens = self.sampler.sample(seqs, logits_for_sampling)
        reset_context()
        return next_tokens
