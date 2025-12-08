from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nanorlhf.kernels.utils.vllm import reset_context, set_context, paged_flash_attention_forward
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import Config


class ModelRunner:
    """A simplified model runner.

    This implementation focuses on correctness over performance. It batches
    incoming sequences, runs them through a Hugging Face causal LM, and samples
    the next token for each sequence using temperature-based sampling.
    """

    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        assert torch.cuda.is_available()
        if "nanoRLHF" not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS["nanoRLHF_paged"] = paged_flash_attention_forward

        self.config = config
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=getattr(config.hf_config, "torch_dtype", None),
            attn_implementation="nanoRLHF_paged",
        ).to(self.device)

        self.model.eval()
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.block_size = config.kvcache_block_size
        self.max_blocks = config.num_kvcache_blocks
        self._allocate_kv_cache()

    def _prepare_batch(self, seqs: Iterable[Sequence]):
        tensors: List[torch.Tensor] = [torch.tensor(seq.token_ids, dtype=torch.long) for seq in seqs]
        max_len = max(t.size(0) for t in tensors)
        input_ids = []
        attention_masks = []
        lengths = []
        for t in tensors:
            lengths.append(t.size(0))
            pad_len = max_len - t.size(0)
            padded = F.pad(t, (0, pad_len), value=self.pad_token_id)
            mask = torch.cat(
                [
                    torch.ones_like(t, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )
            input_ids.append(padded)
            attention_masks.append(mask)
        batch_ids = torch.stack(input_ids).to(self.device)
        batch_mask = torch.stack(attention_masks).to(self.device)
        positions = [torch.arange(length, device=self.device) for length in lengths]
        pos_padded = [F.pad(pos, (0, max_len - pos.numel())) for pos in positions]

        return batch_ids, batch_mask, pos_padded, lengths

    def _sample(self, logits: torch.Tensor, seqs: Iterable[Sequence]):
        token_ids = []
        for logit, seq in zip(logits, seqs):
            scaled = logit / seq.temperature
            probs = torch.softmax(scaled, dim=-1)
            token = torch.multinomial(probs, 1).item()
            token_ids.append(token)
        return token_ids

    def _prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=self.device)
        return block_tables

    def _prepare_context(self, seqs: list[Sequence], lengths: list[int], is_prefill: bool):
        block_tables = self._prepare_block_tables(seqs)
        if is_prefill:
            cu_seqlens_q = [0]
            slot_mapping = []
            for seq, length in zip(seqs, lengths):
                cu_seqlens_q.append(cu_seqlens_q[-1] + length)
                for pos in range(length):
                    block_idx = pos // self.block_size
                    offset = pos % self.block_size
                    block_id = seq.block_table[block_idx]
                    slot_mapping.append(block_id * self.block_size + offset)
            cu = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device)
            slot = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
            max_len = max(lengths)
            set_context(
                True,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_len,
                max_seqlen_k=max_len,
                slot_mapping=slot,
                block_tables=block_tables,
            )
            return {
                "cu_seq_lens_q": cu,
                "cu_seq_lens_k": cu,
            }

        # decode path
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            block_id = seq.block_table[-1]
            slot_mapping.append(block_id * self.block_size + seq.last_block_num_tokens - 1)
            context_lens.append(len(seq))
        slot = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
        set_context(False, slot_mapping=slot, context_lens=context_lens_t, block_tables=block_tables)
        return {}

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # The "is_prefill" flag is kept for API compatibility but does not change
        # the minimal runner behaviour yet.
        if not seqs:
            return []
        input_ids, attention_mask, position_ids, lengths = self._prepare_batch(seqs)
        attn_kwargs = self._prepare_context(seqs, lengths, is_prefill)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=torch.stack(position_ids).to(self.device),
            **attn_kwargs,
        )
        logits = outputs.logits
        last_token_logits = [logits[idx, length - 1] for idx, length in enumerate(lengths)]
        stacked = torch.stack(last_token_logits)
        token_ids = self._sample(stacked, seqs)
        if is_prefill:
            for seq in seqs:
                seq.num_cached_tokens = len(seq)
        reset_context()
        return token_ids

    def _allocate_kv_cache(self):
        hf_config = self.config.hf_config
        num_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        cache_shape = (
            2,
            hf_config.num_hidden_layers,
            self.max_blocks,
            self.block_size,
            num_heads,
            head_dim,
        )
        dtype = getattr(hf_config, "torch_dtype", torch.float16) or torch.float16
        self.kv_cache = torch.zeros(cache_shape, device=self.device, dtype=dtype)

        layer_id = 0
        for module in self.model.modules():
            if "Attention" in module.__class__.__qualname__:
                if not (hasattr(module, "k_cache") and hasattr(module, "v_cache")):
                    k_cache = v_cache = torch.tensor([])
                    setattr(module, "k_cache", k_cache)
                    setattr(module, "v_cache", v_cache)
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
                if layer_id >= hf_config.num_hidden_layers:
                    break
