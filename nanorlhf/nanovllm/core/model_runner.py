from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.kernels.utils.vllm import set_context, reset_context, get_context
from nanorlhf.nanotron import TensorParallel, MPU, ParallelMode
from nanorlhf.nanotron.distributed.collectives import Collectives
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig, MIN_TEMPERATURE


@nanoray.remote
class ModelRunner:
    """
    ModelRunner is responsible for running inference on a language model
    using tensor parallelism and CUDA graph optimizations.

    Args:
        config (NanoVLLMConfig): Configuration for the model runner.
        rank (int): The global rank of this model runner.
        actor_config: Optional configuration for actor models.
    """
    def __init__(self, config: NanoVLLMConfig, rank: int, actor_config=None):
        self.config = config
        self.rank = rank
        self.block_size = int(config.kvcache_block_size)
        self.device = torch.device("cuda")
        self.kv_cache = None

        self.max_graph_batch_size = min(int(config.max_num_seqs), 512)
        self.graph_batch_size_list = self.make_graph_batch_size_list(self.max_graph_batch_size)
        self.tensor_parallel_size = int(config.tensor_parallel_size)

        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if actor_config is not None:
            actor_data_parallel_size = actor_config.data_parallel_size
            actor_tensor_parallel_size = actor_config.tensor_parallel_size
            actor_pipeline_parallel_size = actor_config.pipeline_parallel_size
            global_world_size = (config.tensor_parallel_size * config.data_parallel_size) + (
                actor_config.data_parallel_size
                * actor_config.tensor_parallel_size
                * actor_config.pipeline_parallel_size
            )
        else:
            actor_data_parallel_size = actor_tensor_parallel_size = actor_pipeline_parallel_size = 0
            global_world_size = config.tensor_parallel_size * config.data_parallel_size

        self.mpu = MPU(
            rank=rank,
            local_rank=rank,
            world_size=global_world_size,
            local_world_size=global_world_size,
            data_parallel_size=actor_data_parallel_size,
            pipeline_parallel_size=actor_pipeline_parallel_size,
            tensor_parallel_size=actor_tensor_parallel_size,
            rollout_tensor_parallel_size=config.tensor_parallel_size,
            rollout_data_parallel_size=config.data_parallel_size,
            host=config.host if actor_config is None else actor_config.host,
            port=config.port if actor_config is None else actor_config.port,
            backend=config.backend if actor_config is None else actor_config.backend,
            seed=config.seed if actor_config is None else actor_config.seed,
        )

        self.model = TensorParallel(model, self.mpu, is_rollout=True)
        self.model.parallelize()

        self.is_first_tensor_parallel_rank = (self.tensor_parallel_size <= 1) or (
            (self.mpu.get_local_rank(ParallelMode.ROLLOUT_TENSOR) == 0)
        )

        # paged attention kernel patch
        self.model = patch_kernel(self.model, use_paged_attention=True)
        self.warmup_model()
        self.allocate_kv_cache()

        # CUDA graph capture
        self.max_num_blocks = (int(config.max_model_len) + self.block_size - 1) // self.block_size
        self.graphs = {}
        self.graph_pool = None
        self.graph_vars = {}
        if not self.config.enforce_eager:
            self.capture_decode_cudagraphs()

    def make_graph_batch_size_list(self, max_batch_size: int) -> List[int]:
        """
        Create a list of batch size buckets for CUDA graph decoding.

        Args:
            max_batch_size (int): The maximum batch size for decoding.

        Returns:
            List[int]: A list of batch size buckets.
        """
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        output = [b for b in batch_sizes if b <= max_batch_size]
        if not output:
            return [1]
        if output[-1] != max_batch_size:
            output.append(max_batch_size)
        return output

    def select_batch_size_bucket(self, batch_size: int) -> int:
        """
        Select the appropriate batch size bucket for CUDA graph decoding.

        Args:
            batch_size (int): The current batch size.

        Returns:
            int: The selected batch size bucket.
        """
        for batch_size_cap in self.graph_batch_size_list:
            if batch_size <= batch_size_cap:
                return batch_size_cap
        return self.graph_batch_size_list[-1]

    def get_config(self):
        """
        Get the configuration of the model runner.
        This is because there's no way to pass attribute of remote actor in nanoray.

        Returns:
            NanoVLLMConfig: The configuration of the model runner.
        """
        return self.config

    def warmup_model(self):
        """
        Warm up the model by running a dummy forward pass to initialize caches.
        This helps to allocate necessary resources and optimize performance for subsequent inference calls.
        """
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
        num_sequences = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        sequences = [Sequence([0] * max_model_len) for _ in range(num_sequences)]
        self.run(sequences, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        Allocate the key-value cache for the model based on available GPU memory.
        """
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

        dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
        itemsize = torch.tensor([], dtype=dtype).dtype.itemsize
        block_bytes = 2 * hf_config.num_hidden_layers * config.kvcache_block_size * num_kv_heads * head_dim * itemsize

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

    def prepare_block_tables(self, sequences):
        """
        Prepare block tables for a list of sequences.

        Args:
            sequences (List[Sequence]): The list of sequences.

        Returns:
            torch.Tensor or None: The prepared block tables tensor or None if any sequence has an empty block table.
        """
        if any(len(sequence.block_table) == 0 for sequence in sequences):
            return None
        max_len = max(len(sequence.block_table) for sequence in sequences)
        block_tables = [sequence.block_table + [-1] * (max_len - len(sequence.block_table)) for sequence in sequences]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, sequences):
        """
        Prepare inputs for prefill step.

        Args:
            sequences (List[Sequence]): The list of sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: input_ids, position_ids, attention_mask
        """
        lengths = [len(s) for s in sequences]
        seq_lens_q = []
        seq_lens_k = []
        packed_ids = []
        packed_positions = []
        slot_mapping = []
        block_tables = self.prepare_block_tables(sequences)

        for sequence in sequences:
            length = len(sequence)
            prefix = int(sequence.num_cached_tokens)
            assert 0 <= prefix <= length

            suffix_ids = list(sequence.token_ids[prefix:length])
            suffix_length = len(suffix_ids)
            assert suffix_length > 0, "suffix_len must be > 0 for prefill"

            packed_ids.extend(suffix_ids)
            packed_positions.extend(range(prefix, length))
            seq_lens_q.append(suffix_length)
            seq_lens_k.append(length)

            if block_tables is not None:
                assert len(sequence.block_table) > 0, "block_tables is not None but sequence.block_table is empty"
                for position in range(prefix, length):
                    block_idx = position // self.block_size
                    assert block_idx < len(sequence.block_table)
                    page_id = sequence.block_table[block_idx]
                    offset = position % self.block_size
                    slot_mapping.append(page_id * self.block_size + offset)

        if block_tables is None:
            seq_lens_k = seq_lens_q[:]

        cu_seq_lens_q = [0]
        cu_seq_lens_k = [0]
        for q_length in seq_lens_q:
            cu_seq_lens_q.append(cu_seq_lens_q[-1] + q_length)
        for k_length in seq_lens_k:
            cu_seq_lens_k.append(cu_seq_lens_k[-1] + k_length)

        cu_seq_lens_q = torch.tensor(cu_seq_lens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seq_lens_k = torch.tensor(cu_seq_lens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_seq_len_q = int(max(seq_lens_q))
        max_seq_len_k = int(max(seq_lens_k))

        input_ids = torch.tensor([packed_ids], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        position_ids = torch.tensor([packed_positions], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
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
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
        )
        return input_ids, position_ids, attention_mask

    def prepare_decode(self, sequences):
        """
        Prepare inputs for decode step.

        Args:
            sequences (List[Sequence]): The list of sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: input_ids, position_ids, attention_mask
        """
        input_ids, position_ids = [], []
        slot_mapping, context_lens = [], []

        for sequence in sequences:
            assert len(sequence.block_table) > 0, "decode requires allocated block_table"
            length = len(sequence)
            input_ids.append(sequence.last_token)
            position_ids.append(length - 1)
            context_lens.append(length)
            offset_in_block = (length - 1) % self.block_size
            slot_mapping.append(sequence.block_table[-1] * self.block_size + offset_in_block)

        block_tables = self.prepare_block_tables(sequences)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        position_ids = torch.tensor(position_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping, context_lens, block_tables)
        return input_ids, position_ids, attention_mask

    def sample(self, logits, sequences):
        """
        Sample token ids from logits based on the sampling parameters of each sequence.

        Args:
            logits (torch.Tensor): The logits tensor of shape (batch_size, vocab_size).
            sequences (List[Sequence]): The list of sequences with sampling parameters.

        Returns:
            torch.Tensor: The sampled token ids of shape (batch_size,).
        """
        device = logits.device
        temperatures = torch.tensor(
            [seq.temperature for seq in sequences],
            dtype=torch.float32,
            device=device,
        )
        top_ps = torch.tensor(
            [seq.top_p for seq in sequences],
            dtype=torch.float32,
            device=device,
        ).clamp_(0.0, 1.0)

        greedy_mask = temperatures <= MIN_TEMPERATURE
        output_token_ids = torch.empty((logits.size(0),), device=device, dtype=torch.long)

        if greedy_mask.any():
            output_token_ids[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

        sampling_mask = ~greedy_mask
        if sampling_mask.any():
            idx = sampling_mask.nonzero(as_tuple=True)[0]

            logits_sampling = logits[idx].float()
            temps_sampling = temperatures[idx].unsqueeze(1)
            top_ps_sampling = top_ps[idx].unsqueeze(1)

            logits_sampling.div_(temps_sampling)

            sorted_logits, sorted_indices = torch.sort(logits_sampling, dim=-1, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_mask = cumulative_probs > top_ps_sampling
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False

            sorted_logits.masked_fill_(remove_mask, float("-inf"))

            filtered_logits = torch.empty_like(logits_sampling)
            filtered_logits.scatter_(dim=1, index=sorted_indices, src=sorted_logits)

            gumbel = -torch.empty_like(filtered_logits).exponential_(1).log_()
            sampled_token_ids = (filtered_logits + gumbel).argmax(dim=-1)
            output_token_ids[idx] = sampled_token_ids

        return output_token_ids

    @torch.inference_mode()
    def capture_decode_cudagraphs(self):
        """
        Capture CUDA graphs for decode step with different batch size buckets.
        """
        hf_config = self.config.hf_config
        vocab_size = int(getattr(hf_config, "vocab_size"))
        torch.cuda.synchronize()

        for batch_size_cap in reversed(self.graph_batch_size_list):
            input_ids = torch.zeros((batch_size_cap, 1), device=self.device, dtype=torch.int64)
            position_ids = torch.zeros((batch_size_cap, 1), device=self.device, dtype=torch.int64)
            attention_mask = torch.ones((batch_size_cap, 1), device=self.device, dtype=torch.int64)

            slot_mapping = torch.zeros((batch_size_cap,), device=self.device, dtype=torch.int32)
            context_lens = torch.ones((batch_size_cap,), device=self.device, dtype=torch.int32)
            block_tables = torch.full((batch_size_cap, self.max_num_blocks), -1, device=self.device, dtype=torch.int32)

            last_logits = torch.empty(
                (batch_size_cap, vocab_size // self.tensor_parallel_size),
                device=self.device,
                dtype=torch.bfloat16,
            )

            # warm up with eager mode
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            _ = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
            ).logits
            reset_context()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            with torch.cuda.graph(graph, pool=self.graph_pool):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits
                last_logits.copy_(logits[:, -1, :])
            reset_context()

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[batch_size_cap] = graph
            self.graph_vars[batch_size_cap] = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "slot_mapping": slot_mapping,
                "context_lens": context_lens,
                "block_tables": block_tables,
                "last_logits_out": last_logits,
            }
            torch.cuda.synchronize()

    def fill_decode_graph_vars(self, sequences: List[Sequence], bs_cap: int):
        """
        Fill the CUDA graph variables for decode step.

        Args:
            sequences (List[Sequence]): The list of sequences.
            bs_cap (int): The batch size cap for the CUDA graph.
        """
        graph_vars = self.graph_vars[bs_cap]
        batch_size = len(sequences)

        input_ids = graph_vars["input_ids"]
        position_ids = graph_vars["position_ids"]
        slot_mapping = graph_vars["slot_mapping"]
        context_lens = graph_vars["context_lens"]
        block_tables = graph_vars["block_tables"]

        block_tables.fill_(-1)
        slot_mapping.fill_(-1)
        context_lens.zero_()
        input_ids.zero_()
        position_ids.zero_()

        for i, sequence in enumerate(sequences):
            assert len(sequence.block_table) > 0, "decode requires allocated block_table"
            length = len(sequence)

            input_ids[i, 0] = int(sequence.last_token)
            position_ids[i, 0] = int(length - 1)
            context_lens[i] = int(length)
            offset_in_block = (length - 1) % self.block_size
            slot_mapping[i] = int(sequence.block_table[-1] * self.block_size + offset_in_block)
            block_table = sequence.block_table
            num_blocks = min(len(block_table), self.max_num_blocks)
            if num_blocks > 0:
                for j in range(num_blocks):
                    block_tables[i, j] = int(block_table[j])

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        return batch_size

    @torch.inference_mode()
    def run_decode_with_graph(self, sequences: List[Sequence]) -> torch.Tensor:
        """
        Run decode step using CUDA graph for a list of sequences.

        Args:
            sequences (List[Sequence]): The list of sequences.

        Returns:
            torch.Tensor: The logits for sampling of shape (batch_size, vocab_size).
        """
        batch_size = len(sequences)
        batch_size_cap = self.select_batch_size_bucket(batch_size)
        actual_batch_size = self.fill_decode_graph_vars(sequences, batch_size_cap)

        graph = self.graphs[batch_size_cap]
        graph.replay()

        output = self.graph_vars[batch_size_cap]["last_logits_out"][:actual_batch_size]
        reset_context()
        return output

    @torch.inference_mode()
    def run(self, sequences, is_prefill):
        """
        Run the model for a list of sequences, either in prefill or decode mode.

        Args:
            sequences (List[Sequence]): The list of sequences.
            is_prefill (bool): Whether to run in prefill mode.

        Returns:
            List[int]: The generated token IDs.
        """
        try:
            if is_prefill:
                # prefill
                input_ids, position_ids, attention_mask = self.prepare_prefill(sequences)
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits
                last_pos = get_context().cu_seq_lens_q[1:].to(logits.device, dtype=torch.int64) - 1
                logits_for_sampling = logits[0, last_pos, :]
            else:
                # decode
                if len(sequences) <= self.max_graph_batch_size and not self.config.enforce_eager:
                    # CUDA graph decode
                    logits_for_sampling = self.run_decode_with_graph(sequences)
                else:
                    # fallback eager decode
                    input_ids, position_ids, attention_mask = self.prepare_decode(sequences)
                    input_ids = input_ids.unsqueeze(1)
                    position_ids = position_ids.unsqueeze(1)
                    attention_mask = attention_mask.unsqueeze(1)
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    ).logits
                    logits_for_sampling = logits[:, -1, :]

            if self.tensor_parallel_size > 1:
                collectives = Collectives(self.mpu, ParallelMode.ROLLOUT_TENSOR)
                logits_for_sampling = collectives.all_gather(logits_for_sampling, dim=-1)

            if self.is_first_tensor_parallel_rank:
                return self.sample(logits_for_sampling, sequences).tolist()
            return []

        finally:
            reset_context()
