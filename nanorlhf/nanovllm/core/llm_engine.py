from dataclasses import fields
from time import perf_counter
from typing import Iterable, List, Sequence as SeqType, Tuple

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanorlhf.nanovllm.core.scheduler import Scheduler
from nanorlhf.nanovllm.core.sequence import Sequence
from nanorlhf.nanovllm.core.model_runner import ModelRunner
from nanorlhf.nanovllm.utils.config import Config
from nanorlhf.nanovllm.utils.sampling_params import SamplingParams


class LLMEngine:
    """A minimal LLM engine that drives scheduling and model execution."""

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Ensure eos and kv cache defaults are populated before other components use them.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if config.eos == -1:
            config.eos = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
        if config.num_kvcache_blocks == -1:
            # Provide a conservative default so the scheduler has space to operate.
            config.num_kvcache_blocks = max(1, config.max_num_batched_tokens // config.kvcache_block_size)

        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config, self.tokenizer)

    def add_request(self, prompt: str | SeqType[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _collect_outputs(self, seqs: Iterable[Sequence]) -> List[Tuple[int, list[int]]]:
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = self._collect_outputs(seqs)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        progress = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        while not self.is_finished():
            start = perf_counter()
            step_outputs, num_tokens = self.step()
            if use_tqdm and progress is not None:
                elapsed = max(perf_counter() - start, 1e-6)
                if num_tokens > 0:
                    progress.set_postfix({"Prefill": f"{int(num_tokens / elapsed)}tok/s"})
                else:
                    progress.set_postfix({"Decode": f"{int(-num_tokens / elapsed)}tok/s"})
            for seq_id, token_ids in step_outputs:
                outputs[seq_id] = token_ids
                if progress is not None:
                    progress.update(1)

        ordered_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        decoded = [self.tokenizer.decode(token_ids) for token_ids in ordered_outputs]
        if progress is not None:
            progress.close()
        return [
            {"text": text, "token_ids": token_ids}
            for text, token_ids in zip(decoded, ordered_outputs)
        ]