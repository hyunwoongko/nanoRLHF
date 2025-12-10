from copy import copy
from enum import Enum, auto
from itertools import count

from nanorlhf.kernels.utils.vllm import KVCACHE_BLOCK_SIZE
from nanorlhf.nanovllm.utils.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class FinishReason(Enum):
    LENGTH = auto()
    STOP = auto()
    NOT_FINISHED = auto()

    def __str__(self):
        if self == FinishReason.LENGTH:
            return "length"
        elif self == FinishReason.STOP:
            return "stop"
        else:
            return "not_finished"


class Sequence:
    counter = count()
    block_size = KVCACHE_BLOCK_SIZE

    def __init__(self, token_ids, sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1] if len(token_ids) > 0 else None
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_p = sampling_params.top_p
        self.finish_reason = FinishReason.NOT_FINISHED

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, item):
        return self.token_ids[item]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.temperature,
            self.top_p,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.temperature,
            self.top_p,
        ) = state[:-1]

        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
