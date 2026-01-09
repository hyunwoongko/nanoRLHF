from copy import copy
from enum import Enum, auto
from itertools import count

from nanorlhf.kernels.flash_attn_decode.ops import KVCACHE_BLOCK_SIZE
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
    """
    Represents a sequence of token IDs being processed by the LLM engine.

    Attributes:
        sequence_id (int): Unique identifier for the sequence.
        status (SequenceStatus): Current status of the sequence.
        token_ids (list): List of token IDs in the sequence.
        last_token (int): The last token ID in the sequence.
        num_tokens (int): Total number of tokens in the sequence.
        num_prompt_tokens (int): Number of prompt tokens in the sequence.
        num_cached_tokens (int): Number of cached tokens in the sequence.
        block_table (list): List of block IDs used by the sequence.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        ignore_eos (bool): Whether to ignore end-of-sequence tokens.
        top_p (float): Top-p sampling parameter.
        finish_reason (FinishReason): Reason for finishing the sequence.
    """
    counter = count()
    block_size = KVCACHE_BLOCK_SIZE

    def __init__(self, token_ids, sampling_params=SamplingParams()):
        self.sequence_id = next(Sequence.counter)
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
        """
        Returns the total number of tokens in the sequence.

        Returns:
            int: The number of tokens.
        """
        return self.num_tokens

    def __getitem__(self, item):
        """
        Get the token ID at the specified index.

        Args:
            item (int): The index of the token ID to retrieve.
        """
        return self.token_ids[item]
    
    @property
    def is_finished(self):
        """
        Check if the sequence has finished processing.

        Returns:
            bool: True if the sequence is finished, False otherwise.
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        Returns the number of completion tokens in the sequence.

        Returns:
            int: The number of completion tokens.
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """
        Returns the list of prompt token IDs.

        Returns:
            list: The prompt token IDs.
        """
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def num_cached_blocks(self):
        """
        Returns the number of cached blocks in the sequence.

        Returns:
            int: The number of cached blocks.
        """
        return self.num_cached_tokens // self.block_size

    @property
    def completion_token_ids(self):
        """
        Returns the list of completion token IDs.

        Returns:
            list: The completion token IDs.
        """
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self):
        """
        Returns the total number of blocks in the sequence.

        Returns:
            int: The number of blocks.
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        Returns the number of tokens in the last block of the sequence.

        Returns:
            int: The number of tokens in the last block.
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        Get the token IDs of the i-th block.

        Args:
            i (int): The index of the block to retrieve.

        Returns:
            list: The token IDs of the i-th block.
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id):
        """
        Append a new token ID to the sequence.

        Args:
            token_id (int): The token ID to append.
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        Get the state of the sequence for serialization.

        Returns:
            tuple: The state of the sequence.
        """
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
        """
        Set the state of the sequence from serialization.

        Args:
            state (tuple): The state of the sequence.
        """
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