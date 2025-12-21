from collections import deque

from nanorlhf.nanovllm.core.block_manager import BlockManager
from nanorlhf.nanovllm.core.sequence import SequenceStatus, FinishReason
from nanorlhf.nanovllm.utils.config import NanoVLLMConfig


class Scheduler:

    def __init__(self, config: NanoVLLMConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
        )
        self.waiting = deque()
        self.running = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, sequence):
        self.waiting.append(sequence)

    def schedule(self):
        scheduled_sequences = []
        num_sequences = 0
        num_batched_tokens = 0
        while self.waiting and num_sequences < self.max_num_seqs:
            sequence = self.waiting[0]
            if num_batched_tokens + len(sequence) > self.max_num_batched_tokens or not self.block_manager.can_allocate(sequence):
                break
            num_sequences += 1
            self.block_manager.allocate(sequence)
            num_batched_tokens += len(sequence) - sequence.num_cached_tokens
            sequence.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(sequence)
            scheduled_sequences.append(sequence)
        if scheduled_sequences:
            return scheduled_sequences, True

        while self.running and num_sequences < self.max_num_seqs:
            sequence = self.running.popleft()
            while not self.block_manager.can_append(sequence):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(sequence)
                    break
            else:
                num_sequences += 1
                self.block_manager.may_append(sequence)
                scheduled_sequences.append(sequence)
        assert scheduled_sequences
        self.running.extendleft(reversed(scheduled_sequences))
        return scheduled_sequences, False

    def preempt(self, sequence):
        # pause the sequence and deallocate its blocks.
        # but this sequence has higher priority than other waiting sequences,
        # so we put it to the front of the waiting queue. (waiting.appendleft)
        sequence.status = SequenceStatus.WAITING
        self.block_manager.deallocate(sequence)
        self.waiting.appendleft(sequence)

    def postprocess(self, sequences, generated_token_ids):
        # add newly generated tokens to sequences
        for sequence, generated_token_id in zip(sequences, generated_token_ids):
            sequence.append_token(generated_token_id)
            finished = False
            if not sequence.ignore_eos and generated_token_id == self.eos:
                # if the generated token is eos token, we finish this sequence.
                sequence.finish_reason = FinishReason.STOP
                finished = True
            elif sequence.num_completion_tokens >= sequence.max_tokens:
                # and if the sequence reaches max_tokens, we also finish it.
                sequence.finish_reason = FinishReason.LENGTH
                finished = True
            if finished:
                sequence.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(sequence)
                self.running.remove(sequence)
