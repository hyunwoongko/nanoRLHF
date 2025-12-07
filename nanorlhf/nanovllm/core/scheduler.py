from collections import deque

from nanorlhf.nanovllm.core.block_manager import BlockManager
from nanorlhf.nanovllm.core.sequence import SequenceStatus, FinishReason
from nanorlhf.nanovllm.utils.config import Config


class Scheduler:

    def __init__(self, config: Config):
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

    def add(self, seq):
        self.waiting.append(seq)

    def schedule(self):
        # prefill stage:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # first come, first served
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break  # cannot prefill more sequences in this batch
            num_seqs += 1
            self.block_manager.allocate(seq)
            # do not need to compute cached tokens
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode stage:
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # if we don't have enough blocks, we need to release some running sequences
                # but will give them higher priority in the next scheduling round.
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # we can append a new token to this sequence
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(self.running))
        # why reversed?
        # because we want to keep the original order of running sequences.
        return scheduled_seqs, False

    def preempt(self, seq):
        # pause the sequence and deallocate its blocks.
        # but this sequence has higher priority than other waiting sequences,
        # so we put it to the front of the waiting queue. (waiting.appendleft)
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs, generated_token_ids):
        # add newly generated tokens to sequences
        for seq, generated_token_id in zip(seqs, generated_token_ids):
            seq.append_token(generated_token_id)
            finished = False
            if not seq.ignore_eos and generated_token_id == self.eos:
                # if the generated token is eos token, we finish this sequence.
                seq.finish_reason = FinishReason.STOP
                finished = True
            elif seq.num_completion_tokens >= seq.max_tokens:
                # and if the sequence reaches max_tokens, we also finish it.
                seq.finish_reason = FinishReason.LENGTH
                finished = True
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
