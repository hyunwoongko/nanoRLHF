from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.amp import custom_fwd, custom_bwd
from torch.nn.modules.loss import _Loss

from nanorlhf.nanotron.distributed.collectives import Collectives
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


class VocabParallelCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx: Any, vocab_parallel_logits: torch.Tensor, targets: torch.Tensor, mpu: MPU):
        collectives = Collectives(mpu, mode=ParallelMode.TENSOR)

        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_max = collectives.all_reduce(logits_max, op=dist.ReduceOp.MAX)
        normalized_logits = vocab_parallel_logits - logits_max.unsqueeze(-1)

        partition_vocab_size = normalized_logits.size(-1)
        local_rank = mpu.get_local_rank(ParallelMode.TENSOR)
        vocab_start_idx = local_rank * partition_vocab_size
        vocab_end_idx = vocab_start_idx + partition_vocab_size

        target_mask = (targets < vocab_start_idx) | (targets >= vocab_end_idx)
        masked_targets = targets.clone() - vocab_start_idx
        masked_targets[target_mask] = 0

        logits_2d = normalized_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_targets.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size(0), device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(targets)
        predicted_logits[target_mask] = 0.0
        predicted_logits = collectives.all_reduce(predicted_logits)

        exp_logits = torch.exp(normalized_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1)
        sum_exp_logits = collectives.all_reduce(sum_exp_logits)

        loss = torch.log(sum_exp_logits) - predicted_logits
        exp_logits.div_(sum_exp_logits.unsqueeze(-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        grad_input = softmax
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        grad_input.mul_(grad.unsqueeze(-1))
        return grad_input, None, None


class VocabParallelCrossEntropyLoss(_Loss):
    def __init__(self, reduce_mean: bool = True, ignore_index: int = -100, mpu: Optional[MPU] = None):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.ignore_index = ignore_index
        self.mpu = mpu

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = VocabParallelCrossEntropyFunction.apply(logits, targets, self.mpu)
        loss[targets == self.ignore_index] = 0.0
        if self.reduce_mean:
            loss = loss.sum() / (targets != self.ignore_index).sum()
        return loss


def maybe_vocab_parallel_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mpu: MPU):
    if mpu.get_world_size(ParallelMode.TENSOR) > 1:
        loss_fn = VocabParallelCrossEntropyLoss(mpu=mpu)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(logits, labels)
