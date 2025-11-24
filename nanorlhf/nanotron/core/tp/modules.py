from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from nanorlhf.nanotron.core.tp.ops import (
    tp_broadcast,
    tp_all_reduce,
    tp_all_gather,
    tp_scatter,
)
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.tracing import ModuleParallelPlan
from nanorlhf.nanotron.utils.wrapping import tag_module


class ParallelizableModuleMixin:
    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU):
        raise NotImplementedError

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU):
        raise NotImplementedError

    @classmethod
    def convert_to_parallel_module(cls, plan: ModuleParallelPlan, mpu: MPU, **kwargs):
        module = plan.module
        original_module_class = module.__class__
        module.__class__ = cls
        module.original_module_class = original_module_class

        module.mpu = mpu
        module.world_size = mpu.get_world_size(ParallelMode.TENSOR)
        module.rank = mpu.get_local_rank(ParallelMode.TENSOR)

        for key, val in kwargs.items():
            setattr(module, key, val)

        return module

    @classmethod
    def restore_to_original_module(cls, plan: ModuleParallelPlan, **kwargs):
        module = plan.module
        module.__class__ = module.original_module_class
        del module.original_module_class
        del module.mpu
        del module.world_size
        del module.rank

        for key, val in kwargs.items():
            if val is None:
                if hasattr(module, key):
                    delattr(module, key)
            else:
                setattr(module, key, val)

        return module


class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int):
        first_idx = rank * per_partition_vocab_size
        last_idx = first_idx + per_partition_vocab_size - 1
        return first_idx, last_idx

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int):
        assert global_vocab_size % world_size == 0, (
            f"Global vocab size ({global_vocab_size}) must be divisible by " f"the world size ({world_size})."
        )
        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank)


class VocabParallelEmbedding(nn.Embedding, ParallelizableModuleMixin):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        mpu: Optional[MPU] = None,
    ):
        self.mpu = mpu
        self.world_size = mpu.get_world_size(ParallelMode.TENSOR)
        self.rank = mpu.get_local_rank(ParallelMode.TENSOR)
        self.vocab_start_idx, self.vocab_end_idx = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings, mpu.get_local_rank(ParallelMode.TENSOR), self.world_size
        )
        super().__init__(
            num_embeddings=num_embeddings // self.world_size,
            embedding_dim=embedding_dim,
            dtype=dtype,
        )

    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU):
        module = plan.module
        rank = mpu.get_local_rank(ParallelMode.TENSOR)
        world_size = mpu.get_world_size(ParallelMode.TENSOR)

        assert module.num_embeddings % world_size == 0, (
            f"Num embeddings ({module.num_embeddings}) must be divisible by "
            f"the world size ({world_size})."
        )

        vocab_start_idx, vocab_end_idx = VocabUtility.vocab_range_from_global_vocab_size(
            module.num_embeddings, rank, world_size
        )

        with torch.no_grad():
            chunked_weight = module.weight.chunk(world_size, dim=0)
            module.weight.data = chunked_weight[rank].contiguous()
            tag_module(module, ParallelMode.TENSOR, rank)

        return cls.convert_to_parallel_module(
            plan=plan,
            mpu=mpu,
            vocab_start_idx=vocab_start_idx,
            vocab_end_idx=vocab_end_idx,
            num_embeddings=module.weight.size(0),
        )

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU):
        module = plan.module
        world_size = mpu.get_world_size(ParallelMode.TENSOR)

        with torch.no_grad():
            tensor_list = [torch.zeros_like(module.weight.data) for _ in range(world_size)]
            dist.all_gather(tensor_list, module.weight.data.contiguous(), mpu.get_group(ParallelMode.TENSOR))
            weight = torch.cat(tensor_list, dim=0)
            module.weight.data = weight[: module.original_num_embeddings, :].contiguous()

        return cls.restore_to_original_module(
            plan=plan,
            num_embeddings=module.original_num_embeddings,
            vocab_start_idx=None,
            vocab_end_idx=None,
        )

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"vocab_start_idx={self.vocab_start_idx}, "
            f"vocab_end_idx={self.vocab_end_idx}"
        )

    def forward(self, input: torch.Tensor):
        if self.world_size > 1:
            input_mask = (input < self.vocab_start_idx) | (input > self.vocab_end_idx)
            masked_input = input.clone() - self.vocab_start_idx
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        if self.world_size > 1:
            output_parallel[input_mask, :] = 0.0

        return tp_all_reduce(output_parallel, self.mpu)


class ColumnParallelLinear(nn.Linear, ParallelizableModuleMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = False,
        mpu: Optional[MPU] = None,
    ):
        self.mpu = mpu
        self.world_size = mpu.get_world_size(ParallelMode.TENSOR)
        self.rank = mpu.get_local_rank(ParallelMode.TENSOR)
        self.gather_output = gather_output

        assert out_features % self.world_size == 0, (
            f"Out features ({out_features}) must be divisible by "
            f"the world size ({self.world_size})."
        )

        super().__init__(
            in_features=in_features,
            out_features=out_features // self.world_size,
            bias=bias,
            dtype=dtype,
        )

    @staticmethod
    def _has_bias(module):
        return hasattr(module, "bias") and module.bias is not None and module.bias.dim() >= 1

    @classmethod
    def _scatter_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, tensor_type: str):
        tensor = getattr(plan.module, tensor_type)
        attention_type = plan.attention_type
        rank = mpu.get_local_rank(ParallelMode.TENSOR)
        world_size = mpu.get_world_size(ParallelMode.TENSOR)

        if attention_type is not None and attention_type.value > 1:
            num_fused = attention_type.value
            scattered = tensor.chunk(num_fused * world_size, dim=0)
            scattered = [scattered[i * world_size: (i + 1) * world_size] for i in range(num_fused)]
            scattered = list(map(lambda t: torch.cat([*t], dim=0), zip(*scattered)))
        else:
            scattered = tensor.chunk(world_size, dim=0)

        tensor.data = scattered[rank].contiguous()
        return tensor

    @classmethod
    def _gather_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, tensor_type: str):
        tensor = getattr(plan.module, tensor_type)
        world_size = mpu.get_world_size(ParallelMode.TENSOR)
        attention_type = plan.attention_type
        num_fused = attention_type.value if attention_type is not None else 1

        final_outputs = []
        for t in tensor.chunk(num_fused, dim=0):
            gather_outputs = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(gather_outputs, t.contiguous(), mpu.get_group(ParallelMode.TENSOR))
            final_outputs.append(torch.cat(gather_outputs, dim=0))

        tensor.data = torch.cat(final_outputs, dim=0).contiguous()
        return tensor

    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU, scatter_tensor: bool = True):
        module = plan.module

        if not hasattr(module, "weight") or module.weight is None or module.weight.dim() != 2:
            return module

        rank = mpu.get_local_rank(ParallelMode.TENSOR)
        with torch.no_grad():
            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()

            if scatter_tensor is True:
                module.weight = cls._scatter_tensor(plan, mpu, "weight")
                if cls._has_bias(module):
                    module.bias = cls._scatter_tensor(plan, mpu, "bias")
                tag_module(module, ParallelMode.TENSOR, rank)

        return cls.convert_to_parallel_module(
            plan=plan,
            mpu=mpu,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            gather_output=False,
        )

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU, gather_tensor: bool = True):
        module = plan.module
        with torch.no_grad():
            if gather_tensor is True:
                module.weight = cls._gather_tensor(plan, mpu, "weight")
                if cls._has_bias(module):
                    module.bias = cls._gather_tensor(plan, mpu, "bias")

            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()

        return cls.restore_to_original_module(
            plan=plan,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            gather_output=None,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, input: torch.Tensor):
        input = tp_broadcast(input, self.mpu)
        outputs = F.linear(input, self.weight, bias=self.bias)

        if hasattr(self, "original_out_features"):
            vocab_size = int(self.original_out_features)
            shard_size = int(self.out_features)
            start = self.rank * shard_size
            if vocab_size <= start:
                valid_local = 0
            else:
                valid_local = min(shard_size, vocab_size - start)

            if valid_local < shard_size:
                outputs[..., valid_local:].fill_(torch.finfo(outputs.dtype).min)

        if self.gather_output:
            outputs = tp_all_gather(outputs, dim=-1, mpu=self.mpu)

        if not outputs.is_contiguous():
            outputs = outputs.contiguous()

        return outputs


class RowParallelLinear(nn.Linear, ParallelizableModuleMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_input: bool = True,
        mpu: Optional[MPU] = None,
    ):
        self.mpu = mpu
        self.world_size = mpu.get_world_size(ParallelMode.TENSOR)
        self.rank = mpu.get_local_rank(ParallelMode.TENSOR)
        self.parallel_input = parallel_input

        assert in_features % self.world_size == 0, (
            f"In features ({in_features}) must be divisible by "
            f"the world size ({self.world_size})."
        )

        super().__init__(
            in_features=in_features // self.world_size,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
        )

    @classmethod
    def _scatter_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, tensor_type: str):
        tensor = getattr(plan.module, tensor_type)
        rank = mpu.get_local_rank(ParallelMode.TENSOR)
        world_size = mpu.get_world_size(ParallelMode.TENSOR)

        chunked = tensor.chunk(world_size, dim=1)
        tensor.data = chunked[rank].contiguous()
        return tensor

    @classmethod
    def _gather_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, tensor_type: str):
        tensor = getattr(plan.module, tensor_type)
        world_size = mpu.get_world_size(ParallelMode.TENSOR)

        gather_outputs = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_outputs, tensor.contiguous(), mpu.get_group(ParallelMode.TENSOR))
        tensor.data = torch.cat(gather_outputs, dim=1).contiguous()
        return tensor

    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU):
        module = plan.module

        if not hasattr(module, "weight") or module.weight is None or module.weight.dim() != 2:
            return module

        with torch.no_grad():
            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()
            module.weight = cls._scatter_tensor(plan, mpu, "weight")

        return cls.convert_to_parallel_module(
            plan=plan,
            mpu=mpu,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_input=True,
        )

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU):
        module = plan.module
        with torch.no_grad():
            module.weight = cls._gather_tensor(plan, mpu, "weight")
            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()

        return cls.restore_to_original_module(
            plan=plan,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_input=None,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, input: torch.Tensor):
        if not self.parallel_input:
            input = tp_scatter(input, dim=-1, mpu=self.mpu)

        outputs = F.linear(input, self.weight, bias=None)
        outputs = tp_all_reduce(outputs, self.mpu)

        if self.bias is not None:
            outputs = outputs + self.bias

        if not outputs.is_contiguous():
            outputs = outputs.contiguous()

        return outputs
