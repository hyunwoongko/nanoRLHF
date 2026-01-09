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
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode):
        raise NotImplementedError

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode):
        raise NotImplementedError

    @classmethod
    def convert_to_parallel_module(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, **kwargs):
        """
        Convert the original module to its parallel version by changing its class.

        Args:
            plan (ModuleParallelPlan): The plan containing the original module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            **kwargs: Additional attributes to set on the parallel module.
        """
        module = plan.module
        original_module_class = module.__class__
        module.__class__ = cls
        module.original_module_class = original_module_class

        module.mpu = mpu
        module.mode = mode
        module.world_size = mpu.get_world_size(mode)
        module.rank = mpu.get_local_rank(mode)

        for key, val in kwargs.items():
            setattr(module, key, val)

        return module

    @classmethod
    def restore_to_original_module(cls, plan: ModuleParallelPlan, **kwargs):
        """
        Restore the parallel module back to its original version by changing its class.

        Args:
            plan (ModuleParallelPlan): The plan containing the parallel module.
            **kwargs: Additional attributes to set on the restored module.
        """
        module = plan.module
        module.__class__ = module.original_module_class
        del module.original_module_class
        del module.mpu
        del module.world_size
        del module.rank

        if hasattr(module, "mode"):
            del module.mode

        for key, value in kwargs.items():
            if value is None:
                if hasattr(module, key):
                    delattr(module, key)
            else:
                setattr(module, key, value)

        return module


class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int):
        """
        Get the vocabulary index range for a given partition based on its size and rank.

        Args:
            per_partition_vocab_size (int): The size of the vocabulary partition.
            rank (int): The rank of the current partition.
        """
        first_idx = rank * per_partition_vocab_size
        last_idx = first_idx + per_partition_vocab_size - 1
        return first_idx, last_idx

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int):
        """
        Get the vocabulary index range for a given partition based on the global vocabulary size, rank, and world size.

        Args:
            global_vocab_size (int): The total size of the vocabulary.
            rank (int): The rank of the current partition.
            world_size (int): The total number of partitions.
        """
        assert global_vocab_size % world_size == 0, (
            f"Global vocab size ({global_vocab_size}) must be divisible by " f"the world size ({world_size})."
        )
        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank)


class VocabParallelEmbedding(nn.Embedding, ParallelizableModuleMixin):
    """
    Vocab parallel embedding layer.

    Args:
        num_embeddings (int): Total number of embeddings (vocabulary size).
        embedding_dim (int): Dimension of each embedding vector.
        mode (ParallelMode): The parallel mode for communication.
        dtype (Optional[torch.dtype]): Data type of the embeddings.
        mpu (Optional[MPU]): The model parallel unit.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: ParallelMode,
        dtype: Optional[torch.dtype] = None,
        mpu: Optional[MPU] = None,
    ):
        self.mpu = mpu
        self.mode = mode
        self.world_size = mpu.get_world_size(mode)
        self.rank = mpu.get_local_rank(mode)
        self.vocab_start_idx, self.vocab_end_idx = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings, self.rank, self.world_size
        )
        super().__init__(
            num_embeddings=num_embeddings // self.world_size,
            embedding_dim=embedding_dim,
            dtype=dtype,
        )

    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode):
        """
        Parallelize the embedding layer by partitioning the vocabulary across multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the original module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
        """
        module = plan.module
        rank = mpu.get_local_rank(mode)
        world_size = mpu.get_world_size(mode)

        assert module.num_embeddings % world_size == 0, (
            f"Num embeddings ({module.num_embeddings}) must be divisible by " f"the world size ({world_size})."
        )

        vocab_start_idx, vocab_end_idx = VocabUtility.vocab_range_from_global_vocab_size(
            module.num_embeddings, rank, world_size
        )

        with torch.no_grad():
            chunked_weight = module.weight.chunk(world_size, dim=0)
            module.weight.data = chunked_weight[rank].contiguous()
            tag_module(module, mode, rank)

        return cls.convert_to_parallel_module(
            plan=plan,
            mpu=mpu,
            vocab_start_idx=vocab_start_idx,
            vocab_end_idx=vocab_end_idx,
            num_embeddings=module.weight.size(0),
            mode=mode,
        )

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode):
        """
        Deparallelize the embedding layer by gathering the partitioned vocabulary from multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the parallel module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
        """
        module = plan.module
        world_size = mpu.get_world_size(mode)

        with torch.no_grad():
            tensor_list = [torch.zeros_like(module.weight.data) for _ in range(world_size)]
            dist.all_gather(tensor_list, module.weight.data.contiguous(), mpu.get_group(mode))
            weight = torch.cat(tensor_list, dim=0)
            module.weight.data = weight[: module.original_num_embeddings, :].contiguous()

        return cls.restore_to_original_module(
            plan=plan,
            num_embeddings=module.original_num_embeddings,
            vocab_start_idx=None,
            vocab_end_idx=None,
        )

    def extra_repr(self) -> str:
        """
        Extra representation of the VocabParallelEmbedding layer.

        Returns:
            str: A string representation of the layer's configuration.
        """
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"vocab_start_idx={self.vocab_start_idx}, "
            f"vocab_end_idx={self.vocab_end_idx}"
        )

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass for the vocab parallel embedding layer.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices.
        """
        if self.world_size > 1:
            input_mask = (inputs < self.vocab_start_idx) | (inputs > self.vocab_end_idx)
            masked_input = inputs.clone() - self.vocab_start_idx
            masked_input[input_mask] = 0
        else:
            masked_input = inputs

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
            output_parallel[input_mask, :] = 0.0  # noqa

        return tp_all_reduce(output_parallel, self.mpu, self.mode)


class ColumnParallelLinear(nn.Linear, ParallelizableModuleMixin):
    """
    Column parallel linear layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        mode (ParallelMode): The parallel mode for communication.
        bias (bool): Whether to include a bias term.
        dtype (Optional[torch.dtype]): Data type of the weights.
        gather_output (bool): Whether to gather the output across all processes.
        mpu (Optional[MPU]): The model parallel unit.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: ParallelMode,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = False,
        mpu: Optional[MPU] = None,
    ):
        self.mpu = mpu
        self.world_size = mpu.get_world_size(mode)
        self.rank = mpu.get_local_rank(mode)
        self.gather_output = gather_output

        assert out_features % self.world_size == 0, (
            f"Out features ({out_features}) must be divisible by " f"the world size ({self.world_size})."
        )

        super().__init__(
            in_features=in_features,
            out_features=out_features // self.world_size,
            bias=bias,
            dtype=dtype,
        )

    @staticmethod
    def has_bias(module):
        """
        Check if the module has a bias term.

        Args:
            module (nn.Module): The module to check.
        """
        return hasattr(module, "bias") and module.bias is not None and module.bias.dim() >= 1

    @classmethod
    def scatter_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, tensor_type: str):
        """
        Scatter the tensor across multiple processes based on the attention type.

        Args:
            plan (ModuleParallelPlan): The plan containing the original module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            tensor_type (str): The type of tensor to scatter (e.g., "weight" or "bias").
        """
        tensor = getattr(plan.module, tensor_type)
        attention_type = plan.attention_type
        rank = mpu.get_local_rank(mode)
        world_size = mpu.get_world_size(mode)

        if attention_type is not None and attention_type.value > 1:
            num_fused = attention_type.value
            scattered = tensor.chunk(num_fused * world_size, dim=0)
            scattered = [scattered[i * world_size : (i + 1) * world_size] for i in range(num_fused)]
            scattered = list(map(lambda t: torch.cat([*t], dim=0), zip(*scattered)))
        else:
            scattered = tensor.chunk(world_size, dim=0)

        tensor.data = scattered[rank].contiguous()
        return tensor

    @classmethod
    def gather_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, tensor_type: str):
        """
        Gather the tensor from multiple processes based on the attention type.

        Args:
            plan (ModuleParallelPlan): The plan containing the parallel module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            tensor_type (str): The type of tensor to gather (e.g., "weight" or "bias").
        """
        tensor = getattr(plan.module, tensor_type)
        world_size = mpu.get_world_size(mode)
        attention_type = plan.attention_type
        num_fused = attention_type.value if attention_type is not None else 1

        final_outputs = []
        for t in tensor.chunk(num_fused, dim=0):
            gather_outputs = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(gather_outputs, t.contiguous(), mpu.get_group(mode))
            final_outputs.append(torch.cat(gather_outputs, dim=0))

        tensor.data = torch.cat(final_outputs, dim=0).contiguous()
        return tensor

    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, scatter_tensor: bool = True):
        """
        Parallelize the linear layer by partitioning the output features across multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the original module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            scatter_tensor (bool): Whether to scatter the weight and bias tensors.
        """
        module = plan.module

        if not hasattr(module, "weight") or module.weight is None or module.weight.dim() != 2:
            return module

        rank = mpu.get_local_rank(mode)
        with torch.no_grad():
            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()

            if scatter_tensor is True:
                module.weight = cls.scatter_tensor(plan, mpu, mode, "weight")
                if cls.has_bias(module):
                    module.bias = cls.scatter_tensor(plan, mpu, mode, "bias")
                tag_module(module, mode, rank)

        return cls.convert_to_parallel_module(
            plan=plan,
            mpu=mpu,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            gather_output=False,
            mode=mode,
        )

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, gather_tensor: bool = True):
        """
        Deparallelize the linear layer by gathering the partitioned output features from multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the parallel module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            gather_tensor (bool): Whether to gather the weight and bias tensors.
        """
        module = plan.module
        with torch.no_grad():
            if gather_tensor is True:
                module.weight = cls.gather_tensor(plan, mpu, mode, "weight")
                if cls.has_bias(module):
                    module.bias = cls.gather_tensor(plan, mpu, mode, "bias")

            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()

        return cls.restore_to_original_module(
            plan=plan,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            gather_output=None,
        )

    def extra_repr(self) -> str:
        """
        Extra representation of the ColumnParallelLinear layer.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, input: torch.Tensor):
        """
        Forward pass for the column parallel linear layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        input = tp_broadcast(input, self.mpu, self.mode)
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
            outputs = tp_all_gather(outputs, dim=-1, mpu=self.mpu, mode=self.mode)

        if not outputs.is_contiguous():
            outputs = outputs.contiguous()

        return outputs


class RowParallelLinear(nn.Linear, ParallelizableModuleMixin):
    """
    Row parallel linear layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        mode (ParallelMode): The parallel mode for communication.
        bias (bool): Whether to include a bias term.
        dtype (Optional[torch.dtype]): Data type of the weights.
        parallel_input (bool): Whether the input is already parallelized.
        mpu (Optional[MPU]): The model parallel unit.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: ParallelMode,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_input: bool = True,
        mpu: Optional[MPU] = None,
    ):
        self.mpu = mpu
        self.world_size = mpu.get_world_size(mode)
        self.rank = mpu.get_local_rank(mode)
        self.parallel_input = parallel_input

        assert in_features % self.world_size == 0, (
            f"In features ({in_features}) must be divisible by " f"the world size ({self.world_size})."
        )

        super().__init__(
            in_features=in_features // self.world_size,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
        )

    @classmethod
    def scatter_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, tensor_type: str):
        """
        Scatter the tensor across multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the original module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            tensor_type (str): The type of tensor to scatter (e.g., "weight" or "bias").
        """
        tensor = getattr(plan.module, tensor_type)
        rank = mpu.get_local_rank(mode)
        world_size = mpu.get_world_size(mode)

        chunked = tensor.chunk(world_size, dim=1)
        tensor.data = chunked[rank].contiguous()
        return tensor

    @classmethod
    def gather_tensor(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode, tensor_type: str):
        """
        Gather the tensor from multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the parallel module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
            tensor_type (str): The type of tensor to gather (e.g., "weight" or "bias").
        """
        tensor = getattr(plan.module, tensor_type)
        world_size = mpu.get_world_size(mode)

        gather_outputs = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_outputs, tensor.contiguous(), mpu.get_group(mode))
        tensor.data = torch.cat(gather_outputs, dim=1).contiguous()
        return tensor

    @classmethod
    def parallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode):
        """
        Parallelize the linear layer by partitioning the input features across multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the original module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
        """
        module = plan.module

        if not hasattr(module, "weight") or module.weight is None or module.weight.dim() != 2:
            return module

        with torch.no_grad():
            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()
            module.weight = cls.scatter_tensor(plan, mpu, mode, "weight")

        return cls.convert_to_parallel_module(
            plan=plan,
            mpu=mpu,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_input=True,
            mode=mode,
        )

    @classmethod
    def deparallelize(cls, plan: ModuleParallelPlan, mpu: MPU, mode: ParallelMode):
        """
        Deparallelize the linear layer by gathering the partitioned input features from multiple processes.

        Args:
            plan (ModuleParallelPlan): The plan containing the parallel module.
            mpu (MPU): The model parallel unit.
            mode (ParallelMode): The parallel mode for communication.
        """
        module = plan.module
        with torch.no_grad():
            module.weight = cls.gather_tensor(plan, mpu, mode, "weight")
            if not plan.is_reversed:
                module.weight.data = module.weight.data.t()

        return cls.restore_to_original_module(
            plan=plan,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_input=None,
        )

    def extra_repr(self) -> str:
        """
        Extra representation of the RowParallelLinear layer.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass for the row parallel linear layer.

        Args:
            inputs (torch.Tensor): Input tensor.
        """
        if not self.parallel_input:
            inputs = tp_scatter(inputs, dim=-1, mpu=self.mpu, mode=self.mode)

        outputs = F.linear(inputs, self.weight, bias=None)
        outputs = tp_all_reduce(outputs, self.mpu, self.mode)

        if self.bias is not None:
            outputs = outputs + self.bias

        if not outputs.is_contiguous():
            outputs = outputs.contiguous()

        return outputs
