from typing import Callable, Dict, Any

import torch
from torch import nn

from nanorlhf.nanotron.core.tp.modules import VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.utils.huggingface import is_causal_lm, post_process_hf_model
from nanorlhf.nanotron.utils.snapshot import to_kwargs, ModuleSnapshotGenerator
from nanorlhf.nanotron.utils.tracing import ModuleParallelPlan, SlicingType
from nanorlhf.nanotron.utils.wrapping import ParallelizationWrapper, tag_module

ATTRS_TO_UPDATE = [
    'all_head_size',
    'd_model',
    'd_model_size',
    'embed_dim',
    'hidden_size',
    'hidden_size_per_partition',
    'inner_dim',
    'kv_n_heads',
    'n_head',
    'n_heads',
    'num_attention_heads',
    'num_attention_heads_per_partition',
    'num_attn_heads',
    'num_heads',
    'num_key_value_heads',
    'num_kv',
    'num_kv_heads',
    'num_multi_query_groups_per_partition',
    'split_size',
]


class TensorParallelWrapper(ParallelizationWrapper):

    def __init__(self, model: nn.Module, mpu: MPU):
        super().__init__(model, mpu, parallelization_priority=1)
        self.world_size = self.mpu.get_world_size(ParallelMode.TENSOR)
        self.rank = self.mpu.get_local_rank(ParallelMode.TENSOR)
        self.device = torch.cuda.current_device()

    def _pad_tensor(self, module: nn.Module, name: str, dim: int):
        original_tensor = getattr(module, name)
        original_size = original_tensor.size(dim)
        resized_size = original_size
        while resized_size % self.world_size != 0:
            resized_size += 1

        if resized_size != original_size:
            padding_shape = list(original_tensor.size())
            padding_shape[dim] = resized_size - original_size
            padding = torch.zeros(
                padding_shape,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            new_tensor = torch.cat([original_tensor.data, padding], dim=dim)
            original_tensor.data = new_tensor
        return resized_size

    def _pad_embedding_related_params(self):
        embedding = self.mp_plan.embedding_plan.module
        original_num_embeddings = embedding.weight.size(0)
        padded_num_embedding = self._pad_tensor(embedding, "weight", dim=0)
        setattr(embedding, "num_embeddings", padded_num_embedding)
        setattr(embedding, "original_num_embeddings", original_num_embeddings)

        class_name = self.model.__class__.__qualname__
        is_causal_language_model = class_name.endswith("CausalLM") or class_name.endswith("LMHeadModel")
        # We only convert head to `ColumnParallelLinear` when it is a projection of large vocabulary.
        # Otherwise, the head is small and doesn't benefit from tensor parallelism.
        # Introducing additional broadcast and all-reduce communication to save only a few megabytes
        # of memory is counterproductive due to the communication overhead.

        if is_causal_language_model:
            head = self.mp_plan.head_plan.module
            is_tied_head = self.mp_plan.tied_plan is not None

            if is_tied_head:
                # Already padded because the head is tied with embedding
                setattr(head, "out_features", padded_num_embedding)
                setattr(head, "original_out_features", original_num_embeddings)
            else:
                original_out_features = head.weight.size(0)
                padded_out_features = self._pad_tensor(head, "weight", dim=0)
                setattr(head, "out_features", padded_out_features)
                setattr(head, "original_out_features", original_out_features)

            if hasattr(head, "bias") and head.bias is not None:
                self._pad_tensor(head, "bias", dim=0)

    def _load_data(self, kwargs: Dict[str, Any]):
        loaded = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                if v.device != self.device:
                    v = v.clone().detach().to(self.device)
                v.requires_grad = v.is_floating_point()
            loaded[k] = v
        return loaded

    def _update_existing_attrs(self, fn: Callable):
        for module in self.model.modules():
            for attr in ATTRS_TO_UPDATE:
                if hasattr(module, attr):
                    original_value = getattr(module, attr)
                    if original_value < self.world_size:
                        raise RuntimeError(
                            f"After tensor parallelism, the attribute '{attr}' "
                            f"in the module '{module.__class__.__qualname__}' would be zero. "
                            f"That means this model is too small to be "
                            f"tensor parallelized across {self.world_size} GPUs. "
                            f"({attr}={original_value}, tp_world_size={self.world_size})."
                        )
                    updated_value = fn(original_value)
                    setattr(module, attr, updated_value)

    def _replicate_module(self, plan: ModuleParallelPlan):
        tag_module(plan.module, ParallelMode.TENSOR, self.rank)

    def _parallelize(self):
        self._pad_embedding_related_params()
        self._update_existing_attrs(lambda x: x // self.world_size)

        # Parallelize embedding
        embedding_plan = self.mp_plan.embedding_plan
        embedding_plan.module = VocabParallelEmbedding.parallelize(embedding_plan, self.mpu)

        # Parallelize pre module list
        for plan in self.mp_plan.pre_module_list_plans:
            self._replicate_module(plan)

        # Parallelize main module list
        for plan in self.mp_plan.main_module_list_plans:
            if plan.slicing_type == SlicingType.COLUMN:
                plan.module = ColumnParallelLinear.parallelize(plan, self.mpu)
            elif plan.slicing_type == SlicingType.ROW:
                plan.module = RowParallelLinear.parallelize(plan, self.mpu)
            else:
                self._replicate_module(plan)

        # Parallelize post module list
        for plan in self.mp_plan.post_module_list_plans:
            self._replicate_module(plan)

        # Parallelize head if needed
        head_plan = self.mp_plan.head_plan
        if head_plan is not None and is_causal_lm(self.model):
            has_not_tied_head = self.mp_plan.tied_plan is None
            head_plan.module = ColumnParallelLinear.parallelize(
                head_plan, self.mpu, scatter_tensor=has_not_tied_head
            )

    def _deparallelize(self):
        self._update_existing_attrs(lambda x: x * self.world_size)

        # Deparallelize embedding
        embedding_plan = self.mp_plan.embedding_plan
        embedding_plan.module = VocabParallelEmbedding.deparallelize(embedding_plan, self.mpu)

        # Deparallelize main module list
        for plan in self.mp_plan.main_module_list_plans:
            if plan.slicing_type == SlicingType.COLUMN:
                plan.module = ColumnParallelLinear.deparallelize(plan, self.mpu)
            elif plan.slicing_type == SlicingType.ROW:
                plan.module = RowParallelLinear.deparallelize(plan, self.mpu)

        # Deparallelize head if needed
        head_plan = self.mp_plan.head_plan
        if head_plan is not None and is_causal_lm(self.model):
            has_not_tied_head = self.mp_plan.tied_plan is None
            head_plan.module = ColumnParallelLinear.deparallelize(
                head_plan, self.mpu, gather_tensor=has_not_tied_head
            )

    def _forward(self, *args, **kwargs):
        _kwargs = to_kwargs(self.model_forward, args, kwargs)
        _kwargs = self._load_data(_kwargs)

        need_snapshot = (
            is_causal_lm(self.model)
            and "labels" in _kwargs
            and _kwargs["labels"] is not None
            and self.mpu.get_world_size(ParallelMode.PIPELINE) == 1
            # must generate snapshot from the first layer when using pp.
        )

        if need_snapshot:
            last_layer = self.mp_plan.main_module_list[-1]
            snapshot_generator = ModuleSnapshotGenerator(last_layer)
            snapshot = snapshot_generator.generate(self.model, _kwargs)
            if snapshot is None:
                raise RuntimeError("Failed to generate a snapshot for the last layer.")

            payload = {
                "input_param_name": snapshot.input_param_name,
                "hidden_states": snapshot.output_tensor,
                "module_list_kwargs": snapshot.kwargs,
                "user_inputs": _kwargs,
            }

            logits = None
            for plan in self.mp_plan.post_module_list_plans:
                payload["hidden_states"] = plan.module(payload["hidden_states"])
            if self.mp_plan.head_plan is not None:
                logits = self.mp_plan.head_plan.module(payload["hidden_states"])

            return post_process_hf_model(
                model=self.model,
                mpu=self.mpu,
                logits=logits,
                payload=payload,
            )

        return self.model_forward(**_kwargs)
