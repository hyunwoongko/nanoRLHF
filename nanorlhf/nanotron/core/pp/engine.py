from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch import nn

from nanorlhf.nanotron.core.pp.buffer import PipelineBuffer
from nanorlhf.nanotron.core.pp.loss import MicroLossTensor
from nanorlhf.nanotron.core.pp.utils import (
    partition_layers,
    get_layer_owner,
    guess_batch_size,
    zero_grads,
)
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.distributed.p2p import P2P
from nanorlhf.nanotron.utils.huggingface import run_layer, post_process_hf_model
from nanorlhf.nanotron.utils.snapshot import ModuleSnapshotGenerator, to_kwargs
from nanorlhf.nanotron.utils.wrapping import (
    ParallelizationWrapper,
    tag_modules,
    tag_module,
)


class PipelineParallelWrapper(ParallelizationWrapper):
    """
    Discussion:
        Q. How to determine the pipeline stages?
            This engine first traces the model to identify layers and structure. Then, it partitions the layers
            into contiguous chunks, assigning each chunk to a pipeline stage. The partitioning aims to balance
            the number of layers per stage, but does not consider computational load or memory usage.
            See details of `ModelParallelTracer` in `nanorlhf.nanotron.utils.tracing.py` for how the model is traced.

        Q. What scheduling algorithm does this use?
            This engine does not enforce a fixed schedule. You can implement scheduing algorithms such as GPipe, 1F1B,
            by controlling the forward and backward passes in your training loop. See below for details.

        Q. What are common scheduling algorithms?
            - GPipe (Google Pipe):
                GPipe divides a model into stages. All micro-batches go through the forward pass first,
                and then the backward pass runs in reverse order. This can create idle time called pipeline bubbles,
                when some devices wait for others. It also uses more memory because intermediate activations are kept
                until the backward pass.

                GPipe usually performs a final flush to make sure every micro-batch has finished before the step.
                This can add extra latency when there are many micro-batches or stages.

                                                Large Pipeline Bubble
                                                        ↓    ↓
                |------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|----|----|----|----|
                | GPU0 | F1 | F2 | F3 | F4 |    |    |    |    |    |    | B1 | B2 | B3 | B4 |FLUSH| F5 | F6 | F7 | F8 |
                | GPU1 |    | F1 | F2 | F3 | F4 |    |    |    |    | B1 | B2 | B3 | B4 |    |FLUSH|    | F5 | F6 | F7 |
                | GPU2 |    |    | F1 | F2 | F3 | F4 |    |    | B1 | B2 | B3 | B4 |    |    |FLUSH|    |    | F5 | F6 |
                | GPU3 |    |    |    | F1 | F2 | F3 | F4 | B1 | B2 | B3 | B4 |    |    |    |FLUSH|    |    |    | F5 |
                |------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|----|----|----|----|
                                                                                                ↑
                                                                                         Pipeline Flush
                Implementation Example:
                    >>> model = ...
                    >>> optimizer = ...
                    >>> data_loader = ...
                    >>> for batch in data_loader:
                    >>>     micro_outputs = []
                    >>>     for micro_output in model(**batch):
                    >>>         micro_outputs.append(micro_output)          # ← Store intermediate outputs
                    >>>     for micro_output in reversed(micro_outputs):
                    >>>         micro_output.loss.backward()                # ← Backward pass (Pipeline Flush)
                    >>>     optimizer.step()                                # ← Update weights

            - 1F1B (One Forward One Backward):
                 1F1B interleaves forward and backward. A device runs a forward on one micro-batch and then runs
                 a backward soon after on another. This reduces idle time and lowers memory use because activations
                 can be released sooner. We strongly recommend using 1F1B for most use cases.

                |------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|
                | GPU0 | F1 | F2 | F3 | F4 |    |    |    | B1 | F5 | B2 | F6 | B3 | F7 | B4 | F8 | B5 | F9 | B6 | F10 |
                | GPU1 |    | F1 | F2 | F3 |    |    | B1 | F4 | B2 | F5 | B3 | F6 | B4 | F7 | B5 | F8 | B6 | F9 | B7  |
                | GPU2 |    |    | F1 | F2 |    | B1 | F3 | B2 | F4 | B3 | F5 | B4 | F6 | B5 | F7 | B6 | F8 | B7 | F9  |
                | GPU3 |    |    |    | F1 | B1 | F2 | B2 | F3 | B3 | F4 | B4 | F5 | B5 | F6 | B6 | F7 | B7 | F8 | B8  |
                |------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|-----|

                Implementation Example:
                    >>> model = ...
                    >>> optimizer = ...
                    >>> data_loader = ...
                    >>> for batch in data_loader:
                    >>>     for micro_output in model(**batch):
                    >>>         micro_output.loss.backward()                # ← Backward pass
                    >>>     optimizer.step()                                # ← Update weights
    """

    def __init__(self, model: nn.Module, mpu: MPU, micro_batch_size: int = 1):
        super().__init__(model, mpu, parallelization_priority=0)
        # distributed related
        self.p2p = P2P(mpu, mode=ParallelMode.PIPELINE)
        self.device = torch.device(torch.cuda.current_device())

        # stage related
        self.num_stages = mpu.get_world_size(ParallelMode.PIPELINE)
        self.stage_id = mpu.get_local_rank(ParallelMode.PIPELINE)
        self.stage_to_rank = mpu.get_ranks_in_group(ParallelMode.PIPELINE)
        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == self.num_stages - 1
        self.first_stage = 0
        self.last_stage = self.num_stages - 1
        self.prev_stage = None if self.is_first_stage else self.stage_to_rank[self.stage_id - 1]
        self.next_stage = None if self.is_last_stage else self.stage_to_rank[self.stage_id + 1]

        # buffer related
        self.num_buffers = 0
        self.buffer = PipelineBuffer()

        # batch related
        self.batch_size = None
        self.micro_offset = 0
        self.micro_batches = None
        self.micro_batch_size = micro_batch_size

    def _forward(self, *args, **kwargs):
        """Do forward pass with pipeline parallelism."""
        _kwargs = to_kwargs(self.model_forward, args, kwargs)
        _kwargs["use_cache"] = False

        self.micro_offset = 0
        self.micro_batches = self._split_batches(_kwargs)
        self._reserve_buffers(self.num_micro_batches)

        for micro_idx in range(self.num_micro_batches):
            if self.is_first_stage:
                self._exec_load_micro_batch(micro_idx)

            if not self.is_first_stage:
                self._exec_recv_activations(micro_idx)

            self._exec_forward_pass(micro_idx)

            if not self.is_last_stage:
                self._exec_send_activations(micro_idx)

            yield self._exec_postprocess(micro_idx)

        self._exec_all_reduce_embedding()

        for micro_idx in range(self.num_micro_batches):
            self._free_buffers("inputs", micro_idx)
            self._free_buffers("outputs", micro_idx)
            self._free_buffers("grads", micro_idx)

    def _parallelize(self):
        """
        Set up pipeline parallelism by partitioning layers and moving them to the current device.
        """
        # 1) Partition layers among pipeline stages
        module_list = self.mp_plan.main_module_list
        num_layers = len(module_list)
        partitions = partition_layers(self.num_stages, num_layers)

        # 2) Assign local layer range for this stage
        self.local_start = partitions[self.stage_id]
        self.local_end = partitions[self.stage_id + 1]

        # 3) extract modules from model parallelization plan
        embeddings, pre_modules, post_modules, heads = self.mp_plan.extract_modules()

        # 4) Tag {mode: local_rank} to parameters and buffers
        if self.is_first_stage:
            self._collect_tied_modules()
            tag_modules(
                modules=embeddings + pre_modules,
                mode=ParallelMode.PIPELINE,
                local_rank=self.stage_id,
            )

        for idx in range(self.local_start, self.local_end):
            tag_module(
                module=self.mp_plan.main_module_list[idx],
                mode=ParallelMode.PIPELINE,
                local_rank=self.stage_id,
            )

        if self.is_last_stage:
            self._collect_tied_modules()
            tag_modules(
                modules=post_modules + heads,
                mode=ParallelMode.PIPELINE,
                local_rank=self.stage_id,
            )

    def _deparallelize(self):
        """
        Tear down pipeline parallelism by collecting parameters from all stages.
        """
        # 1) Compute layer to stage mapping
        module_list = self.mp_plan.main_module_list
        num_layers = len(module_list)
        partitions = partition_layers(self.num_stages, num_layers)
        layer_owner = get_layer_owner(partitions)

        # 2) extract modules from model parallelization plan
        embeddings, pre_modules, post_modules, heads = self.mp_plan.extract_modules()

        # 3) Collect parameters from all stages
        self._deparallelize_modules(embeddings + pre_modules, src_rank=self.first_stage)

        for idx in range(num_layers):
            self._deparallelize_modules([self.mp_plan.main_module_list[idx]], src_rank=layer_owner[idx])

        self._deparallelize_modules(post_modules + heads, src_rank=self.last_stage)

    def _collect_tied_modules(self):
        """
        Collect tied modules (e.g., embeddings) and ensure they share weights across stages.
        """
        if self.mp_plan.tied_plan is None:
            return

        input_emb_plan, output_emb_plan = self.mp_plan.tied_plan
        input_emb = input_emb_plan.module
        output_emb = output_emb_plan.module

        if (
            hasattr(input_emb, "weight")
            and hasattr(output_emb, "weight")
            and input_emb.weight is not None
            and output_emb.weight is not None
        ):
            self.buffer.embeddings[output_emb.weight] = input_emb.weight
        if (
            hasattr(input_emb, "bias")
            and hasattr(output_emb, "bias")
            and input_emb.bias is not None
            and output_emb.bias is not None
        ):
            self.buffer.embeddings[output_emb.bias] = input_emb.bias

    def _deparallelize_modules(self, modules: List[nn.Module], src_rank: int):
        """
        Collect parameters from the specified source rank and update the local modules.

        Args:
            modules (List[nn.Module]): List of modules to update.
            src_rank (int): The source pipeline stage rank to collect parameters from.
        """
        for module in modules:
            for m in module.modules():
                if hasattr(m, "weight") and m.weight is not None:
                    weight = m.weight.data.clone().contiguous().cuda()
                    dist.broadcast(
                        weight,
                        src=self.stage_to_rank[src_rank],
                        group=self.mpu.get_group(ParallelMode.PIPELINE)
                    )
                    m.weight.data = weight
                if hasattr(m, "bias") and m.bias is not None:
                    bias = m.bias.data.clone().contiguous().cuda()
                    dist.broadcast(
                        bias,
                        src=self.stage_to_rank[src_rank],
                        group=self.mpu.get_group(ParallelMode.PIPELINE)
                    )
                    m.bias.data = bias

    def _split_batches(self, batches: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split mini-batches to micro-batches.

        Args:
            batches (Dict[str, Any]): The input batches to split.

        Returns:
            List[Dict[str, Any]]: A list of micro-batches.
        """
        self.batch_size = guess_batch_size(batches)
        assert self.batch_size % self.micro_batch_size == 0, (
            "The batch size must be divisible by the micro batch size. "
            f"Got batch_size={self.batch_size} and micro_batch_size={self.micro_batch_size}."
        )

        self.num_micro_batches = self.batch_size // self.micro_batch_size
        micro_batches = [{} for _ in range(self.num_micro_batches)]

        for k, v in batches.items():
            if torch.is_tensor(v):
                if v.size(0) == self.batch_size:
                    micro_batch = v.chunk(self.num_micro_batches, dim=0)
                    for i, m in enumerate(micro_batch):
                        micro_batches[i][k] = m
                else:
                    for i in range(self.num_micro_batches):
                        micro_batches[i][k] = v
            else:
                for i in range(self.num_micro_batches):
                    micro_batches[i][k] = v

        return micro_batches

    def _reserve_buffers(self, num_buffers: int):
        """
        Allocate buffer slots for inputs, outputs, and gradients.

        Args:
            num_buffers (int): Number of buffer slots to reserve.
        """
        if self.num_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_buffers
        reserved_keys = ["inputs", "outputs", "grads"]
        for key in self.buffer.keys():
            if key in reserved_keys:
                current_buffer = getattr(self.buffer, key)
                current_buffer.extend([{} for _ in range(num_added)])
        self.num_buffers = num_buffers

    def _free_buffers(self, buffer_key: str, buffer_id: int):
        """
        Free a specific buffer slot.

        Args:
            buffer_key (str): The buffer attribute name to free (e.g., 'inputs', 'outputs', 'grads').
            buffer_id (int): The index of the buffer slot to free.
        """
        getattr(self.buffer, buffer_key)[buffer_id] = {}

    def _run_forward_pass_first_stage(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # According to our `self._partition_layers` method,
        # the first layer in the `ModuleList` can only belong to the first stage.
        # Send the snapshot `input_param_name` to the next stage
        first_layer = self.mp_plan.main_module_list[0]
        snapshot_generator = ModuleSnapshotGenerator(first_layer)
        snapshot = snapshot_generator.generate(self.model, inputs)
        if snapshot is None:
            raise RuntimeError("Failed to generate a snapshot for the first layer.")

        # We start from the second layer because the first is already executed
        # when generating the snapshot
        hidden_states = snapshot.output_tensor
        layer_inputs = {snapshot.input_param_name: snapshot.output_tensor, **snapshot.kwargs}
        for layer in self.mp_plan.main_module_list[1 : self.local_end]:
            hidden_states = run_layer(layer, layer_inputs)
            layer_inputs[snapshot.input_param_name] = hidden_states

        return {
            "input_param_name": snapshot.input_param_name,
            "hidden_states": hidden_states,
            "module_list_kwargs": snapshot.kwargs,
            "user_inputs": inputs,
        }

    def _run_forward_pass_remained_stages(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        input_param_name = payload["input_param_name"]
        module_list_kwargs = payload.get("module_list_kwargs", {})
        layer_inputs = {input_param_name: payload["hidden_states"], **module_list_kwargs}

        hidden_states = layer_inputs[input_param_name]
        for layer in self.mp_plan.main_module_list[self.local_start : self.local_end]:
            hidden_states = run_layer(layer, layer_inputs)
            layer_inputs[input_param_name] = hidden_states

        return {
            "input_param_name": input_param_name,
            "hidden_states": hidden_states,
            "module_list_kwargs": module_list_kwargs,
            "user_inputs": payload["user_inputs"],
        }

    def _exec_load_micro_batch(self, buffer_id: int):
        """
        Load a micro-batch into the specified buffer slot.

        Args:
            buffer_id (int): The buffer slot index to load the micro-batch into.
        """
        micro_batch = self.micro_batches[self.micro_offset]
        if self.is_first_stage:
            loaded = {}
            for k, v in micro_batch.items():
                if torch.is_tensor(v):
                    v = v.clone().detach().to(self.device)
                    v.requires_grad = v.is_floating_point()
                loaded[k] = v
            self.buffer.inputs[buffer_id] = loaded

    def _exec_send_activations(self, buffer_id: int):
        """
        Send activations from the specified buffer slot to the next pipeline stage.

        Args:
            buffer_id (int): The buffer slot index containing the activations to send.
        """
        self.p2p.send(self.buffer.outputs[buffer_id], dst_rank=self.next_stage)

    def _exec_recv_activations(self, buffer_id: int):
        """
        Receive activations into the specified buffer slot from the previous pipeline stage.

        Args:
            buffer_id (int): The buffer slot index to receive the activations into.
        """
        self.buffer.inputs[buffer_id] = self.p2p.recv(src_rank=self.prev_stage)

    def _exec_forward_pass(self, buffer_id: int):
        self.micro_offset += 1
        inputs = self.buffer.inputs[buffer_id]

        if self.is_first_stage:
            inputs = zero_grads(inputs)
            payload = self._run_forward_pass_first_stage(inputs)
        else:
            payload = self._run_forward_pass_remained_stages(inputs)

        self.buffer.outputs[buffer_id] = payload

    def _exec_postprocess(self, buffer_id: int):
        if self.is_last_stage:
            payload = self.buffer.outputs[buffer_id]
            last_hidden_states = payload["hidden_states"]
            for post_module_plan in self.mp_plan.post_module_list_plans:
                last_hidden_states = post_module_plan.module(last_hidden_states)

            logits = None
            if self.mp_plan.head_plan is not None:
                head_plan = self.mp_plan.head_plan
                logits = head_plan.module(last_hidden_states)

            dict_output = {"logits": logits, "payload": payload}
        else:
            dict_output = {}

        if not self.is_last_stage:
            dict_output = self.p2p.recv(src_rank=self.next_stage)

        if not self.is_first_stage:
            self.p2p.send(dict_output, dst_rank=self.prev_stage)

        modeling_output = post_process_hf_model(
            model=self.model,
            mpu=self.mpu,
            logits=dict_output["logits"],
            payload=dict_output["payload"],
        )

        if hasattr(modeling_output, "loss") and modeling_output.loss is not None:
            modeling_output.loss.__class__ = MicroLossTensor
            modeling_output.loss.set_arguments(self.mpu, self.buffer, buffer_id)

        return modeling_output

    def _exec_all_reduce_embedding(self):
        for head, embedding in self.buffer.embeddings.items():
            if head.grad is not None and embedding.grad is not None:
                dist.all_reduce(
                    tensor=embedding.grad,
                    group=self.mpu.get_group(ParallelMode.TIED_EMBEDDING),
                )
                dist.all_reduce(
                    tensor=head.grad,
                    group=self.mpu.get_group(ParallelMode.TIED_EMBEDDING),
                )
