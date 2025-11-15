import torch

from nanorlhf.nanotron.core.pp.buffer import PipelineBuffer
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.distributed.p2p import P2P


class MicroLossTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2p = None
        self.buffer = None
        self.buffer_id = None
        self.num_stages = None
        self.stage_id = None
        self.prev_stage = None
        self.next_stage = None
        self.is_first_stage = None
        self.is_last_stage = None

    def set_arguments(self, mpu: MPU, buffer: PipelineBuffer, buffer_id: int):
        self.p2p = P2P(mpu, mode=ParallelMode.PIPELINE)
        self.buffer = buffer
        self.buffer_id = buffer_id
        self.num_stages = mpu.get_world_size(ParallelMode.PIPELINE)
        self.stage_id = mpu.get_local_rank(ParallelMode.PIPELINE)

        stage_to_rank = mpu.get_ranks_in_group(ParallelMode.PIPELINE)
        self.prev_stage = stage_to_rank[self.stage_id - 1] if self.stage_id > 0 else None
        self.next_stage = stage_to_rank[self.stage_id + 1] if self.stage_id < self.num_stages - 1 else None

        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == self.num_stages - 1

    def backward(self, **kwargs):
        if not self.is_last_stage:
            self._exec_recv_gradient(self.buffer_id)

        self._exec_backward_pass(self.buffer_id, **kwargs)

        if not self.is_first_stage:
            self._exec_send_gradient(self.buffer_id)

    def _exec_send_gradient(self, buffer_id: int):
        assert len(self.buffer.inputs[buffer_id]) > 0, (
            "Input buffer of pipeline parallelized model is empty. "
            "You must call `loss.backward()` inside of micro batch for loop context."
        )
        for key, value in self.buffer.inputs[buffer_id].items():
            if torch.is_tensor(value) and value.grad is not None:
                self.buffer.grads[buffer_id][key] = value.grad

        gradient = self.buffer.grads[buffer_id]
        self.p2p.send(gradient, self.prev_stage)

        self._free_buffers("inputs", buffer_id)
        self._free_buffers("outputs", buffer_id)
        self._free_buffers("grads", buffer_id)

    def _exec_recv_gradient(self, buffer_id: int):
        self.buffer.grads[buffer_id] = self.p2p.recv(self.next_stage)

    def _exec_backward_pass(self, buffer_id: int, **kwargs):
        if self.is_last_stage:
            super().backward(**kwargs)
            return

        assert len(self.buffer.outputs[buffer_id]) > 0, (
            "Input buffers of pipeline parallelized model is empty. "
            "You must call `loss.backward()` inside of micro batch for loop context."
        )

        outputs = self.buffer.outputs[buffer_id]
        grads = self.buffer.grads[buffer_id]
        trainable_outputs = [outputs[key] for key in outputs if key in grads]

        assert len(trainable_outputs) == len(grads), (
            "The number of received gradients does not match the number of trainable outputs. "
            "Please check outputs of your model."
        )

        torch.autograd.backward(
            tensors=tuple(trainable_outputs),
            grad_tensors=tuple(grads.values()),
        )

    def _free_buffers(self, buffer_key: str, buffer_id: int):
        """
        Free a specific buffer slot.

        Args:
            buffer_key (str): The buffer attribute name to free (e.g., 'inputs', 'outputs', 'grads').
            buffer_id (int): The index of the buffer slot to free.
        """
        getattr(self.buffer, buffer_key)[buffer_id] = {}
