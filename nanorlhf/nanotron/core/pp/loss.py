import torch

from nanorlhf.nanotron.core.pp.buffer import PipelineBuffer
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.distributed.p2p import P2P


class MicroLossTensor(torch.Tensor):
    p2p = None
    buffer = None
    buffer_id = None
    num_stages = None
    stage_id = None
    prev_stage = None
    next_stage = None
    is_first_stage = None
    is_last_stage = None

    @classmethod
    def set_arguments(cls, mpu: MPU, buffer: PipelineBuffer, buffer_id: int):
        """
        Set class-level arguments for MicroLossTensor.

        Args:
            mpu (MPU): The model parallel unit.
            buffer (PipelineBuffer): The pipeline buffer.
            buffer_id (int): The buffer slot index.
        """
        cls.p2p = P2P(mpu, mode=ParallelMode.PIPELINE)
        cls.buffer = buffer
        cls.buffer_id = buffer_id
        cls.num_stages = mpu.get_world_size(ParallelMode.PIPELINE)
        cls.stage_id = mpu.get_local_rank(ParallelMode.PIPELINE)

        stage_to_rank = mpu.get_ranks_in_group(ParallelMode.PIPELINE)
        cls.prev_stage = stage_to_rank[cls.stage_id - 1] if cls.stage_id > 0 else None
        cls.next_stage = stage_to_rank[cls.stage_id + 1] if cls.stage_id < cls.num_stages - 1 else None

        cls.is_first_stage = cls.stage_id == 0
        cls.is_last_stage = cls.stage_id == cls.num_stages - 1

    def backward(self, **kwargs):
        """
        Custom backward method for MicroLossTensor to handle pipeline parallelism.
        """
        if not self.is_last_stage:
            self.exec_recv_gradient(self.buffer_id)

        self.exec_backward_pass(self.buffer_id, **kwargs)

        if not self.is_first_stage:
            self.exec_send_gradient(self.buffer_id)

    def exec_send_gradient(self, buffer_id: int):
        """
        Send gradients to the previous pipeline stage.

        Args:
            buffer_id (int): The buffer slot index.
        """
        assert len(self.buffer.inputs[buffer_id]) > 0, (
            "Input buffer of pipeline parallelized model is empty. "
            "You must call `loss.backward()` inside of micro batch for loop context."
        )
        for key, value in self.buffer.inputs[buffer_id].items():
            if torch.is_tensor(value) and value.grad is not None:
                self.buffer.grads[buffer_id][key] = value.grad

        gradient = self.buffer.grads[buffer_id]
        self.p2p.send(gradient, self.prev_stage)

        self.free_buffers("inputs", buffer_id)
        self.free_buffers("outputs", buffer_id)
        self.free_buffers("grads", buffer_id)

    def exec_recv_gradient(self, buffer_id: int):
        """
        Receive gradients from the next pipeline stage.

        Args:
            buffer_id (int): The buffer slot index.
        """
        self.buffer.grads[buffer_id] = self.p2p.recv(self.next_stage)

    def exec_backward_pass(self, buffer_id: int, **kwargs):
        """
        Execute the backward pass using the received gradients.

        Args:
            buffer_id (int): The buffer slot index.
            **kwargs: Additional arguments for the backward function.
        """
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

    def free_buffers(self, buffer_key: str, buffer_id: int):
        """
        Free a specific buffer slot.

        Args:
            buffer_key (str): The buffer attribute name to free (e.g., 'inputs', 'outputs', 'grads').
            buffer_id (int): The index of the buffer slot to free.
        """
        getattr(self.buffer, buffer_key)[buffer_id] = {}
