import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanoverl.utils.optim_utils import get_optimizer_param_groups, get_scheduler
from nanorlhf.nanoverl.utils.packing_utils import split_packed_batch


def initialize_model(config, rank):
    """
    Initialize the model, optimizer, and parallelism wrappers.

    Args:
        config: Configuration object containing model and training settings.
        rank (int): The rank of the current process in a distributed setup.

    Returns:
        model: The initialized and parallelized model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.train()

    optimizer = AdamW(
        get_optimizer_param_groups(model, float(config.optim.weight_decay)),
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.weight_decay),
        betas=(float(config.optim.beta1), float(config.optim.beta2)),
    )

    if config.model.gradient_checkpointing_enable:
        if config.model.pipeline_parallel_size == 1:
            # pipeline parallel engine controls grad checkpointing itself.
            model.gradient_checkpointing_enable()

    global_world_size = (
        config.model.data_parallel_size * config.model.tensor_parallel_size * config.model.pipeline_parallel_size
    )
    assert rank < global_world_size, "rank must be < dp*tp*pp"

    mpu = MPU(
        rank=rank,
        local_rank=rank,
        world_size=global_world_size,
        local_world_size=global_world_size,
        host=config.model.host,
        port=config.model.port,
        data_parallel_size=config.model.data_parallel_size,
        pipeline_parallel_size=config.model.pipeline_parallel_size,
        tensor_parallel_size=config.model.tensor_parallel_size,
        rollout_data_parallel_size=0,
        rollout_tensor_parallel_size=0,
        backend=config.model.backend,
        seed=config.model.seed,
    )
    model = TensorParallel(
        model,
        mpu=mpu,
    )
    model = PipelineParallel(
        model,
        mpu=mpu,
        micro_batch_size=config.data.train_micro_batch_size_per_gpu,
        gradient_checkpointing_enable=config.model.gradient_checkpointing_enable,
    )
    accum_steps = max(
        1,
        config.data.train_batch_size
        // (config.model.data_parallel_size * config.data.train_micro_batch_size_per_gpu),
    )
    model, optimizer = DataParallel(
        model,
        mpu=mpu,
        optimizer=optimizer,
        zero_stage=config.model.zero_stage,
        accum_steps=accum_steps,
    )
    model.parallelize()
    model = patch_kernel(model)
    return model, mpu, optimizer


@nanoray.remote
class SFTWorker:
    """
    Supervised Fine-Tuning (SFT) worker that handles model training and evaluation.

    Args:
        config: Configuration object containing model and training settings.
        rank (int): The rank of the worker in a distributed setup.
        total_steps (int): Total number of training steps for scheduler initialization.
    """
    def __init__(self, config, rank, total_steps: int):
        self.config = config
        self.rank = rank

        # Data is already tokenized so we don't use tokenizer here, but for saving it in the checkpoint path together.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name_or_path, trust_remote_code=True)

        self.model, self.mpu, self.optimizer = initialize_model(config, rank)
        self.scheduler = get_scheduler(config, self.optimizer, total_steps)

    def step(self, input_batch: dict, train: bool):
        """
        Perform a single training or evaluation step.

        Args:
            input_batch (dict): A batch of input data.
            train (bool): Whether to perform a training step (True) or evaluation step (False).

        Returns:
            dict: A dictionary containing the loss and learning rate.
        """
        batch = {}
        for k, v in input_batch.items():
            batch[k] = v.cuda(non_blocking=True) if torch.is_tensor(v) else v

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            micro_batch_size = self.config.data.train_micro_batch_size_per_gpu
        else:
            self.model.eval()
            micro_batch_size = self.config.data.valid_micro_batch_size_per_gpu

        batch_size = (batch["position_ids"] == 0).sum().item()
        num_of_micro_batches = batch_size // micro_batch_size

        if self.config.model.pipeline_parallel_size > 1:
            pp_wrapper = self.model.__nanotron_wrappers__[ParallelMode.PIPELINE]
            pp_wrapper.micro_batch_size = micro_batch_size
            micro_batches = pp_wrapper.split_packed_batches(batch)
            micro_batch_iterator = enumerate(self.model(**batch))
        else:
            micro_batches = [split_packed_batch(batch, i, num_of_micro_batches) for i in range(num_of_micro_batches)]
            micro_batch_iterator = enumerate(micro_batches)

        device = batch["input_ids"].device
        sum_of_valid_losses = torch.zeros((), device=device, dtype=torch.float32)
        num_of_valid_losses = torch.zeros((), device=device, dtype=torch.float32)

        num_of_micro_valid_tokens_per_batch = [(m["labels"][:, 1:] != -100).sum() for m in micro_batches]
        num_of_total_valid_tokens = sum(num_of_micro_valid_tokens_per_batch).to(device).clamp_min(1)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for mico_idx, micro_input_or_output in micro_batch_iterator:
                    if self.config.model.pipeline_parallel_size > 1:
                        micro_loss = micro_input_or_output.loss
                    else:
                        micro_loss = self.model(**micro_input_or_output).loss

                    num_of_micro_valid_tokens = num_of_micro_valid_tokens_per_batch[mico_idx].to(device).detach()
                    sum_of_valid_losses += (micro_loss.detach() * num_of_micro_valid_tokens).float()
                    num_of_valid_losses += num_of_micro_valid_tokens.float()

                    if train:
                        contribution = num_of_micro_valid_tokens / num_of_total_valid_tokens
                        (micro_loss * contribution).backward()

            if train and self.optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.clip_grad)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        if self.config.model.data_parallel_size > 1:
            dist.all_reduce(sum_of_valid_losses, op=dist.ReduceOp.SUM, group=self.mpu.get_group(ParallelMode.DATA))
            dist.all_reduce(num_of_valid_losses, op=dist.ReduceOp.SUM, group=self.mpu.get_group(ParallelMode.DATA))

        final_loss = (sum_of_valid_losses / num_of_valid_losses.clamp_min(1.0)).item()
        lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": float(final_loss), "lr": float(lr)}

    def save_parallelized(self, save_dir: str):
        """
        Save the parallelized model and tokenizer to the specified directory.

        Args:
            save_dir (str): Directory to save the model and tokenizer.
        """
        self.model.save_parallelized(save_dir)
        if self.mpu is None or self.mpu.get_global_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        return {"ok": True, "save_dir": save_dir}
