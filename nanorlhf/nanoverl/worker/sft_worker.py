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
    model = AutoModelForCausalLM.from_pretrained(
        config.model.partial_pretrain,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

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

    total_world_size = (
        config.model.data_parallel_size * config.model.tensor_parallel_size * config.model.pipeline_parallel_size
    )

    assert total_world_size <= 8, (
        "Currently don't support multi-node training. "
        "Please set data_parallel_size * tensor_parallel_size * pipeline_parallel_size <= 8."
    )

    mpu = None
    if total_world_size > 1:
        assert rank < total_world_size, "rank must be < dp*tp*pp"
        mpu = MPU(
            rank=rank,
            local_rank=rank,
            world_size=total_world_size,
            local_world_size=total_world_size,
            host=config.model.host,
            port=config.model.port,
            data_parallel_size=config.model.data_parallel_size,
            pipeline_parallel_size=config.model.pipeline_parallel_size,
            tensor_parallel_size=config.model.tensor_parallel_size,
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
            micro_batch_size=config.data.train_micro_batch_size,
            gradient_checkpointing_enable=config.model.gradient_checkpointing_enable,
        )
        model, optimizer = DataParallel(
            model,
            mpu=mpu,
            optimizer=optimizer,
            zero_stage=config.model.zero_stage,
            accum_steps=config.data.train_batch_size // config.data.train_micro_batch_size,
        )
        model.parallelize()
    else:
        model.cuda()

    model = patch_kernel(model)
    return model, mpu, optimizer


@nanoray.actor
class SFTWorker:
    def __init__(self, config, rank, total_steps: int):
        self.config = config
        self.rank = rank

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.partial_pretrain, trust_remote_code=True)
        self.model, self.mpu, self.optimizer = initialize_model(config, rank)
        self.model.train()
        self.scheduler = get_scheduler(config, self.optimizer, total_steps)

    def step(self, input_batch: dict, train: bool):
        batch = {}
        for k, v in input_batch.items():
            batch[k] = v.cuda(non_blocking=True) if torch.is_tensor(v) else v

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            micro_batch_size = self.config.data.train_micro_batch_size
        else:
            self.model.eval()
            micro_batch_size = self.config.data.valid_micro_batch_size

        batch_size = (batch["position_ids"] == 0).sum().item()
        num_micro_batches = batch_size // micro_batch_size

        if self.config.model.pipeline_parallel_size > 1:
            pp_wrapper = self.model.__nanotron_wrappers__[ParallelMode.PIPELINE]
            pp_wrapper.micro_batch_size = micro_batch_size
            micro_batches = pp_wrapper._split_packed_batches(batch)
            micro_batch_iterator = enumerate(self.model(**batch))
        else:
            micro_batches = [split_packed_batch(batch, i, num_micro_batches) for i in range(num_micro_batches)]
            micro_batch_iterator = enumerate(micro_batches)

        device = batch["input_ids"].device
        sum_of_valid_losses = torch.zeros((), device=device, dtype=torch.float32)
        num_of_valid_losses = torch.zeros((), device=device, dtype=torch.float32)

        num_micro_valid_token_per_batch = [(m["labels"][:, 1:] != -100).sum() for m in micro_batches]
        num_total_valid_tokens = sum(num_micro_valid_token_per_batch).to(device).clamp_min(1)

        with torch.set_grad_enabled(train):
            for mico_idx, micro_input_or_output in micro_batch_iterator:
                if self.config.model.pipeline_parallel_size > 1:
                    micro_loss = micro_input_or_output.loss
                else:
                    micro_loss = self.model(**micro_input_or_output).loss

                num_micro_valid_tokens = num_micro_valid_token_per_batch[mico_idx].to(device).detach()
                sum_of_valid_losses += (micro_loss.detach() * num_micro_valid_tokens).float()
                num_of_valid_losses += num_micro_valid_tokens.float()

                if train:
                    contribution = num_micro_valid_tokens / num_total_valid_tokens
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
        self.model.save_parallelized(save_dir)
        if self.mpu is None or self.mpu.get_global_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        return {"ok": True, "save_dir": save_dir}
