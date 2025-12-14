import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.kernels import patch_kernel
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanoverl.utils.optim_utils import get_optimizer_param_groups, get_scheduler


def initialize_model(config, rank, enable_gradient: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        config.model.partial_pretrain,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if enable_gradient:
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
    else:
        optimizer = None
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

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
            zero_stage=config.model.zero_stage if enable_gradient else 0,
        )
        model.parallelize()
    else:
        model.cuda()

    model = patch_kernel(model)
    return model, optimizer, mpu


@nanoray.actor
class ActorRolloutRef:
    def __init__(self, config, rank, total_steps: int, initialize_ref: bool = False, initialize_rollout: bool = False):
        self.config = config
        self.rank = rank

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.partial_pretrain, trust_remote_code=True)
        self.actor, self.optimizer, self.actor_mpu = initialize_model(config, rank, enable_gradient=True)
        self.actor.train()
        self.scheduler = get_scheduler(config, self.optimizer, total_steps)

        if initialize_ref:
            raise NotImplementedError
        if initialize_rollout:
            raise NotImplementedError

    def step(self, input_batch: dict, train: bool):
        batch = {}
        for k, v in input_batch.items():
            batch[k] = v.cuda(non_blocking=True) if torch.is_tensor(v) else v

        if train:
            self.actor.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.actor.eval()

        loss_num = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float32)
        loss_den = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float32)

        with torch.set_grad_enabled(train):
            if self.config.model.pipeline_parallel_size > 1:
                pp_wrapper = self.actor.__nanotron_wrappers__[ParallelMode.PIPELINE]
                pp_wrapper.micro_batch_size = (
                    self.config.data.train_micro_batch_size if train else self.config.data.valid_micro_batch_size
                )
                micro_batches = pp_wrapper._split_packed_batches(batch)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for micro_idx, micro_output in enumerate(self.actor(**batch)):
                        assert micro_idx < len(micro_batches)
                        micro_batch = micro_batches[micro_idx]
                        micro_loss = micro_output.loss

                        if train:
                            micro_loss.backward()

                        micro_denom = (micro_batch["labels"][:, 1:] != -100).sum()
                        micro_denom = micro_denom.to(dtype=micro_loss.dtype, device=micro_loss.device)
                        loss_num += (micro_loss.detach() * micro_denom.detach()).float()
                        loss_den += micro_denom.detach().float()
            else:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = self.actor(**batch).loss

                    if train:
                        loss.backward()

                    denom = (batch["labels"][:, 1:] != -100).sum()
                    denom = denom.to(dtype=loss.dtype, device=loss.device)
                    loss_num += (loss.detach() * denom).float()
                    loss_den += denom.float()

            if train and self.optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.optim.clip_grad)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        if self.config.model.data_parallel_size > 1:
            dist.all_reduce(loss_num, op=dist.ReduceOp.SUM, group=self.actor_mpu.get_group(ParallelMode.DATA))
            dist.all_reduce(loss_den, op=dist.ReduceOp.SUM, group=self.actor_mpu.get_group(ParallelMode.DATA))

        final_loss = (loss_num / loss_den.clamp_min(1.0)).item()
        lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": float(final_loss), "lr": float(lr)}

    def save_parallelized(self, save_dir: str):
        self.actor.save_parallelized(save_dir)
        if self.actor_mpu is None or self.actor_mpu.get_global_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        return {"ok": True, "save_dir": save_dir}
