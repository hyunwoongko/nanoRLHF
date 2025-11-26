import argparse
import json
import os
from typing import List, Dict

import datasets
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanorlhf.nanotron import PipelineParallel, TensorParallel, DataParallel, MPU, ParallelMode


class CausalLMDataset(Dataset):
    def __init__(self, rows: List[str], tokenizer: AutoTokenizer, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.rows[idx]
        enc = self.tok(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_determinism(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train_baseline(args, model, optimizer, loader, device, steps: int) -> List[float]:
    model.train()
    model.to(device)

    pbar = tqdm(range(steps), desc="[Baseline]")
    losses: List[float] = []
    it = iter(loader)

    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return losses


def train_parallel(args, model, optimizer, loader, device, steps: int, mpu: MPU) -> List[float]:
    model.train()
    dp_rank = mpu.get_local_rank(ParallelMode.DATA)
    pp_rank = mpu.get_local_rank(ParallelMode.PIPELINE)
    tp_rank = mpu.get_local_rank(ParallelMode.TENSOR)
    log_this_rank = dp_rank == 0 and pp_rank == 0 and tp_rank == 0

    if log_this_rank:
        desc = f"[3D] tp={args.tp}, pp={args.pp}, dp={args.dp}, stg={args.stg}"
        pbar = tqdm(range(steps), desc=desc)
    else:
        pbar = range(steps)

    dp_size = args.dp
    assert dp_size == mpu.get_world_size(ParallelMode.DATA)
    assert args.batch_size % dp_size == 0, "batch_size must be divisible by dp_size"
    local_batch_size = args.batch_size // dp_size
    assert local_batch_size > 0, "local batch size must be > 0"
    if args.pp > 1:
        assert (
            local_batch_size % args.micro_batch_size == 0
        ), f"local_batch_size({local_batch_size}) must be divisible by micro_batch_size({args.micro_batch_size})"

    losses: List[float] = []
    it = iter(loader)

    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        if dp_size > 1:
            # This is same as DistributedSampler.
            start_idx = dp_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            for k in batch:
                batch[k] = batch[k][start_idx:end_idx]

        optimizer.zero_grad(set_to_none=True)

        if args.pp > 1:
            micro_losses = []
            for out in model(**batch):
                loss = out.loss
                loss.backward()
                micro_losses.append(loss.detach())
            local_loss = torch.stack(micro_losses).mean()
        else:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            local_loss = loss.detach()

        if dp_size > 1:
            loss_sum = local_loss.clone()
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=mpu.get_group(ParallelMode.DATA))
            global_loss = loss_sum / dp_size
        else:
            global_loss = local_loss
        last_loss_val = float(global_loss.cpu())

        optimizer.step()

        if log_this_rank:
            losses.append(last_loss_val)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{last_loss_val:.4f}")

    if log_this_rank:
        return losses
    else:
        return []


def save_losses_json(losses: List[float], label: str, out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"loss_{label}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"label": label, "losses": losses}, f, indent=2)
    print(f"[save] wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--stg", type=int, default=0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    assert (
        world_size == args.tp * args.pp * args.dp
    ), f"world_size({world_size}) != tp*pp*dp ({args.tp}*{args.pp}*{args.dp})"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    set_determinism(args.seed)

    data = datasets.load_dataset("google-research-datasets/poem_sentiment", split="train")
    rows = list(data["verse_text"])
    dataset = CausalLMDataset(rows, tokenizer, max_length=args.max_length)

    if world_size == 1:
        assert args.dp == 1 and args.pp == 1 and args.tp == 1, "tp, pp, dp must be 1 in single GPU mode"

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        model_base = AutoModelForCausalLM.from_pretrained(args.model_name)
        model_base.config.pad_token_id = tokenizer.pad_token_id
        opt_base = AdamW(model_base.parameters(), lr=args.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        baseline_losses = train_baseline(args, model_base, opt_base, loader, device, args.steps)
        label = f"tp{args.tp}_pp{args.pp}_dp{args.dp}_stg{args.stg}"
        save_losses_json(baseline_losses, label, out_dir="./losses")
        return

    mpu = MPU.from_torch(
        data_parallel_size=args.dp,
        pipeline_parallel_size=args.pp,
        tensor_parallel_size=args.tp,
        seed=args.seed,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
    )

    if rank == 0:
        print(f"{model}\n\n")

    model = TensorParallel(model, mpu=mpu)
    model = PipelineParallel(model, mpu=mpu, micro_batch_size=args.micro_batch_size)
    model, optimizer = DataParallel(model, mpu=mpu, optimizer=optimizer, zero_stage=args.stg)
    model.parallelize()

    if rank == 0:
        print(f"{model}\n\n")

    device = torch.device(torch.cuda.current_device())
    losses = train_parallel(args, model, optimizer, loader, device, args.steps, mpu)

    if rank == 0 and len(losses) > 0:
        label = f"tp{args.tp}_pp{args.pp}_dp{args.dp}_stg{args.stg}"
        save_losses_json(losses, label, out_dir="./losses")

    dist.barrier()
    mpu.destroy()


if __name__ == "__main__":
    main()
