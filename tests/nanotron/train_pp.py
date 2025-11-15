import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import datasets
import matplotlib
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanorlhf.nanotron.distributed.mpu import MPU
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.api import PipelineParallel

MAX_LEN = 256


class RowCausalLMDataset(Dataset):

    def __init__(self, rows: List[str], tokenizer: AutoTokenizer, max_length: int = MAX_LEN):
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
        input_ids = enc["input_ids"].squeeze(0)  # [L]
        attention_mask = enc["attention_mask"].squeeze(0)  # [L]
        labels = input_ids.clone()  # causal LM: next-token prediction
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def set_hf_gpt2_padding(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def set_determinism(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass
class Args:
    model_name: str = "gpt2"
    steps: int = 512
    batch_size: int = 8
    lr: float = 5e-6
    seed: int = 1234
    micro_batch_size: int = 2
    use_cache: bool = False
    max_length: int = MAX_LEN


def train_baseline(args: Args, model, optimizer, loader, device, steps: int) -> List[float]:
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

        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return losses


def train_pipeline(args: Args, model_pp, optimizer, loader, device, steps: int, mpu: MPU) -> List[float]:
    model_pp.train()
    if mpu.get_local_rank(ParallelMode.PIPELINE) == 0:
        pbar = tqdm(range(steps), desc="[Pipeline]")
    else:
        pbar = range(steps)

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
        last_loss: Optional[float] = None
        for out in model_pp(**batch):
            loss = out.loss
            loss.backward()
            last_loss = float(loss.detach().cpu())

        optimizer.step()

        if mpu.get_local_rank(ParallelMode.PIPELINE) == 0 and last_loss is not None:
            losses.append(last_loss)
            pbar.set_postfix(loss=f"{last_loss:.4f}")

    if mpu.get_local_rank(ParallelMode.PIPELINE) == 0:
        return losses
    else:
        return []


def plot_losses(baseline: Optional[List[float]], pipeline: Optional[List[float]], out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)

    if baseline is not None and len(baseline) > 0:
        plt.figure(figsize=(9, 4.5))
        plt.plot(range(len(baseline)), baseline)
        plt.title("Baseline: loss per step")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "baseline_loss.png"))
        plt.close()

    if pipeline is not None and len(pipeline) > 0:
        plt.figure(figsize=(9, 4.5))
        plt.plot(range(len(pipeline)), pipeline)
        plt.title("Pipeline Parallel: loss per step")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pipeline_loss.png"))
        plt.close()

    if baseline is not None and pipeline is not None and len(baseline) > 0 and len(pipeline) > 0:
        n = min(len(baseline), len(pipeline))
        plt.figure(figsize=(10, 5))
        plt.plot(range(n), baseline[:n], label="Baseline")
        plt.plot(range(n), pipeline[:n], label="Pipeline")
        plt.title("Baseline vs Pipeline: loss per step")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "combined_loss.png"))
        plt.close()


def main():
    args = Args()

    mpu = MPU.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=4,
        tensor_parallel_size=1,
        backend="nccl",
        seed=args.seed,
    )
    mpu.set_device()
    device = torch.device(torch.cuda.current_device())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    set_hf_gpt2_padding(tokenizer)

    data = datasets.load_dataset("google-research-datasets/poem_sentiment", split="train")
    rows = list(data["verse_text"])
    dataset = RowCausalLMDataset(rows, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    set_determinism(args.seed)

    if dist.get_rank() == 0:
        model_base = AutoModelForCausalLM.from_pretrained(args.model_name)
        model_base.config.use_cache = args.use_cache
        model_base.config.pad_token_id = tokenizer.pad_token_id
        opt_base = AdamW(model_base.parameters(), lr=args.lr)

        baseline_losses = train_baseline(args, model_base, opt_base, loader, device, args.steps)
    else:
        baseline_losses = None

    model_pp_raw = AutoModelForCausalLM.from_pretrained(args.model_name)
    model_pp_raw.config.use_cache = args.use_cache
    model_pp_raw.config.pad_token_id = tokenizer.pad_token_id

    model_pp = PipelineParallel(model_pp_raw, mpu=mpu, micro_batch_size=args.batch_size // 2)  # 예: 2-way micro
    model_pp.parallelize()

    opt_pp = AdamW(model_pp.parameters(), lr=args.lr)

    pp_losses = train_pipeline(args, model_pp, opt_pp, loader, device, args.steps, mpu)

    if dist.get_rank() == 0:
        print("\n=== DONE ===")
        print(f"Baseline steps: {len(baseline_losses)}")
        print(f"PP steps (rank0): {len(pp_losses)}")

        n = min(len(baseline_losses) if baseline_losses is not None else 0, len(pp_losses), 16)
        if n > 0:
            print("Compare first 16 steps:")
            for i in range(n):
                b = baseline_losses[i]
                p = pp_losses[i]
                print(f"  step {i:03d}  baseline={b:.4f}  pp={p:.4f}  |diff|={abs(b - p):.4f}")

        out_dir = "./plots_pp_vs_baseline"
        plot_losses(baseline_losses, pp_losses, out_dir)
        print(f"\nSaved plots to: {out_dir}")
        print(" - baseline_loss.png")
        print(" - pipeline_loss.png")
        print(" - combined_loss.png")

    dist.barrier()
    mpu.destroy()


if __name__ == "__main__":
    main()
