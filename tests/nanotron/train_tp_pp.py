import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanorlhf.nanotron.api import PipelineParallel, TensorParallel
from nanorlhf.nanotron.distributed.mpu import MPU


def set_device_from_env() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def is_global_zero() -> bool:
    return dist.is_initialized() and dist.get_rank() == 0


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def toy_corpus():
    base = [
        "I love training language models with tensor and pipeline parallelism.",
        "Scaling laws suggest larger models perform better with enough data and compute.",
        "Nanotron provides simple wrappers for TP and PP without FSDP.",
        "GPT-2 is a small causal language model commonly used for experiments.",
        "Micro-batching helps keep pipeline bubbles small across stages.",
        "AdamW with cosine decay is often a solid default for fine-tuning.",
        "Korean and English mixed corpus can help multilingual capabilities.",
    ]
    return base * 64


def make_batch(tok, texts, seq_len, batch_size, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    x = enc["input_ids"][:, :seq_len]
    m = enc["attention_mask"][:, :seq_len]
    if x.size(0) < batch_size:
        reps = (batch_size + x.size(0) - 1) // x.size(0)
        x = x.repeat(reps, 1)[:batch_size]
        m = m.repeat(reps, 1)[:batch_size]
    y = x.clone()
    return {"input_ids": x.to(device), "attention_mask": m.to(device), "labels": y.to(device)}


def build_model(model_name: str, micro_bs: int, mpu: MPU, device: torch.device):
    base = AutoModelForCausalLM.from_pretrained(model_name)
    model = PipelineParallel(base, mpu, micro_batch_size=micro_bs)
    model = TensorParallel(model, mpu)
    model.parallelize()
    return model


def build_optimizer(model: nn.Module, lr: float, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight", "ln.weight"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--micro_batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save_dir", type=str, default="./ckpt_gpt2_tp2_pp2")
    args = ap.parse_args()

    device = set_device_from_env()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert world % 4 == 0 or world == 4, f"WORLD_SIZE={world} → TP=2 × PP=2 조합으로 실행해줘."

    mpu = MPU.from_torch(tensor_parallel_size=2, pipeline_parallel_size=2)
    tok = build_tokenizer(args.model_name)
    texts = toy_corpus()
    model = build_model(args.model_name, args.micro_batch_size, mpu, device)
    optim = build_optimizer(model, lr=args.lr, wd=args.weight_decay)
    model.train()
    start = time.time()

    for step in range(args.steps):
        batch = make_batch(tok, texts[step % len(texts): (step % len(texts)) + args.batch_size],
                           args.seq_len, args.batch_size, device)

        micro_losses = []
        for micro_out in model(**batch):
            loss = micro_out.loss
            loss.backward()
            micro_losses.append(loss.detach())

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % 10 == 0 and is_global_zero():
            ml = torch.stack(micro_losses).mean().item() if micro_losses else float("nan")
            ppl = math.exp(ml) if ml < 20 else float("inf")
            elapsed = time.time() - start
            print(f"[step {step:4d}/{args.steps}] loss={ml:.4f} ppl={ppl:.2f} elapsed={elapsed:.1f}s")

    model.deparallelize()

    if is_global_zero():
        os.makedirs(args.save_dir, exist_ok=True)
        target = model if hasattr(model, "save_pretrained") else getattr(model, "module", model)
        if hasattr(target, "save_pretrained"):
            target.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"Saved to {args.save_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
