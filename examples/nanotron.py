"""
For training model:
torchrun --nproc_per_node=8 nanotron.py --mode train

For merging checkpoint:
torchrun --nproc_per_node=8 nanotron.py --mode ckpt

For testing model:
python3 nanotron.py --mode test
"""

from argparse import ArgumentParser
from typing import List, Dict

import datasets
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanorlhf.nanosets import Dataset
from nanorlhf.nanotron import MPU, ParallelMode, TensorParallel, PipelineParallel, DataParallel


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


def train_model(model, tokenizer, mpu, optimizer, loader, train_steps, save_iter):
    model.train()
    iterator = iter(loader)
    pbar = tqdm(range(train_steps)) if mpu.get_global_rank() == 0 else range(train_steps)

    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        for k in batch:
            batch[k] = batch[k].to(torch.cuda.current_device(), non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if mpu.get_world_size(ParallelMode.PIPELINE) > 1:
            micro_losses = []
            for out in model(**batch):
                micro_loss = out.loss
                micro_loss.backward()
                micro_losses.append(micro_loss.detach())
            local_loss = torch.stack(micro_losses).mean()
        else:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            local_loss = loss.detach()

        if mpu.get_world_size(ParallelMode.DATA) > 1:
            loss_sum = local_loss.clone()
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=mpu.get_group(ParallelMode.DATA))
            global_loss = loss_sum / mpu.get_world_size(ParallelMode.DATA)
        else:
            global_loss = local_loss

        optimizer.step()

        if mpu.get_global_rank() == 0:
            pbar.set_postfix(loss=f"{global_loss.item():.4f}")

        if (save_iter > 0) and ((step + 1) % save_iter == 0):
            shard_dir = f"./ckpt/global_step_{step+1}"
            tokenizer.save_pretrained(shard_dir)
            model.save_parallelized(shard_dir, merge_checkpoints=False)
            with open("./ckpt/latest_checkpointed_iteration.txt", "w") as f:
                f.write(str(step + 1))

            if mpu.get_global_rank() == 0:
                print(f"[SAVE] Saved checkpoint at step {step+1} to {shard_dir}")


def merge_ckpt(model, tokenizer, mpu):
    latest_checkpoint_file = "./ckpt/latest_checkpointed_iteration.txt"
    with open(latest_checkpoint_file, "r") as f:
        latest_step = int(f.read().strip())
    shard_dir = f"./ckpt/global_step_{latest_step}"

    model.from_parallelized(shard_dir)
    if mpu.get_global_rank() == 0:
        print(f"[LOAD] Loaded checkpoint from {shard_dir} for merging.")

    model.save_parallelized("./ckpt/merged_checkpoint", merge_checkpoints=True)
    if mpu.get_global_rank() == 0:
        print(f"[SAVE] Saved merged checkpoint to ./ckpt/merged_checkpoint")

    tokenizer.save_pretrained("./ckpt/merged_checkpoint")


def test_model():
    model = AutoModelForCausalLM.from_pretrained("./ckpt/merged_checkpoint").eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained("./ckpt/merged_checkpoint")
    print("[LOAD] Loaded merged checkpoint for testing.")

    prompts = [
        "with pale blue berries,",
        "the which she bearing home it burned her nest,",
        "pilgrim and soldier,",
        "and then he shut his little eyes,",
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_p=0.7, temperature=0.6)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("-" * 40)
    for i, text in enumerate(output_texts):
        print(f"[PROMPT] {prompts[i]}")
        print(f"[GENERATED] {text}")
        print("-" * 40)


if __name__ == '__main__':
    parser = ArgumentParser()
    # mode argument
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'ckpt', 'test'])
    # model and hyperparameter arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--save_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    # parallelism arguments
    parser.add_argument("--data_parallel_size", type=int, default=2)
    parser.add_argument("--pipeline_parallel_size", type=int, default=2)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--zero_stage", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "test":
        # testing does not require distributed setup
        test_model()
    else:
        # create mpu for distributed training or checkpoint merging
        mpu = MPU.from_torch(
            data_parallel_size=args.data_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
        )

        # load tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        raw_data = datasets.load_dataset("google-research-datasets/poem_sentiment", split="train")
        dataset = CausalLMDataset(list(raw_data["verse_text"]), tokenizer, max_length=args.max_length)

        # create data loader
        if args.data_parallel_size > 1:
            shuffle_by_loader = False
            sampler = DistributedSampler(
                dataset,
                num_replicas=mpu.get_world_size(ParallelMode.DATA),
                rank=mpu.get_local_rank(ParallelMode.DATA),
                shuffle=True,
            )
        else:
            shuffle_by_loader = True
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_by_loader,
            pin_memory=True,
            drop_last=True,
        )

        # create model and optimizer
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        optimizer = AdamW(model.parameters(), lr=args.lr)

        # apply 3D parallelism
        model = TensorParallel(model, mpu=mpu)
        model = PipelineParallel(model, mpu=mpu, micro_batch_size=args.micro_batch_size)
        model, optimizer = DataParallel(model, mpu=mpu, optimizer=optimizer, zero_stage=args.zero_stage)
        model.parallelize()

        # run training or checkpoint merging
        if args.mode == "train":
            train_model(model, tokenizer, mpu, optimizer, loader, args.train_steps, args.save_iter)
        elif args.mode == "ckpt":
            merge_ckpt(model, tokenizer, mpu)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        # clean up mpu
        mpu.destroy()
