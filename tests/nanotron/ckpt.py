import argparse
import os
from typing import Dict

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf.nanotron import PipelineParallel, TensorParallel, ParallelMode, MPU, DataParallel


def set_determinism(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_rank_info(mpu: MPU):
    return (
        mpu.get_local_rank(ParallelMode.TENSOR),
        mpu.get_local_rank(ParallelMode.PIPELINE),
        mpu.get_local_rank(ParallelMode.DATA),
    )


def allclose_state_dict(
    sd1: Dict[str, torch.Tensor],
    sd2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    if sd1.keys() != sd2.keys():
        return False
    for k in sd1.keys():
        t1 = sd1[k]
        t2 = sd2[k]
        if t1.shape != t2.shape:
            return False
        if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
            return False
    return True


def run_test(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    mpu = MPU.from_torch(
        data_parallel_size=args.dp,
        pipeline_parallel_size=args.pp,
        tensor_parallel_size=args.tp,
        seed=args.seed,
    )
    device = torch.device(torch.cuda.current_device())
    set_determinism(args.seed)

    if rank == 0:
        print(f"[INFO] world_size={world_size}, tp={args.tp}, pp={args.pp}, dp={args.dp}, stg={args.stg}")
        print(f"[INFO] using model: {args.model_name}")

    if rank == 0:
        baseline_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        baseline_model.eval()
        baseline_model.to("cuda")
        baseline_sd = {k: v.detach().cpu().clone() for k, v in baseline_model.state_dict().items()}
    else:
        baseline_model = None
        baseline_sd = None

    if dist.is_initialized():
        dist.barrier()

    base_for_parallel = AutoModelForCausalLM.from_pretrained(args.model_name)
    optimizer = torch.optim.AdamW(base_for_parallel.parameters(), lr=5e-7)
    parallel_model = base_for_parallel
    parallel_model.eval()
    parallel_model.to(device)

    parallel_model = PipelineParallel(parallel_model, mpu=mpu, micro_batch_size=args.micro_batch_size)
    parallel_model = TensorParallel(parallel_model, mpu=mpu)
    parallel_model, zero_optimizer = DataParallel(parallel_model, mpu=mpu, optimizer=optimizer, zero_stage=args.stg)
    parallel_model.parallelize()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "Hello, Qwen3 parallelism test!"
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=args.seq_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    shard_dir = os.path.join(args.out_dir, f"ckpt_shard_tp{args.tp}_pp{args.pp}_dp{args.dp}_stg{args.stg}")
    os.makedirs(shard_dir, exist_ok=True)

    if rank == 0:
        print(f"[TEST] save_parallelized(merge_checkpoints=False) -> {shard_dir}")

    parallel_model.save_parallelized(
        save_directory=shard_dir,
        merge_checkpoints=False,
        save_config=True,
    )

    if dist.is_initialized():
        dist.barrier()

    reloaded_parallel_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    reloaded_optimizer = torch.optim.AdamW(base_for_parallel.parameters(), lr=5e-7)
    reloaded_parallel_model.eval()
    reloaded_parallel_model.to(device)
    reloaded_parallel_model = PipelineParallel(
        reloaded_parallel_model, mpu=mpu, micro_batch_size=args.micro_batch_size
    )
    reloaded_parallel_model = TensorParallel(reloaded_parallel_model, mpu=mpu)
    reloaded_parallel_model, reloaded_zero_optimizer = DataParallel(
        reloaded_parallel_model, mpu=mpu, zero_stage=args.stg, optimizer=reloaded_optimizer
    )
    reloaded_parallel_model.parallelize()

    reloaded_parallel_model.from_parallelized(shard_dir, strict=False)
    orig_sd = {k: v.detach().cpu() for k, v in parallel_model.state_dict().items()}
    rel_sd = {k: v.detach().cpu() for k, v in reloaded_parallel_model.state_dict().items()}

    shard_ok = allclose_state_dict(orig_sd, rel_sd, rtol=1e-5, atol=1e-5)
    shard_ok_tensor = torch.tensor(1 if shard_ok else 0, device=device, dtype=torch.int32)

    if dist.is_initialized():
        dist.all_reduce(shard_ok_tensor, op=dist.ReduceOp.MIN)
    global_shard_ok = shard_ok_tensor.item() == 1

    if rank == 0:
        print(f"[RESULT] shard param allclose: {global_shard_ok}")

    with torch.no_grad():
        out1 = parallel_model(input_ids=input_ids, attention_mask=attention_mask)
        out2 = reloaded_parallel_model(input_ids=input_ids, attention_mask=attention_mask)

        if args.pp > 1:
            logits1 = torch.stack([o.logits for o in out1], dim=0).detach().cpu()
            logits2 = torch.stack([o.logits for o in out2], dim=0).detach().cpu()
        else:
            logits1 = out1.logits.detach().cpu()
            logits2 = out2.logits.detach().cpu()

    logits_shard_ok_local = torch.allclose(logits1, logits2, rtol=1e-5, atol=1e-5)
    logits_shard_ok_tensor = torch.tensor(1 if logits_shard_ok_local else 0, device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(logits_shard_ok_tensor, op=dist.ReduceOp.MIN)
    global_logits_shard_ok = logits_shard_ok_tensor.item() == 1

    if rank == 0:
        print(f"[RESULT] shard logits allclose: {global_logits_shard_ok}")

    merged_dir = os.path.join(args.out_dir, "ckpt_merged")
    os.makedirs(merged_dir, exist_ok=True)

    if rank == 0:
        print(f"[TEST] save_parallelized(merge_checkpoints=True) -> {merged_dir}")

    parallel_model.save_parallelized(
        save_directory=merged_dir,
        merge_checkpoints=True,
        save_config=True,
    )

    if dist.is_initialized():
        dist.barrier()

    merged_param_ok = True
    merged_logits_ok = True

    if rank == 0:
        merged_path = os.path.join(merged_dir, "pytorch_model.bin")
        if not os.path.isfile(merged_path):
            raise FileNotFoundError(f"merged checkpoint not found: {merged_path}")

        state_dict_merged = torch.load(merged_path, map_location="cpu")

        merged_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        merged_model.eval()
        merged_model.to("cuda")

        missing, unexpected = merged_model.load_state_dict(state_dict_merged, strict=False)
        if missing or unexpected:
            print(f"[WARN] merged load_state_dict missing={missing}, unexpected={unexpected}")

        merged_sd = {k: v.detach().cpu() for k, v in merged_model.state_dict().items()}
        merged_param_ok = allclose_state_dict(baseline_sd, merged_sd, rtol=1e-5, atol=1e-5)
        print(f"[RESULT] merged param allclose with baseline(single GPU): {merged_param_ok}")

        baseline_model.eval()
        merged_model.eval()
        with torch.no_grad():
            base_out = baseline_model(input_ids=input_ids.to("cuda"), attention_mask=attention_mask.to("cuda"))
            merg_out = merged_model(input_ids=input_ids.to("cuda"), attention_mask=attention_mask.to("cuda"))

        base_logits = base_out.logits.detach().cpu()
        merg_logits = merg_out.logits.detach().cpu()
        merged_logits_ok = torch.allclose(base_logits, merg_logits, rtol=1e-5, atol=1e-5)
        print(f"[RESULT] merged logits allclose with baseline(single GPU): {merged_logits_ok}")

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        if global_shard_ok and global_logits_shard_ok and merged_param_ok and merged_logits_ok:
            print("\n[OK] ALL TESTS PASSED (merge_checkpoints True/False)")
        else:
            print("\n[FAIL] Some tests failed.")
            print(f"  shard_param_ok={global_shard_ok}")
            print(f"  shard_logits_ok={global_logits_shard_ok}")
            print(f"  merged_param_ok={merged_param_ok}")
            print(f"  merged_logits_ok={merged_logits_ok}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--stg", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="./ckpts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
