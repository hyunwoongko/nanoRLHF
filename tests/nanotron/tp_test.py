import torch
import torch.distributed as dist
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf.nanotron.api import TensorParallel
from nanorlhf.nanotron.distributed.mode import ParallelMode
from nanorlhf.nanotron.distributed.mpu import MPU


@torch.no_grad()
def all_gather_lastdim(x: torch.Tensor, group) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    shards = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(shards, x, group=group)
    return torch.cat(shards, dim=-1)


def main():
    torch.manual_seed(0)

    model_id = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token

    inputs = tok(["hello tensor parallel world!"], return_tensors="pt")
    labels = inputs["input_ids"].clone()

    ref_model = AutoModelForCausalLM.from_pretrained(model_id).eval()

    tp_model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    tp_model.load_state_dict(deepcopy(ref_model.state_dict()))

    mpu = MPU.from_torch(tensor_parallel_size=4)
    wrapper = TensorParallel(tp_model, mpu)
    wrapper.parallelize()

    with torch.no_grad():
        ref_out = ref_model(**inputs, labels=labels)
        ref_logits, ref_loss = ref_out.logits, ref_out.loss

    with torch.no_grad():
        tp_out = wrapper(**inputs, labels=labels)
        tp_group = mpu.get_group(ParallelMode.TENSOR)
        tp_logits_full = all_gather_lastdim(tp_out.logits, group=tp_group)
        tp_logits_full = tp_logits_full[..., : ref_logits.size(-1)]
        tp_loss = tp_out.loss

    wrapper.deparallelize()
    with torch.no_grad():
        detp_out = wrapper(**inputs, labels=labels)
        detp_logits, detp_loss = detp_out.logits, detp_out.loss

    a = ref_logits.float().cpu()
    b = tp_logits_full.float().cpu()
    c = detp_logits.float().cpu()

    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: ref={a.shape}, tp={b.shape}")

    atol = rtol = 1e-5
    logits_ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    loss_ok = torch.allclose(ref_loss.float().cpu(), tp_loss.float().cpu(), rtol=rtol, atol=atol)

    detp_logits_ok = torch.allclose(a, c, rtol=rtol, atol=atol)
    detp_loss_ok = torch.allclose(ref_loss.float().cpu(), detp_loss.float().cpu(), rtol=rtol, atol=atol)

    if dist.get_rank() == 0:
        diff = (a - b).abs()
        print(
            f"[LOGITS] allclose={logits_ok}  "
            f"max_abs_diff={diff.max().item():.6e}  mean_abs_diff={diff.mean().item():.6e}"
        )
        print(f"ref logits: {tuple(a.shape)}  tp logits(gathered): {tuple(b.shape)}")
        print(f"[LOSS]   ref={ref_loss.item():.8f}  tp={tp_loss.item():.8f}  allclose={loss_ok}")

        diff = (a - c).abs()
        print(
            f"[DE_TP LOGITS] allclose={detp_logits_ok}  "
            f"max_abs_diff={diff.max().item():.6e}  mean_abs_diff={diff.mean().item():.6e}"
        )
        print(f"ref logits: {tuple(a.shape)}  de_tp logits: {tuple(c.shape)}")
        print(f"[DE_TP LOSS]   ref={ref_loss.item():.8f}  detp={detp_loss.item():.8f}  allclose={detp_loss_ok}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
