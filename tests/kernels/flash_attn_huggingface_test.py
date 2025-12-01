import torch
from transformers import AutoModelForCausalLM, set_seed
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nanorlhf.kernels.utils.huggingface import flash_attention_forward

if "nanoRLHF" not in ALL_ATTENTION_FUNCTIONS:
    ALL_ATTENTION_FUNCTIONS["nanoRLHF"] = flash_attention_forward

set_seed(42)

print("Loading models...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()

vocab_size = model.config.vocab_size
input_ids = torch.randint(low=0, high=vocab_size, size=(8, 1024), device="cuda", dtype=torch.long)
attention_mask = torch.ones_like(input_ids)
eager_logits = None

for attn_impl in ["eager", "nanoRLHF", "flash_attention_2"]:
    model.config._attn_implementation = attn_impl

    for _ in range(3):
        # Warming up
        model(input_ids, attention_mask=attention_mask, use_cache=False)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)

    print(f"Implementation: {attn_impl}")
    print(f"Time taken: {elapsed_time:.2f} ms")

    if attn_impl == "eager":
        eager_logits = logits
    else:
        max_diff = (eager_logits.float() - logits.float()).abs().max().item()
        mean_diff = (eager_logits.float() - logits.float()).abs().mean().item()
        print(f"Max difference with eager: {max_diff:.6f}")
        print(f"Mean difference with eager: {mean_diff:.6f}")
    print("-" * 50)

"""
Implementation: eager
Time taken: 96.42 ms
--------------------------------------------------
Implementation: nanoRLHF
Time taken: 61.06 ms
Max difference with eager: 1.371094
Mean difference with eager: 0.036597
--------------------------------------------------
Implementation: flash_attention_2
Time taken: 50.17 ms
Max difference with eager: 1.546875
Mean difference with eager: 0.036664
--------------------------------------------------
"""
