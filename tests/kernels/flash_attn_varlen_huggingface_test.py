import torch
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer

from nanorlhf.kernels import patch_kernel


@torch.no_grad()
def main():
    set_seed(42)

    print("Loading models...")
    model_name = "Qwen/Qwen3-0.6B-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    model = patch_kernel(model)

    data = [
        "User: Write a Python function that computes the factorial of a number.\nAssistant:",
        "User: Explain the concept of blockchain technology in simple terms.\nAssistant:",
        "User: What are the benefits of using renewable energy sources?\nAssistant:",
        "User: How does a neural network learn from data?\nAssistant:",
    ]

    path: str = "c"  # "a" | "b" | "c"
    # import os
    # os.environ['nanorlhf-attn-path'] = path

    def prepare_inputs(path: str):
        device = "cuda"

        if path in ['a', 'c']:
            inputs = tokenizer(
                data,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                add_special_tokens=True,
            )
            return {k: v.to(device) for k, v in inputs.items()}

        if path == "b":
            tokenized = tokenizer(
                data,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            packed_ids: list[int] = []
            packed_pos: list[int] = []

            for ids in tokenized:
                packed_ids.extend(ids)
                packed_pos.extend(list(range(len(ids))))

            input_ids = torch.tensor([packed_ids], dtype=torch.long, device=device)
            position_ids = torch.tensor([packed_pos], dtype=torch.long, device=device)

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
            }

        raise ValueError(f"Unknown path: {path}")

    inputs = prepare_inputs(path)
    input_lengths = (
        inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).tolist()
        if path in ["a", 'c']
        else [inputs["input_ids"].shape[1]]
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    for i in range(outputs.size(0)):
        gen_ids = outputs[i, input_lengths[i] :]
        print(f"Output {i}: {repr(tokenizer.decode(gen_ids, skip_special_tokens=True))}")
        print("-" * 50)


if __name__ == "__main__":
    main()
