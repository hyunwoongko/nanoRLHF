from transformers.utils import logging

logging.set_verbosity_error()

import time

import datasets
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import modeling_utils
from nanorlhf.kernels.utils.huggingface import flash_attention_forward
from nanorlhf.nanovllm import LLM, SamplingParams

prompts = list(datasets.load_dataset("google-research-datasets/poem_sentiment", split="train")["verse_text"])
warm_ups = ["This is a warm-up prompt.", "This is also a warm-up prompt."]


def nano_generation(model_name, max_new_tokens, temperature):
    params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
    engine = LLM(model_name)
    engine.generate(warm_ups, sampling_params=params)

    start = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params=params)
    outputs_decoded = [output['text'] for output in outputs]
    end = time.perf_counter()
    torch.cuda.synchronize()
    elapsed = end - start
    return outputs_decoded, elapsed


def hf_generation(model_name, max_new_tokens, temperature):
    hf_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=getattr(hf_config, "torch_dtype", torch.float16),
    )

    if "nanoRLHF" not in modeling_utils.ALL_ATTENTION_FUNCTIONS:
        modeling_utils.ALL_ATTENTION_FUNCTIONS["nanoRLHF"] = flash_attention_forward
    if not hasattr(model.config, "_attention_implementation"):
        model.config._attention_implementation = "nanoRLHF"

    warm_ups_tokenized = tokenizer(warm_ups, return_tensors="pt", padding=True).to(model.device)
    model.generate(
        **warm_ups_tokenized, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature
    )

    max_batch_size = 32
    batched_outputs_decoded = []
    start = time.perf_counter()
    for i in tqdm(range(0, len(prompts), max_batch_size), desc="HF Generating"):
        batch_prompts = prompts[i : i + max_batch_size]
        prompts_tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **prompts_tokenized, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature
        )
        outputs_decoded = tokenizer.batch_decode(outputs[:, prompts_tokenized['input_ids'].shape[1] :])
        batched_outputs_decoded.extend(outputs_decoded)
    end = time.perf_counter()
    torch.cuda.synchronize()
    elapsed = end - start
    return batched_outputs_decoded, elapsed


if __name__ == "__main__":
    max_new_tokens = 32
    temperature = 0.7

    nano_outputs, nano_time = nano_generation("Qwen/Qwen3-0.6B-base", max_new_tokens, temperature)
    hf_outputs, hf_time = hf_generation("Qwen/Qwen3-0.6B-base", max_new_tokens, temperature)

    for i in range(10):
        print(f"Prompt {i+1}: {repr(prompts[i])}")
        print(f"HuggingFace Output: {repr(hf_outputs[i])}")
        print(f"NanoVLLM Output: {repr(nano_outputs[i])}")
        print("-" * 50)

    print(f"HuggingFace generation time: {hf_time:.2f} seconds")
    print(f"NanoVLLM generation time: {nano_time:.2f} seconds")
