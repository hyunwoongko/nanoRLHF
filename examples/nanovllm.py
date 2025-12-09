from nanorlhf.nanovllm import SamplingParams, LLM

prompts = [
    "User: What is nanoRLHF?\nAssistant:",
    "User: Write a poem about the ocean.\nAssistant:",
    "User: Explain the theory of relativity in simple terms.\nAssistant:",
    "User: What are the benefits of using renewable energy?\nAssistant:",
]

engine = LLM("Qwen/Qwen3-4B-base")
params = SamplingParams(max_tokens=32, temperature=0.8, top_p=0.7)
outputs = engine.generate(prompts, sampling_params=params)

print("-" * 50)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt:{repr(prompt)}")
    print(f"Output:{repr(output)}")
    print("-" * 50)
