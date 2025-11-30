from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf.kernels.patch import patch_kernel


model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda()
model = patch_kernel(model)

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you!"},
    {"role": "user", "content": "What is nanoRLHF?"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to("cuda")
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])
