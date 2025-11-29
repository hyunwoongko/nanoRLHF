from transformers import AutoModelForCausalLM, AutoTokenizer
import nanorlhf  # import to register the flash attention implementation

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="nanorlhf_flash_attention").eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you!"},
    {"role": "user", "content": "Can you tell me what is nanoRLHF?"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to("cuda")
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])
