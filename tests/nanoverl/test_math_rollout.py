if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanorlhf.nanovllm import LLM, SamplingParams

    prompt = """Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.

{}"""
    problem = """Evaluate the integral \\( \\int_{\\gamma} \\frac{e^{2 \\pi z}}{(z+i)^3}dz \\) using the Cauchy Integration Formula, where \\( \\gamma(t)=2e^{it}, t \\in [0,2 \\pi] \\). Determine if the calculation \\( \\int_{\\gamma} \\frac{f(z)}{z-0}dz = 2 \\pi i f(0) = 0 \\) is correct."""


    input_text = prompt.format(problem)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").cuda()

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        enable_thinking=False,
        return_tensors="pt",
        add_generation_prompt=True,
    ).cuda()

    input_text = tokenizer.decode(inputs[0])
    print("Input text:")
    print(input_text)

    # outputs = model.generate(
    #     inputs,
    #     max_new_tokens=2048,
    #     temperature=0.0,
    #     do_sample=False,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    #
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #
    # print("Huggingface Output:")
    # print(generated_text)

    # llm = LLM("Qwen/Qwen3-1.7B", tensor_parallel_size=4)
    # sampling_params = SamplingParams(
    #     max_tokens=2048,
    #     temperature=0.0,
    # )
    #
    # output = llm.generate([input_text, "1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8, 8 + 8 ="], sampling_params)

    # print("NanoVLLM Output:")
    # print(output)

    llm = LLM("Qwen/Qwen3-1.7B", tensor_parallel_size=2, data_parallel_size=2)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
    )

    output = llm.generate([input_text, "1 + 1 = 2, 2 + 2 = 4, 4 + 4 = 8, 8 + 8 ="], sampling_params)

    print("NanoVLLM Output:")
    print(output)