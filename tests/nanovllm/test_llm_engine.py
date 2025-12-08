from nanorlhf.nanovllm.core.llm_engine import LLMEngine
from nanorlhf.nanovllm.utils.sampling_params import SamplingParams


def main():
    sampling = SamplingParams()
    sampling.max_tokens = 16
    sampling.temperature = 0.8

    prompt = "Hello. My name is"
    engine = LLMEngine("Qwen/Qwen2-B")
    outputs = engine.generate([prompt], sampling_params=sampling)

    for result in outputs:
        print("=== nanoVLLM smoke test ===")
        print(f"Prompt: {prompt}")
        print(f"Completion: {result['text']}")
        print(f"Token IDs: {result['token_ids']}")


if __name__ == "__main__":
    main()