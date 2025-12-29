"""
python3 -m nanorlhf.evaluation.math_eval \
    --model <model_name_or_path> \
    --test MATH-500
"""

import json
import os
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Optional

from math_verify import parse, verify
from transformers import AutoTokenizer

from nanorlhf.eval.utils import get_unnormalized_answer
from nanorlhf.nanosets import load_dataset, Dataset
from nanorlhf.nanovllm import LLM, SamplingParams

from transformers import PreTrainedTokenizerBase


def generate(
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    sampling_params: SamplingParams,
    dataset: Dataset,
    formatting_prompt: Optional[str] = None,
) -> List[dict]:
    """
    Generate model outputs for the given dataset.

    Args:
        llm (LLM): The language model to use for generation.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for the model.
        sampling_params (SamplingParams): Sampling parameters for generation.
        dataset (Dataset): The dataset containing problems.
        formatting_prompt (Optional[str]): Path to the formatting prompt file.

    Returns:
        List[dict]: List of model output dictionaries.
    """
    if formatting_prompt is not None:
        formatting_prompt = json.load(open(formatting_prompt, "r"))["prompt"]

    prompts = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if formatting_prompt is not None:
            prompt = formatting_prompt.format(sample['problem'])
        else:
            prompt = sample["problem"]

        messages = [{"role": "user", "content": prompt}]
        messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(messages)

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return outputs


def evaluate(model_outputs: list, dataset: Dataset) -> Tuple[dict, list]:
    """
    Evaluate model outputs against the dataset solutions.

    Args:
        model_outputs (list): List of model output dictionaries.
        dataset (Dataset): The dataset containing problems and solutions.

    Returns:
        Tuple[dict, list]: Evaluation results and model outputs for saving.
    """
    accuracy = 0
    model_outputs_for_saving = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        model_text = model_outputs[idx]["text"]

        if sample["answer"] is None:
            gold_answer = parse(get_unnormalized_answer(sample["solution"]))
        else:
            if "boxed" not in sample["answer"]:
                gold_answer = parse("\\boxed{" + str(sample["answer"]) + "}")
            else:
                gold_answer = parse(sample["answer"])

        model_answer = parse(get_unnormalized_answer(model_text))
        accuracy += int(verify(gold_answer, model_answer))
        model_outputs_for_saving.append(
            {
                "problem": sample["problem"],
                "model_text": model_text,
                "gold_text": sample["solution"],
                "model_answer": str(model_answer),
                "gold_answer": str(gold_answer),
            }
        )

    accuracy /= len(dataset)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result, model_outputs_for_saving


def main(
    args: Namespace,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: PreTrainedTokenizerBase,
):
    """
    Main evaluation loop for math problems.

    Args:
        args (Namespace): Command line arguments.
        llm (LLM): The language model to use for generation.
        sampling_params (SamplingParams): Sampling parameters for generation.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for the model.
    """
    print("Loading test dataset...")
    dataset = load_dataset(f"./data/{args.test}/test.jsonl")
    print("Generating model answers...")

    model_outputs = generate(llm, tokenizer, sampling_params, dataset, args.formatting_prompt)
    print("Evaluating model answers...")
    eval_output, model_outputs_for_saving = evaluate(model_outputs, dataset)
    print(f"Evaluation result: {eval_output}")

    paths = args.model.split("/")
    model_step_path = f"{paths[-2]}_{paths[-1]}"
    eval_result_dir = os.path.join("eval", model_step_path, args.test)
    os.makedirs(eval_result_dir, exist_ok=True)

    eval_result_path = os.path.join(eval_result_dir, "score.json")
    with open(eval_result_path, "w") as f:
        json.dump(eval_output, f)

    output_path = os.path.join(eval_result_dir, "model_outputs.jsonl")
    with open(output_path, "w") as f:
        for model_output in model_outputs_for_saving:
            json.dump(model_output, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved evaluation result to {eval_result_path} 😊")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--test", type=str, default="MATH-500", help="Datasets to evaluate on. comma-separated.")
    parser.add_argument("--formatting_prompt", type=str, default=None, help="Path to the formatting prompt file.")
    args = parser.parse_args()

    llm = LLM(args.model)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datasets = args.test.split(",")
    for test_data in datasets:
        args.test = test_data
        main(args, llm, sampling_params, tokenizer)
