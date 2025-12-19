"""
python3 -m nanorlhf.evaluation.math_eval \
    --model <model_name_or_path> \
    --test MATH-500
"""

import json
import os
from argparse import ArgumentParser

from math_verify import parse, verify
from transformers import AutoTokenizer

from nanorlhf.evaluation.eval_utils import get_unnormalized_answer
from nanorlhf.nanosets import load_dataset
from nanorlhf.nanovllm import LLM, SamplingParams


def load_test_dataset(test):
    data_path = f"./data/{test}/test.jsonl"
    return load_dataset(data_path)


def generate_model_answer(model, dataset, formatting_prompt):
    sampling_params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=1.0)
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model)

    prompts = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if formatting_prompt is not None:
            formatting_prompt = json.load(open(formatting_prompt, "r")).prompt
            prompt = formatting_prompt.format(sample['problem'])
        else:
            prompt = sample["problem"]

        messages = [{"role": "user", "content": prompt}]
        messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(messages)

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return outputs


def evaluate_model_answer(model_outputs, dataset):
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


def evaluate(args):
    print("Loading test dataset...")
    dataset = load_test_dataset(args.test)
    print("Generating model answers...")
    model_outputs = generate_model_answer(args.model, dataset, args.formatting_prompt)
    print("Evaluating model answers...")
    eval_output, model_outputs_for_saving = evaluate_model_answer(model_outputs, dataset)
    print(f"Evaluation result: {eval_output}")

    eval_result_dir = os.path.join(args.model, "eval", args.test)
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

    datasets = args.test.split(",")
    for test_data in datasets:
        args.test = test_data
        evaluate(args)
