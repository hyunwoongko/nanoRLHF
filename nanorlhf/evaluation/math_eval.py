"""
python3 -m nanorlhf.evaluation.math_eval \
    --model <model_name_or_path> \
    --test_data MATH-500
"""
import json
import os
from argparse import ArgumentParser
from typing import Optional

from math_verify import parse, verify

from nanorlhf.nanosets import load_dataset
from nanorlhf.nanovllm import LLM, SamplingParams


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def get_unnormalized_answer(text: str) -> str:
    answer = last_boxed_only_string(text)
    if answer:
        return answer
    else:
        return "[invalidanswer]"


def load_test_dataset(test_data):
    data_path = f"./data/{test_data}/dev.jsonl"
    return load_dataset(data_path)


def generate_model_answer(model, dataset):
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    llm = LLM(model)

    prompts = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        messages = [{"role": "user", "content": sample["problem"]}]
        prompts.append(messages)

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return outputs


def evaluate_model_answer(model_outputs, dataset):
    accuracy = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        model_text = model_outputs[idx]["text"]

        if sample["answer"] is None:
            gold_answer = parse(last_boxed_only_string(sample["solution"]))
        else:
            if "boxed" not in sample["answer"]:
                gold_answer = parse("\\boxed{" + str(sample["answer"]) + "}")
            else:
                gold_answer = parse(sample["answer"])

        model_answer = parse(get_unnormalized_answer(model_text))
        accuracy += int(verify(gold_answer, model_answer))

    accuracy /= len(dataset)
    return {"accuracy": accuracy}


def evaluate(args):
    dataset = load_test_dataset(args.test_data)
    model_outputs = generate_model_answer(args.model, dataset)
    eval_output = evaluate_model_answer(model_outputs, dataset)

    eval_result_dir = f"./eval/{args.model}/{args.test_data}"
    os.makedirs(eval_result_dir, exist_ok=True)
    eval_result_path = os.path.join(eval_result_dir, "score.json")
    json.dump(eval_output, open(eval_result_path, "w"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--test_data", type=str, default="MATH-500", help="Test dataset to evaluate on.")
    evaluate(parser.parse_args())
