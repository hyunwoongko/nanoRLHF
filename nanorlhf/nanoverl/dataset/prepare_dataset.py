import itertools
import json
import multiprocessing
import os
import random
from argparse import ArgumentParser
from functools import partial

from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from nanorlhf.nanosets import Dataset


def prepare_dataset(
    files: str,
    output_path: str,
    tokenizer_name_or_path: str,
    max_length: int = 8192,
    training_type: str = "sft",
    batch_size: int = 256,
    seed: int = 1234,
    num_workers: int = 32,
    mp_chunksize: int = 512,
    messages_key: str = "messages",
    tools_key: str = "tools",
    prompt_key: str = "question",
    answer_key: str = "final_answer",
    reward_type: str = "math_rlvr",
):
    assert training_type in ["sft", "rl"], f"Unsupported training type: {training_type}"

    files = files.split(",")
    for file in files:
        # 1) check the data file is exist
        if not os.path.exists(file):
            raise FileNotFoundError(f"Data file {file} not found.")

        # 2) check the data extension (json and jsonl are supported)
        ext = os.path.splitext(file)[1]
        if ext not in [".json", ".jsonl"]:
            raise ValueError(f"Unsupported data file extension: {ext}. " f"Only .json and .jsonl are supported.")

    # 4) load dataset
    raw_dataset = []
    for file in tqdm(files, desc=f"Loading {len(files)} files"):
        ext = os.path.splitext(file)[1]
        if ext == "json":
            raw_data = json.load(open(file, "r"))
            raw_dataset.extend(raw_data)
        else:
            with open(file, "r") as f:
                for line in f.readlines():
                    line = json.loads(line)
                    raw_dataset.append(line)

    # 5) load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # 6) preprocess dataset
    set_seed(seed)
    random.shuffle(raw_dataset)

    with multiprocessing.Pool(num_workers) as pool:
        # 7) tokenize dataset
        if training_type == "sft":
            fn = partial(
                preprocess_sft,
                tokenizer=tokenizer,
                messages_key=messages_key,
                tools_key=tools_key,
                generation_prompt=extract_generation_prompt(tokenizer),
                max_length=max_length,
            )
            output_dataset = []
            for input_ids, loss_mask in tqdm(
                pool.imap_unordered(fn, raw_dataset, chunksize=mp_chunksize),
                total=len(raw_dataset),
                desc="Tokenizing",
            ):
                if input_ids is not None and loss_mask is not None:
                    rows = {"input_ids": input_ids, "loss_mask": loss_mask}
                    output_dataset.append(rows)

        else:
            fn = partial(
                preprocess_rl,
                tokenizer=tokenizer,
                prompt_key=prompt_key,
                answer_key=answer_key,
                max_length=max_length,
            )
            output_dataset = []
            for input_ids, answer in tqdm(
                pool.imap_unordered(fn, raw_dataset, chunksize=mp_chunksize),
                total=len(raw_dataset),
                desc="Tokenizing",
            ):
                if input_ids is not None and answer is not None:
                    rows = {
                        "input_ids": input_ids,
                        "reward_model": {"ground_truth": answer, "reward_type": reward_type},
                    }
                    output_dataset.append(rows)

        # 8) save dataset as nano format
        print("Converting tokenized dataset to zero-copy nano format...")
        nano_dataset = Dataset.from_list(output_dataset, batch_size=batch_size)
        nano_dataset.save_to_disk(output_path)
        print(f"Processed dataset saved to {output_path}")


def extract_generation_prompt(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )
    return token2[len(token1) :]


def preprocess_sft(sample, tokenizer, messages_key, tools_key, generation_prompt, max_length):
    messages = sample.get(messages_key, [])
    tools = sample.get(tools_key, None)
    if not messages:
        return None, None

    num_history_tokens = 0
    input_ids, loss_mask = [], []
    for turn_idx in range(len(messages)):
        role = messages[turn_idx].get("role")
        tokens = tokenizer.apply_chat_template(
            messages[: turn_idx + 1],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
        current_turn_tokens = tokens[num_history_tokens:]
        num_history_tokens = len(tokens)
        input_ids.append(current_turn_tokens)

        if role in ["system", "user"]:
            mask = [0] * len(current_turn_tokens)
            loss_mask.append(mask)
        elif role == "assistant":
            mask = [1] * len(current_turn_tokens)
            mask[: len(generation_prompt)] = [0] * len(generation_prompt)
            loss_mask.append(mask)
        else:
            raise ValueError(f"Unsupported role: {role}")

    input_ids = list(itertools.chain(*input_ids))[:max_length]
    loss_mask = list(itertools.chain(*loss_mask))[:max_length]
    return input_ids, loss_mask


def preprocess_rl(sample, tokenizer, prompt_key, answer_key, max_length):
    prompt = sample.get(prompt_key, "")
    answer = sample.get(answer_key, "")
    if not prompt or not answer:
        return None, None

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        assert isinstance(prompt, list)
        assert isinstance(prompt[0], dict)
        assert "role" in prompt[0] and "content" in prompt[0]
        assert prompt[0]["role"] == "user" and prompt[-1]["role"] == "user"
        messages = prompt

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    token_length = len(input_ids)
    if token_length >= max_length:
        return None, None

    return input_ids, answer


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    parser = ArgumentParser()
    parser.add_argument('--files', type=str, required=True, help='Comma-separated input data files (json or jsonl).')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the processed dataset.')
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True, help='Path to the tokenizer.')
    parser.add_argument('--max_length', type=int, default=8192, help='Maximum sequence length.')
    parser.add_argument('--training_type', type=str, default='sft', choices=['sft', 'rl'], help='Type of training.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid'], help='Data split.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for saving the dataset.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for shuffling the data.')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of processes for data preprocessing.')
    parser.add_argument('--mp_chunksize', type=int, default=512, help='Chunk size for multiprocessing.')
    # SFT only
    parser.add_argument('--messages_key', type=str, default='messages', help='Key for messages in the sft data.')
    parser.add_argument('--tools_key', type=str, default='tools', help='Key for tools in the sft data.')
    # RL only
    parser.add_argument('--prompt_key', type=str, default='question', help='Key for prompt in the rl data.')
    parser.add_argument('--answer_key', type=str, default='final_answer', help='Key for answer in the rl data.')
    parser.add_argument('--reward_type', type=str, default='math_rlvr', help='Type of reward for rl data.')
    args = parser.parse_args()

    prepare_dataset(
        files=args.files,
        output_path=args.output_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        max_length=args.max_length,
        training_type=args.training_type,
        messages_key=args.messages_key,
        tools_key=args.tools_key,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        mp_chunksize=args.mp_chunksize,
    )
