import json
import os
import random

from nanorlhf.nanoverl.utils.sft_config import SFTConfig
from transformers import AutoTokenizer, set_seed


def prepare_dataset(config: str, files: str, training_type: str = "sft"):
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

    # 3) load config
    if training_type == "str":
        config = SFTConfig.from_yaml(config)
    else:
        raise NotImplemented

    # 4) load dataset
    raw_dataset = []
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext == "json":
            raw_data = json.load(open(file, "r"))
        else:
            raw_data = [json.loads(line) for line in open(file, "r").readlines()]
        raw_dataset.extend(raw_data)

    # 5) load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.partial_pretrain)

    # 6) preprocess dataset
    set_seed(config.training.seed)
    random.shuffle(raw_dataset)

    for sample in raw_dataset:
        messages = sample[config.data.messages_key]
        for