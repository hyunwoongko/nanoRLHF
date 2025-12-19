#!/bin/bash

data_path="./data/NuminaMath-CoT-Small-Hard-200k"

# I split the dataset into 6 parts to avoid github file size limit.
# Each file has 30k rows, total 180k rows.
train0="${data_path}/train-0000-of-0005.jsonl"
train1="${data_path}/train-0001-of-0005.jsonl"
train2="${data_path}/train-0002-of-0005.jsonl"
train3="${data_path}/train-0003-of-0005.jsonl"
train4="${data_path}/train-0004-of-0005.jsonl"
train5="${data_path}/train-0005-of-0005.jsonl"

# Valid dataset has 1k rows and it sampled from training dataset.
valid="${data_path}/valid.jsonl"

# Arguments for dataset preprocessing
tokenizer_name_or_path="Qwen/Qwen3-0.6B"
formatting_prompt="./data/math_prompt.json"
max_length=8192
training_type="sft"
messages_key="messages"
tools_key="tools"
enable_thinking_key="enable_thinking"
allow_thinking=false
batch_size=256
seed=1234
num_workers=32
mp_chunksize=512

#echo "Start to preprocess SFT training dataset..."
python3 -m nanorlhf.nanoverl.dataset.prepare_dataset \
    --files="$train0,$train1,$train2,$train3,$train4,$train5" \
    --output_path="${data_path}/preprocessed/train.nano" \
    --tokenizer_name_or_path="$tokenizer_name_or_path" \
    --formatting_prompt="$formatting_prompt" \
    --max_length=$max_length \
    --training_type="$training_type" \
    --messages_key="$messages_key" \
    --tools_key="$tools_key" \
    --enable_thinking_key="$enable_thinking_key" \
    --allow_thinking=$allow_thinking \
    --batch_size=$batch_size \
    --seed=$seed \
    --num_workers=$num_workers \
    --mp_chunksize=$mp_chunksize

echo "Start to preprocess SFT validation dataset..."
python3 -m nanorlhf.nanoverl.dataset.prepare_dataset \
    --files="$valid" \
    --output_path="${data_path}/preprocessed/valid.nano" \
    --tokenizer_name_or_path="$tokenizer_name_or_path" \
    --formatting_prompt="$formatting_prompt" \
    --max_length=$max_length \
    --training_type="$training_type" \
    --messages_key="$messages_key" \
    --tools_key="$tools_key" \
    --enable_thinking_key="$enable_thinking_key" \
    --allow_thinking=$allow_thinking \
    --batch_size=$batch_size \
    --seed=$seed \
    --num_workers=$num_workers \
    --mp_chunksize=$mp_chunksize