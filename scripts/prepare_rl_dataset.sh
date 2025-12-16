#!/bin/bash

data_path="./data/DeepMath-40k"

# Train dataset has 40k rows and it sampled from training dataset.
# I sampled only data with difficulty level 6 or higher.
train="${data_path}/train.jsonl"

# Valid dataset has 1k rows and it sampled from training dataset.
valid="${data_path}/valid.jsonl"

# Arguments for dataset preprocessing
tokenizer_name_or_path="Qwen/Qwen3-0.6B"
max_length=8192
training_type="rl"
prompt_key="question"
answer_key="final_answer"
batch_size=256
seed=1234
num_workers=32
mp_chunksize=512

#echo "Start to preprocess SFT training dataset..."
python3 -m nanorlhf.nanoverl.dataset.prepare_dataset \
    --files="$train" \
    --output_path="${data_path}/preprocessed/train.nano" \
    --tokenizer_name_or_path="$tokenizer_name_or_path" \
    --max_length=$max_length \
    --training_type="$training_type" \
    --prompt_key="$prompt_key" \
    --answer_key="$answer_key" \
    --batch_size=$batch_size \
    --seed=$seed \
    --num_workers=$num_workers \
    --mp_chunksize=$mp_chunksize

echo "Start to preprocess SFT validation dataset..."
python3 -m nanorlhf.nanoverl.dataset.prepare_dataset \
    --files="$valid" \
    --output_path="${data_path}/preprocessed/valid.nano" \
    --tokenizer_name_or_path="$tokenizer_name_or_path" \
    --max_length=$max_length \
    --training_type="$training_type" \
    --prompt_key="$prompt_key" \
    --answer_key="$answer_key" \
    --batch_size=$batch_size \
    --seed=$seed \
    --num_workers=$num_workers \
    --mp_chunksize=$mp_chunksize