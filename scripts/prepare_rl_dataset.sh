#!/bin/bash

# Arguments for dataset preprocessing
tokenizer_name_or_path="Qwen/Qwen3-0.6B"
formatting_prompt="./data/math_prompt.json"
max_length=8192
training_type="rl"
prompt_key="problem"
answer_key="answer"
enable_thinking_key="enable_thinking"
allow_thinking=false
batch_size=256
seed=1234
num_workers=32
mp_chunksize=512

#echo "Start to preprocess RL training dataset..."
python3 -m nanorlhf.nanoverl.dataset.prepare_dataset \
    --files="./data/DeepMath-84k/train.jsonl" \
    --output_path="./data/DeepMath-84k/preprocessed/train.nano" \
    --tokenizer_name_or_path="$tokenizer_name_or_path" \
    --formatting_prompt="$formatting_prompt" \
    --max_length=$max_length \
    --training_type="$training_type" \
    --prompt_key="$prompt_key" \
    --answer_key="$answer_key" \
    --enable_thinking_key="$enable_thinking_key" \
    --allow_thinking=$allow_thinking \
    --batch_size=$batch_size \
    --seed=$seed \
    --num_workers=$num_workers \
    --mp_chunksize=$mp_chunksize

echo "Start to preprocess RL validation dataset..."
python3 -m nanorlhf.nanoverl.dataset.prepare_dataset \
    --files="./data/MATH-500/test.jsonl" \
    --output_path="./data/MATH-500/preprocessed/valid.nano" \
    --tokenizer_name_or_path="$tokenizer_name_or_path" \
    --formatting_prompt="$formatting_prompt" \
    --max_length=$max_length \
    --training_type="$training_type" \
    --prompt_key="$prompt_key" \
    --answer_key="$answer_key" \
    --enable_thinking_key="$enable_thinking_key" \
    --allow_thinking=$allow_thinking \
    --batch_size=$batch_size \
    --seed=$seed \
    --num_workers=$num_workers \
    --mp_chunksize=$mp_chunksize