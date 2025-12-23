model_path="./checkpoints/math/rl"
formatting_prompt="./data/math_prompt.json"
global_step=$1

python3 -m nanorlhf.eval.math_eval \
  --model "$model_path/step_$global_step/merged" \
  --test MATH-500 \
  --formatting_prompt="$formatting_prompt"