formatting_prompt="./data/math_prompt.json"

python3 -m nanorlhf.eval.math_eval \
  --model "Qwen/Qwen3-0.6B" \
  --test MATH-500 \
  --formatting_prompt="$formatting_prompt"