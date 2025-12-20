model_path="./checkpoints/math/rl"
global_step=279

python3 -m nanorlhf.eval.math_eval \
  --model "$model_path/step_$global_step/merged" \
  --test MATH-500