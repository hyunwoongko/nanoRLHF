model_path="./checkpoints/math/sft"
global_step=2109

python3 -m nanorlhf.eval.math_eval \
  --model "$model_path/step_$global_step/merged" \
  --test MATH-500