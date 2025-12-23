model_path="./checkpoints/math/sft"
global_step=$1

python3 -m nanorlhf.nanoverl.utils.merge_model \
  --model "$model_path/step_$global_step" \
  --config configs/train_sft.yaml \
  --training_type sft
