
## Let's dive into RLHF training
#### 1) Prepare Supervised Fine-tuning Dataset

In the examples included in this project, supervised fine-tuning is performed using [NuminaMath-CoT-Small-Hard-200k](https://huggingface.co/datasets/NotASI/NuminaMath-CoT-Small-Hard-200k). 
From the original dataset, 180k samples are used as training data, and 1k samples are used as validation data.
Running the following command will tokenize the dataset and save it as zero-copy `.nano` format (similar with `.arrow` format)

```bash
bash ./scripts/prepare_sft_data.sh 
```

#### 2) Supervised Fine-tuning

Supervised fine-tuning is performed using [Qwen3-0.6B-base](https://huggingface.co/Qwen/Qwen3-0.6B-base) model with 3D parallelism by default config. 
If you want to modify hyperparameters, please edit `configs/train_sft.yaml` file.
Running the following command will start supervised fine-tuning. 
Moreover, you can monitor the training process if you have wandb account.

![sft_log](assets/sft_log.png)

```bash
bash ./scripts/train_sft.sh 
```

#### 3) Merge Parallelized Checkpoints

After supervised fine-tuning is completed, the parallelized checkpoints are saved in the directory you specified (default is `./checkpoints`)
To use the model for inference or further training, you need to merge the parallelized checkpoints into a single model checkpoint.
The following script will merge the checkpoints and save them in `$YOUR_CHECKPOINT_PATH/merged` directory.

```bash
bash ./scripts/merge_sft_model.sh $STEP
```

#### 4) Evaluate Supervised Fine-tuned Model
After merging the supervised fine-tuned model, you can evaluate it using the following script.
The evaluation is performed using [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) dataset (500 samples from MATH dataset)
and [Math-Verify](https://github.com/huggingface/Math-Verify) is used to parse and verify the model's generated output.

| step     | MATH-500 score |
|----------|----------------|
| 500      | 40.8           |
| 1000     | 39.4           |
| 1500     | 41.2           |
| **2000** | **43.4**       |
| 2109     | 41.8           |

```
bash ./scripts/eval_sft_model.sh $STEP
```

#### 5) Prepare Reinforcement Learning Dataset
Reinforcement learning is performed using [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) dataset.
I removed samples that have one of 'yes', 'no', 'true' or 'false' as the answer, so about 84k samples are used for training.
And [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) dataset is used for validation.
Running the following command will tokenize the dataset and save it as zero-copy `.nano` format.

```bash
bash ./scripts/prepare_rl_data.sh 
```

#### 6) Reinforcement Learning
Reinforcement learning is performed using PPO algorithm with the SFT model at 2000 steps as the initial policy.
To improve training efficiency, [One-step off-policy asynchronous RL](https://github.com/volcengine/verl/tree/main/recipe/one_step_off_policy) is applied.
If you want to modify hyperparameters, please edit `configs/train_rl.yaml` file.
Running the following command will start supervised fine-tuning. 
Moreover, you can monitor the training process if you have wandb account.

![one_step](assets/one_step.png)

![rl_log](assets/rl_log.png)

```bash
bash ./scripts/train_rl.sh 
```

#### 7) Merge Parallelized Checkpoints

After reinforcement learning is completed, the parallelized checkpoints are saved in the directory you specified (default is `./checkpoints`)
To use the model for inference or further training, you need to merge the parallelized checkpoints into a single model checkpoint.
The following script will merge the checkpoints and save them in `$YOUR_CHECKPOINT_PATH/merged` directory.

```bash
bash ./scripts/merge_rl_model.sh $STEP
```

#### 8) Evaluate Reinforcement Learning Model
After merging the reinforcement learning model, you can evaluate it using the following script.
The evaluation is performed same as the supervised fine-tuned model using [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) dataset.
Qwen3-0.6B (non-thinking) model is also evaluated as a reference.

| step                      | MATH-500 score |
|---------------------------|----------------|
| 50                        | 40.8           |
| 100                       | 43.8           |
| **150**                   | **46.6**       |
| 200                       | 43.8           |
| Qwen3-0.6B (non-thinking) | **49.8**       |

```
bash ./scripts/eval_rl_model.sh $STEP
```

```
bash ./scripts/eval_ref_model.sh
```
