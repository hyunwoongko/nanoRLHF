"""
python3 -m nanorlhf.nanoverl.utils.merge_model \
    --model ./checkpoints/math/sft/step_4218 \
    --config ./configs/train_sft.yaml
"""

import os.path
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanorlhf import nanoray
from nanorlhf.nanoray.api.initialization import NANORAY_BASE_PORT
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel
from nanorlhf.nanoverl.configs.rl_config import RLConfig
from nanorlhf.nanoverl.configs.sft_config import SFTConfig


@nanoray.remote
class ModelMerger:
    """
    Model merger that loads a parallelized model and saves the merged version.

    Args:
        model_config: Configuration object containing model settings.
        rank (int): The rank of the current process in a distributed setup.
        model_parallel_world_size (int): The total number of model parallel ranks.
    """
    def __init__(self, model_config, rank, model_parallel_world_size):
        if model_config.zero_stage == 3:
            data_parallel_size = model_parallel_world_size
            tensor_parallel_size = pipeline_parallel_size = 1
        else:
            data_parallel_size = 1
            tensor_parallel_size = model_config.tensor_parallel_size
            pipeline_parallel_size = model_config.pipeline_parallel_size

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        self.mpu = MPU(
            rank=rank,
            local_rank=rank,
            world_size=model_parallel_world_size,
            local_world_size=model_parallel_world_size,
            host=model_config.host,
            port=model_config.port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            rollout_data_parallel_size=0,
            rollout_tensor_parallel_size=0,
            backend=model_config.backend,
            seed=model_config.seed,
        )
        if model_config.zero_stage == 3:
            self.model = DataParallel(self.model, mpu=self.mpu, zero_stage=3)
        else:
            self.model = TensorParallel(self.model, mpu=self.mpu)
            self.model = PipelineParallel(self.model, mpu=self.mpu)
        self.model.parallelize()

    def save_pretrained(self, save_dir):
        """
        Save the merged model and tokenizer to the specified directory.

        Args:
            save_dir (str): The directory where the merged model and tokenizer will be saved.
        """
        merged_save_dir = os.path.join(save_dir, "merged")
        self.model.from_parallelized(save_dir)
        self.model.save_parallelized(merged_save_dir, merge_checkpoints=True)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        tokenizer.save_pretrained(merged_save_dir)


def merge_model(args):
    """
    Merge a parallelized model into a single model and save it.

    Args:
        args: Command-line arguments containing model path, config path, and training type.
    """
    if args.training_type == "sft":
        config = SFTConfig.from_yaml(args.config)
        model_config = config.model
    elif args.training_type == "rl":
        config = RLConfig.from_yaml(args.config)
        model_config = config.actor
    else:
        raise ValueError(f"Unsupported training type: {args.training_type}")

    if model_config.zero_stage == 3:
        model_parallel_world_size = model_config.data_parallel_size
    else:
        model_parallel_world_size = model_config.tensor_parallel_size * model_config.pipeline_parallel_size

    nodes = {}
    for global_rank in range(model_parallel_world_size):
        nodes[f"node-{global_rank + 1}"] = nanoray.NodeConfig(
            rpc=True,
            host=model_config.host,
            port=NANORAY_BASE_PORT + global_rank,
        )

    print("Initialize nanoray session...")
    session = nanoray.init(nodes, default_node_id="node-1")
    node_ids = list(session.workers.keys())
    if len(node_ids) < model_parallel_world_size:
        raise RuntimeError(
            "`nanoray` was initialized with fewer nodes than `model_parallel_world_size`; "
            "please provide at least one NodeConfig per global rank."
        )

    print("Initialize ModelMerger actors...")
    object_refs = []
    for global_rank in range(model_parallel_world_size):
        node_id = node_ids[global_rank % len(node_ids)]
        object_ref = ModelMerger.options(pinned_node_id=node_id).remote(
            model_config,
            rank=global_rank,
            model_parallel_world_size=model_parallel_world_size,
            blocking=False,
        )
        object_refs.append(object_ref)
    model_mergers = nanoray.get(object_refs)

    print("Saving merged model...")
    object_refs = []
    for model_merger in model_mergers:
        object_ref = model_merger.save_pretrained.remote(args.model, blocking=False)
        object_refs.append(object_ref)
    nanoray.get(object_refs)

    print("Merged model saved! 😊")
    print(f"Merged model path: {os.path.join(args.model, 'merged')}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config yaml file.")
    parser.add_argument("--training_type", type=str, required=True, help="Type of training: sft or rl.")
    args = parser.parse_args()
    merge_model(args)
