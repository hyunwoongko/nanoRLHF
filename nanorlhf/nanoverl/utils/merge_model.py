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
from nanorlhf.nanotron import MPU, TensorParallel, PipelineParallel, DataParallel
from nanorlhf.nanoverl.configs.sft_config import SFTConfig


@nanoray.actor
class ModelMerger:
    def __init__(self, config, rank, model_parallel_world_size):
        if config.model.zero_stage == 3:
            data_parallel_size = model_parallel_world_size
            tensor_parallel_size = pipeline_parallel_size = 1
        else:
            data_parallel_size = 1
            tensor_parallel_size = config.model.tensor_parallel_size
            pipeline_parallel_size = config.model.pipeline_parallel_size

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        self.mpu = MPU(
            rank=rank,
            local_rank=rank,
            world_size=model_parallel_world_size,
            local_world_size=model_parallel_world_size,
            host=config.model.host,
            port=config.model.port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            rollout_data_parallel_size=0,
            rollout_tensor_parallel_size=0,
            backend=config.model.backend,
            seed=config.model.seed,
        )
        if config.model.zero_stage == 3:
            self.model = DataParallel(self.model, mpu=self.mpu, zero_stage=3)
        else:
            self.model = TensorParallel(self.model, mpu=self.mpu)
            self.model = PipelineParallel(self.model, mpu=self.mpu)
        self.model.parallelize()

    def save_pretrained(self, save_dir):
        merged_save_dir = os.path.join(save_dir, "merged")
        self.model.from_parallelized(save_dir)
        self.model.save_parallelized(merged_save_dir, merge_checkpoints=True)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        tokenizer.save_pretrained(merged_save_dir)


def merge_model(args):
    config = SFTConfig.from_yaml(args.config)

    if config.model.zero_stage == 3:
        model_parallel_world_size = config.model.data_parallel_size
    else:
        model_parallel_world_size = config.model.tensor_parallel_size * config.model.pipeline_parallel_size

    nodes = {}
    base_port = 9200
    for global_rank in range(model_parallel_world_size):
        nodes[f"node-{global_rank + 1}"] = nanoray.NodeConfig(
            cpus=4.0,
            gpus=1.0,
            rpc=True,
            host=config.model.host,
            port=base_port + global_rank,
        )

    print("Initialize nanoray session...")
    session = nanoray.init(nodes, default_node_id="node-1")
    node_ids = list(session._workers.keys())
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
            config, rank=global_rank, model_parallel_world_size=model_parallel_world_size, blocking=False
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
    args = parser.parse_args()
    merge_model(args)
