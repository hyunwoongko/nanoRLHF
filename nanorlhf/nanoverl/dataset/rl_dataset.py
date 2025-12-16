import torch
from torch.utils.data import Dataset

from nanorlhf.nanosets import load_dataset


class RLDataset(Dataset):
    def __init__(self, file_path: str):
        self.dataset = load_dataset(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, int):
            select = self.dataset[key]
            input_ids = select["input_ids"]
            position_ids = list(range(len(input_ids)))
            cu_seq_lens = [0, len(input_ids)]
            reward_model = select["reward_model"]
        else:
            input_ids = []
            position_ids = []
            cu_seq_lens = [0]
            reward_model = []
            for sample in self.dataset[key]:
                input_ids.extend(sample["input_ids"])
                position_ids.extend(list(range(len(sample["input_ids"]))))
                cu_seq_lens.append(cu_seq_lens[-1] + len(sample["input_ids"]))
                reward_model.append(sample["reward_model"])

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "reward_model": reward_model,
        }
