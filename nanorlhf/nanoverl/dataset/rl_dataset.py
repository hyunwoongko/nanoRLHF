import functools
from typing import Optional

from torch.utils.data import Dataset

from nanorlhf.nanosets import load_dataset


def filter_by_length(sample, max_prompt_length):
    """
    Filters samples based on the maximum prompt length.

    Args:
        sample (dict): A sample from the dataset containing 'input_ids'.
        max_prompt_length (int): The maximum allowed length for 'input_ids'.

    Returns:
        bool: True if the sample's 'input_ids' length is less than or equal to max_prompt_length, False otherwise.
    """
    if sample is None:
        return False
    input_ids = sample["input_ids"]
    return len(input_ids) <= max_prompt_length


class RLDataset(Dataset):
    def __init__(self, file_path: str, max_prompt_length: Optional[int] = None):
        self.dataset = load_dataset(file_path)

        if max_prompt_length is not None:
            filter_fn = functools.partial(filter_by_length, max_prompt_length=max_prompt_length)
            self.dataset = self.dataset.filter(filter_fn)

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
