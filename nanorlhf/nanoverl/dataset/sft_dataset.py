from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, files: str):
        files = files.split(",")

