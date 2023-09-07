import torch.utils.data as data
import torch
import os
from glob import glob
import torchaudio

class AudioDataset(data.Dataset):
    def __init__(self, data_dir, tag = "train"):
        self.data_dir = data_dir
        self.file_list = glob(f"{data_dir}/{tag}/*")

    def __getitem__(self, index):
        data = torch.load(self.file_list[index])
        x = data["x"]
        y = data["y"]
        x = x.reshape(1, -1).float()
        y = y.reshape(1, -1).float()
        return x, y

    def __len__(self):
        return len(self.file_list)
