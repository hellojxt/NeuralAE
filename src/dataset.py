import torch.utils.data as data
import torch
import os
import sox


class AudioDataset(data.Dataset):
    def __init__(self, data_dir, sample_rate=24000):
        self.data_dir = data_dir
        self.load_data()
        self.tfm = sox.Transformer()
        self.tfm.reverb()
        self.sample_rate = sample_rate

    def load_data(self):
        self.data = torch.load(os.path.join(self.data_dir, "patch_data.pt"))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.tfm.build_array(input_array=x.numpy(), sample_rate_in=self.sample_rate)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.data)
