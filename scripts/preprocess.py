import torch
from tqdm import tqdm
import numpy as np
from glob import glob
import torchaudio
import os

def process_musdb():
    y_file_list = glob(f"dataset/{dataset}/*-after.wav")
    idx = 0
    train_num = 0
    test_num = 0
    for y_path in tqdm(y_file_list):
        x_path = y_path.replace("-after.wav", ".wav")
        x = torchaudio.load(x_path)[0]
        y = torchaudio.load(y_path)[0]
        # print(x.shape, y.shape)
        # print(x.min(), x.max(), y.min(), y.max())
        # print(x.min(), x.max(), y.min(), y.max())
        # file_idx = x_path.split("/")[-1].split(".")[0]
        if np.random.rand() < 0.8:
            subset = "train"
            train_num += 1
            file_idx = train_num
        else:
            subset = "test"
            test_num += 1
            file_idx = test_num
        output_dir = f"dataset/{out_dir}/{subset}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        torch.save({"x": x, "y": y}, f"{output_dir}/{file_idx}.pt")
        # if idx == 10:
        #     break
        # idx += 1

import sys
if __name__ == "__main__":
    dataset = sys.argv[1]
    out_dir = sys.argv[2]
    process_musdb()
