import torch
from tqdm import tqdm
import numpy as np
from glob import glob
import torchaudio

def process_musdb(root_dir="dataset", subsets="train"):
    x_dir = f"{root_dir}/musdb18_split/{subsets}"
    y_dir = f"{root_dir}/{filter_name}/{subsets}"
    output_dir = y_dir
    file_list = glob(x_dir + "/*.wav")
    
    for x_path in tqdm(file_list):
        y_path = x_path.replace(x_dir, y_dir)
        x = torchaudio.load(x_path)[0]
        y = torchaudio.load(y_path)[0]
        # print(x.shape, y.shape)
        # print(x.min(), x.max(), y.min(), y.max())
        file_idx = x_path.split("/")[-1].split(".")[0]
        torch.save({"x": x, "y": y}, f"{output_dir}/{file_idx}.pt")


import sys
if __name__ == "__main__":
    filter_name = sys.argv[1]
    process_musdb(
        subsets="train",
    )
    process_musdb(
        subsets="test",
    )
    
