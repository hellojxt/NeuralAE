import musdb
import torch
import resampy
from tqdm import tqdm
import sox
import numpy as np
import os


def process_musdb(root_dir="dataset/musdb18", subsets="train", sr=24000, length=5):
    mus = musdb.DB(root=root_dir, subsets=subsets)
    tfm = sox.Transformer()
    tfm.reverb()
    file_idx = 0
    output_dir = f"{root_dir}/{args.tag}/{subsets}_patch"
    if not os.path.exists(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(output_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for track in tqdm(mus.tracks):
        vocals = track.targets["vocals"].audio
        vocal = vocals.mean(axis=1)
        ori_sample_rate = track.targets["vocals"].rate
        vocal = resampy.resample(vocal, ori_sample_rate, sr)
        # clip 65536 samples
        for i in range(0, len(vocal), sr * length):
            patch = torch.from_numpy(vocal[i : i + sr * length])
            if len(patch) < sr * length:
                continue
            if (patch**2).mean() < 1e-3:
                continue
            patch /= patch.abs().max()
            patch *= 10 ** (-12 / 20.0)
            x = patch
            if torch.isnan(x).any():
                print("nan")
                print(x)
            y = np.copy(tfm.build_array(input_array=x.numpy(), sample_rate_in=sr))
            y = torch.from_numpy(y)
            torch.save({"x": x, "y": y}, f"{output_dir}/{file_idx}.pt")
            file_idx += 1


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--root_dir", type=str, default="dataset/musdb18")
    parser.add_argument("--tag", type=str, default="reverb")
    args = parser.parse_args()
    process_musdb(
        root_dir=args.root_dir,
        subsets="train",
        sr=args.sr,
        length=args.length,
    )
    process_musdb(
        root_dir=args.root_dir,
        subsets="test",
        sr=args.sr,
        length=args.length,
    )
