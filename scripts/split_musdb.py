import musdb
import resampy
from tqdm import tqdm
import numpy as np
import os
from scipy.io import wavfile

def process_musdb(root_dir="dataset/musdb18", subsets="train", sr=24000, length=5):
    mus = musdb.DB(root=root_dir, subsets=subsets)
    file_idx = 0
    output_dir = f"{root_dir}_split/{subsets}"
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
        for i in range(0, len(vocal) - sr * length, sr * length):
            patch = vocal[i : i + sr * length]
            if len(patch) < sr * length:
                continue
            if np.mean(patch**2) < 1e-3:
                continue
            patch /= np.max(np.abs(patch))
            patch *= 10 ** (-12 / 20.0)
            x = patch
            # save wav
            wavfile.write(
                f"{output_dir}/{file_idx}.wav",
                sr,
                x,
            )
            file_idx += 1
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--root_dir", type=str, default="dataset/musdb18")
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
