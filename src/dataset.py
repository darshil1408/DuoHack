from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SemiconductorDataset(Dataset):
    """
    PyTorch Dataset for paired semiconductor image restoration data.

    Each sample contains:
        NoisyLR image: low-resolution noisy input
        GT image: high-resolution ground truth
    """

    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)

        if not self.noisy_dir.exists():
            raise FileNotFoundError(
                f"NoisyLR directory not found: {self.noisy_dir}"
            )

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"GT directory not found: {self.gt_dir}"
            )

        self.noisy_files = sorted(self.noisy_dir.glob("*.npy"))
        self.gt_files = sorted(self.gt_dir.glob("*.npy"))

        if len(self.noisy_files) == 0:
            raise ValueError(
                f"No .npy files found in: {self.noisy_dir}"
            )

        if len(self.gt_files) == 0:
            raise ValueError(
                f"No .npy files found in: {self.gt_dir}"
            )

        if len(self.noisy_files) != len(self.gt_files):
            raise ValueError(
                "Number of NoisyLR and GT files does not match. "
                f"NoisyLR: {len(self.noisy_files)}, "
                f"GT: {len(self.gt_files)}"
            )

        for noisy_file, gt_file in zip(self.noisy_files, self.gt_files):
            if noisy_file.stem != gt_file.stem:
                raise ValueError(
                    "NoisyLR and GT files are not correctly paired: "
                    f"{noisy_file.name} != {gt_file.name}"
                )

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):
        noisy_image = np.load(self.noisy_files[index])
        gt_image = np.load(self.gt_files[index])

        noisy_image = noisy_image.astype(np.float32)
        gt_image = gt_image.astype(np.float32)

        noisy_image = torch.from_numpy(noisy_image)
        gt_image = torch.from_numpy(gt_image)

        if noisy_image.ndim == 2:
            noisy_image = noisy_image.unsqueeze(0)

        if gt_image.ndim == 2:
            gt_image = gt_image.unsqueeze(0)

        return noisy_image, gt_image
