import math

import numpy as np
import torch
import torch.nn.functional as F


def psnr_np(pred, target, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    pred : array-like
        Predicted/restored image.
    target : array-like
        Ground-truth image.
    data_range : float
        Dynamic range of the image values.

    Returns
    -------
    float
        PSNR value in decibels (dB).
    """

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mse = np.mean((pred - target) ** 2)

    if mse <= 1e-15:
        return float("inf")

    return 10.0 * math.log10(
        (data_range ** 2) / mse
    )


def ssim_torch(pred, target, window=11):
    """
    Calculate SSIM using PyTorch average-pooling statistics.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted image tensor.
    target : torch.Tensor
        Ground-truth image tensor.
    window : int
        Size of the local SSIM window.

    Returns
    -------
    float
        Mean SSIM value.
    """

    pad = window // 2

    mu_x = F.avg_pool2d(
        pred,
        window,
        stride=1,
        padding=pad
    )

    mu_y = F.avg_pool2d(
        target,
        window,
        stride=1,
        padding=pad
    )

    sigma_x = (
        F.avg_pool2d(
            pred * pred,
            window,
            stride=1,
            padding=pad
        )
        - mu_x * mu_x
    ).clamp_min(0.0)

    sigma_y = (
        F.avg_pool2d(
            target * target,
            window,
            stride=1,
            padding=pad
        )
        - mu_y * mu_y
    ).clamp_min(0.0)

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            window,
            stride=1,
            padding=pad
        )
        - mu_x * mu_y
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    value = (
        (2 * mu_x * mu_y + c1)
        * (2 * sigma_xy + c2)
    ) / (
        (mu_x * mu_x + mu_y * mu_y + c1)
        * (sigma_x + sigma_y + c2)
    )

    return value.mean().item()


def lpips_score(pred, target):
    """
    Calculate LPIPS perceptual distance when the optional
    lpips package is installed.

    Returns None if LPIPS is unavailable.
    """

    try:
        import lpips
    except ImportError:
        return None

    # LPIPS expects 3-channel tensors in [-1, 1].
    pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
    target3 = target.repeat(1, 3, 1, 1) * 2 - 1

    loss_fn = lpips.LPIPS(net="alex").to(pred.device)

    with torch.no_grad():
        return float(
            loss_fn(pred3, target3).mean().item()
        )