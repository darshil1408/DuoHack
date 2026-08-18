import torch
import torch.nn.functional as F


def charbonnier(pred, target, eps=1e-3):
    """
    Robust L1-like reconstruction loss.
    """

    diff = pred - target

    return torch.sqrt(
        diff * diff + eps * eps
    ).mean()


def ssim_loss(pred, target, window=11):
    """
    Differentiable SSIM approximation.
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
    )

    sigma_y = (
        F.avg_pool2d(
            target * target,
            window,
            stride=1,
            padding=pad
        )
        - mu_y * mu_y
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            window,
            stride=1,
            padding=pad
        )
        - mu_x * mu_y
    )

    sigma_x = sigma_x.clamp_min(0.0)
    sigma_y = sigma_y.clamp_min(0.0)

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    ssim = (
        (
            (2 * mu_x * mu_y + c1)
            *
            (2 * sigma_xy + c2)
        )
        /
        (
            (mu_x * mu_x + mu_y * mu_y + c1)
            *
            (sigma_x + sigma_y + c2)
        )
    )

    return 1.0 - ssim.mean()


def gradient_loss(pred, target):
    """
    Gradient consistency loss.
    """

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    return (
        F.l1_loss(dx_pred, dx_target)
        +
        F.l1_loss(dy_pred, dy_target)
    )


def laplacian(x):
    kernel = x.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ]
    ).view(1, 1, 3, 3)

    return F.conv2d(
        x,
        kernel,
        padding=1
    )


def edge_loss(pred, target):
    """
    Edge preservation loss.
    """

    return F.l1_loss(
        laplacian(pred),
        laplacian(target)
    )


def total_loss(pred, target):
    """
    Combined training objective.
    """

    l_char = charbonnier(pred, target)
    l_ssim = ssim_loss(pred, target)
    l_grad = gradient_loss(pred, target)
    l_edge = edge_loss(pred, target)

    loss = (
        1.00 * l_char
        +
        0.20 * l_ssim
        +
        0.10 * l_grad
        +
        0.05 * l_edge
    )

    return loss, {
        "char": l_char.detach().item(),
        "ssim": l_ssim.detach().item(),
        "grad": l_grad.detach().item(),
        "edge": l_edge.detach().item(),
    }
