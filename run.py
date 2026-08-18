import argparse
from pathlib import Path

import numpy as np
import torch

from src.model import RestorationSRNet


def load_model(weights, device):
    """
    Load the trained RestorationSRNet from a checkpoint.
    """

    checkpoint = torch.load(
        weights,
        map_location=device,
        weights_only=False,
    )

    model = RestorationSRNet(
        channels=int(checkpoint.get("channels", 64)),
        blocks=int(checkpoint.get("blocks", 12)),
    ).to(device)

    state = (
        checkpoint["model"]
        if "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, checkpoint


def load_npy(path):
    """
    Load a single NoisyLR .npy file.

    Supported input:
        [H, W]
        [1, H, W]

    Returned tensor:
        [1, 1, H, W]
    """

    image = np.load(
        path,
        allow_pickle=False,
    ).astype(np.float32)

    if image.ndim == 2:
        image = image[None, None, ...]

    elif image.ndim == 3 and image.shape[0] == 1:
        image = image[None, ...]

    else:
        raise ValueError(
            f"Unsupported input shape {image.shape} "
            f"for {path}"
        )

    return torch.from_numpy(image)


@torch.inference_mode()
def restore_directory(
    input_dir,
    output_dir,
    weights,
    device,
    amp=False,
):
    """
    Restore every .npy image in input_dir and save
    the restored images to output_dir.
    """

    model, checkpoint = load_model(
        weights,
        device,
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        input_dir.glob("*.npy")
    )

    if not files:
        raise RuntimeError(
            f"No .npy files found in {input_dir}"
        )

    print("=" * 60)
    print("KLA IMAGE RESTORATION")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Input files: {len(files)}")
    print(f"Weights: {weights}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "val_psnr" in checkpoint:
        print(
            f"Checkpoint PSNR: "
            f"{checkpoint['val_psnr']}"
        )

    if "val_ssim" in checkpoint:
        print(
            f"Checkpoint SSIM: "
            f"{checkpoint['val_ssim']}"
        )

    print()

    for index, path in enumerate(files, start=1):

        image = load_npy(path).to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            enabled=(
                amp
                and device.type == "cuda"
            ),
        ):
            restored = model(image)

        restored = (
            restored
            .squeeze()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        output_path = output_dir / path.name

        np.save(
            output_path,
            restored,
        )

        if index == 1 or index % 25 == 0:
            print(
                f"Processed "
                f"{index}/{len(files)}"
            )

    print()
    print("Inference complete.")
    print(f"Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Restore KLA NoisyLR .npy images "
            "using a trained RestorationSRNet."
        )
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing NoisyLR .npy files.",
    )

    parser.add_argument(
        "output_dir",
        help="Directory for restored .npy files.",
    )

    parser.add_argument(
        "--weights",
        default="runs_30/best.pt",
        help="Path to the trained checkpoint.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision.",
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device == "cpu"
        or torch.cuda.is_available()
        else "cpu"
    )

    restore_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights=args.weights,
        device=device,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
