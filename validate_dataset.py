from pathlib import Path
import numpy as np

ROOT = Path(r"..\train")

if (ROOT / "GT").is_dir():
    BASE = ROOT
else:
    BASE = ROOT / "train"

GT = BASE / "GT"
LR = BASE / "NoisyLR"

gt = {p.stem: p for p in GT.glob("*.npy")}
lr = {p.stem: p for p in LR.glob("*.npy")}

common = sorted(gt.keys() & lr.keys())

print("Dataset:", BASE.resolve())
print("GT files:", len(gt))
print("NoisyLR files:", len(lr))
print("Paired:", len(common))
print("Missing GT:", len(lr.keys() - gt.keys()))
print("Missing NoisyLR:", len(gt.keys() - lr.keys()))

for name in common[:5]:
    a = np.load(lr[name], allow_pickle=False)
    b = np.load(gt[name], allow_pickle=False)

    print(
        name,
        "LR:", a.shape, a.dtype,
        "range:", float(a.min()), float(a.max()),
        "| GT:", b.shape, b.dtype,
        "range:", float(b.min()), float(b.max())
    )

assert len(common) > 0
assert all(np.load(gt[n], allow_pickle=False).shape == (256, 256) for n in common[:20])
assert all(np.load(lr[n], allow_pickle=False).shape == (128, 128) for n in common[:20])

print("Dataset validation passed.")
