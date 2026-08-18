# AI-Powered Semiconductor Image Restoration

**Team:** Diva & Darshil

## Overview

In semiconductor manufacturing, microscopic inspection images are used to detect and measure extremely small defects on semiconductor wafers and chips.

The quality of these images is critical. A small amount of noise or loss of fine spatial detail can hide important manufacturing defects.

This project develops an **AI-powered image restoration system** that takes a degraded semiconductor inspection image and reconstructs a cleaner, sharper image at its original resolution.

### Problem

The input images contain two major types of degradation:

1. **Speckle Noise**

   * Introduces random variations in pixel intensity.
   * Makes microscopic structures appear grainy or distorted.
   * Can produce pixel values outside the expected clean-image range.

2. **Spatial Resolution Reduction**

   * The original high-resolution image is downsampled.
   * Fine structures and defect information are lost.
   * For example:

```text
Ground Truth
256 × 256
     ↓
Downsampling
     ↓
Noisy Low Resolution
128 × 128
```

The objective is:

```text
Noisy + Low Resolution Image
              ↓
        AI Restoration
              ↓
Clean + High Resolution Image
```

---

## Project Objective

Build a deep-learning pipeline capable of:

* Removing noise from semiconductor inspection images.
* Recovering lost spatial details.
* Upscaling low-resolution images.
* Producing high-quality restored images.
* Quantitatively evaluating restoration quality.
* Creating a reproducible inference pipeline that can be used on new images.

The final system should provide both **visual restoration** and **measurable improvement using image-quality metrics**.

---

## Dataset

The project uses paired degraded and ground-truth images.

### Dataset structure

The current dataset contains approximately **3,200 paired samples**.

### Ground Truth

Ground-truth images are approximately:

```text
Shape: 256 × 256
Datatype: float32
Typical range: 0.0 – 1.0
```

### Noisy Low Resolution

The degraded images are approximately:

```text
Shape: 128 × 128
Datatype: float32
```

The corresponding files share the same identifier.

For example:

```text
NoisyLR/000123.npy
        ↕
GT/000123.npy
```

This pairing allows the model to learn:

```text
NoisyLR → Ground Truth
```

---

## Baseline

Before training a neural network, the project establishes a traditional image-processing baseline.

The degraded `128 × 128` image is first resized to `256 × 256` using **bicubic interpolation**.

```text
NoisyLR
128 × 128
   ↓
Bicubic Upsampling
   ↓
256 × 256
   ↓
Compare with Ground Truth
```

This provides an important reference point.

A trained neural network should ideally outperform this baseline.

---

## Deep Learning Approach

The restoration model learns a mapping:

[
f_\theta(X) \rightarrow \hat{Y}
]

where:

* (X) = degraded input image
* (Y) = ground-truth image
* (\hat{Y}) = model's restored image
* (\theta) = learnable neural-network parameters

The model is trained to minimize the difference between the predicted restoration and the ground truth.

Conceptually:

```text
                 Neural Network
                      │
                      ▼
NoisyLR ──────────► Model ──────────► Restored Image
  │                                      │
  │                                      │
  └──────────── Ground Truth ────────────┘
                     │
                     ▼
                  Loss
                     │
                     ▼
              Backpropagation
                     │
                     ▼
              Update Weights
```

---

## Model

The project uses **PyTorch** for deep-learning development.

The architecture is designed around image-to-image restoration.

Important model requirements include:

* Preserve spatial information.
* Remove noise.
* Recover high-frequency details.
* Increase spatial resolution.
* Avoid introducing artificial artifacts.
* Maintain stable training.

The model architecture can be improved and compared experimentally as development progresses.

---

## Training Pipeline

The complete training pipeline follows:

```text
Load Dataset
     ↓
Read .npy Files
     ↓
Normalize / Validate Data
     ↓
Create PyTorch Dataset
     ↓
Create DataLoader
     ↓
Feed NoisyLR Image
     ↓
Neural Network
     ↓
Generate Restored Image
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Update Model Parameters
     ↓
Validation
     ↓
Calculate PSNR / SSIM
     ↓
Save Best Model
```

---

## Evaluation

Image restoration cannot be evaluated only by looking at the output.

The project therefore uses quantitative image-quality metrics.

### PSNR

**Peak Signal-to-Noise Ratio (PSNR)** measures the reconstruction quality between the predicted image and the ground truth.

Higher PSNR generally indicates better reconstruction quality.

The relationship is:

[
PSNR = 10\log_{10}\left(\frac{MAX_I^2}{MSE}\right)
]

where:

* (MAX_I) = maximum possible pixel value
* (MSE) = Mean Squared Error

For normalized images:

```text
MAX_I = 1
```

Therefore:

[
PSNR = 10\log_{10}\left(\frac{1}{MSE}\right)
]

### SSIM

**Structural Similarity Index Measure (SSIM)** evaluates structural similarity between images.

Unlike MSE alone, SSIM considers properties such as:

* luminance
* contrast
* structure

This makes it particularly useful for image restoration.

---

## Evaluation Strategy

The project compares multiple approaches:

```text
             ┌──────────────────┐
             │  Noisy Low-Res   │
             └────────┬─────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Bicubic Baseline         AI Restoration
          │                       │
          ▼                       ▼
      PSNR / SSIM            PSNR / SSIM
          │                       │
          └───────────┬───────────┘
                      ▼
                Compare Results
```

This allows us to determine whether the AI model actually provides an improvement over conventional interpolation.

---

## Technology Stack

### Programming

* Python

### Deep Learning

* PyTorch
* Torchvision

### Numerical Computing

* NumPy

### Image Processing

* OpenCV
* scikit-image

### Data Analysis

* Pandas
* Matplotlib

### Development

* VS Code
* Jupyter / Google Colab when GPU acceleration is required

### Version Control

* Git
* GitHub

## Reproducibility

The project aims to make experiments reproducible.

Each experiment should record:

```text
Model architecture
Dataset split
Batch size
Learning rate
Number of epochs
Loss function
Optimizer
Random seed
Training hardware
PSNR
SSIM
Training time
```

This makes it possible to compare different experiments fairly.

---

## Current Development

The project is being developed incrementally.

### Phase 1 — Dataset Understanding

* [x] Inspect dataset structure
* [x] Identify GT images
* [x] Identify NoisyLR images
* [x] Verify image dimensions
* [x] Verify NumPy datatypes
* [x] Inspect pixel ranges
* [x] Establish paired-image relationship

### Phase 2 — Baseline

* [x] Implement bicubic upsampling
* [x] Calculate baseline PSNR
* [ ] Calculate baseline SSIM
* [ ] Generate qualitative comparison images

### Phase 3 — Deep Learning

* [x] Create PyTorch dataset pipeline
* [ ] Implement restoration model
* [ ] Implement training loop
* [ ] Implement validation loop
* [ ] Implement checkpoint saving
* [ ] Train initial model

### Phase 4 — Evaluation

* [ ] Compare AI model against bicubic baseline
* [ ] Calculate PSNR
* [ ] Calculate SSIM
* [ ] Analyze failure cases
* [ ] Generate visual comparisons

### Phase 5 — Optimization

* [ ] Improve architecture
* [ ] Experiment with loss functions
* [ ] Tune hyperparameters
* [ ] Improve inference speed
* [ ] Reduce memory usage
* [ ] Test robustness

### Phase 6 — Deployment

* [ ] Build inference pipeline
* [ ] Add command-line interface
* [ ] Document model usage
* [ ] Package reproducible environment
* [ ] Explore potential real-world applications

---

## Team

### Diva

**Focus areas**

* Evaluation pipeline
* Experiment analysis
* Performance benchmarking
* Documentation and presentation
* Deployment and packaging

### Darshil

**Focus areas**

* Dataset pipeline
* PyTorch implementation
* Model development
* Training pipeline
* Optimization

Both team members contribute to experimentation, debugging, documentation, and final system development.

---

## Why This Project Matters

Semiconductor manufacturing depends heavily on accurate inspection.

If an inspection image contains noise or insufficient resolution, subtle defects may become difficult to identify.

An effective restoration system can potentially help by improving the quality of images before downstream analysis.

The broader pipeline can eventually become:

```text
Semiconductor Inspection
          ↓
Raw / Degraded Image
          ↓
AI Image Restoration
          ↓
High-Quality Image
          ↓
Defect Detection
          ↓
Defect Classification
          ↓
Manufacturing Decision
```

Image restoration is therefore potentially one component of a larger **AI-assisted semiconductor inspection system**.

---

## Future Scope

Potential future extensions include:

* Automated semiconductor defect detection.
* Defect classification.
* Wafer inspection analytics.
* Anomaly detection.
* Restoration + segmentation pipeline.
* Confidence estimation.
* Real-time inference.
* Model compression.
* Edge deployment.
* Manufacturing-process analytics.
* Integration with semiconductor inspection workflows.

---

## Results

Results will be updated as experiments are completed.

| Method           | Resolution | PSNR | SSIM |
| ---------------- | ---------: | ---: | ---: |
| Bicubic Baseline |  256 × 256 |  TBD |  TBD |
| AI Restoration   |  256 × 256 |  ~29 | ~0.82-0.88 |

Visual comparisons will also be added:

```text
Ground Truth  |  NoisyLR  |  Bicubic  |  AI Restoration
```
Getting Started

Clone the repository:

git clone <repository-url>
cd semiconductor-image-restoration

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the training pipeline:

python scripts/train.py

Run inference:

python scripts/infer.py

---

## Acknowledgements

This project was developed as an applied research and engineering project focused on **AI-based restoration of semiconductor inspection imagery**.

---

## Authors

**Diva & Darshil**

AI/ML • Computer Vision • Deep Learning • Semiconductor Inspection
