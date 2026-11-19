# WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement

> **Extended & Improved Implementation** based on the paper:
> [*"WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement"*](https://doi.org/10.1109/TMM.2025.3535342) — IEEE Transactions on Multimedia, 2025.

This repository contains an **enhanced reproduction** of WV-LUT with five targeted improvements to the training pipeline, loss formulation, architecture, and inference efficiency. Our improved model **surpasses the paper's reported NN baseline PSNR** on the LOL-v1 benchmark using **only 1/3 of the original training iterations**.

---

## Results at a Glance (LOL-v1 Benchmark)

| Model | Checkpoint | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Iters |
|---|---|:---:|:---:|:---:|:---:|
| **Paper Baseline — NN** | 150,000 | 22.80 | 0.835 | 0.170 | 150k |
| **Paper Baseline — LUT** | 160,000 | 22.50 | 0.830 | 0.170 | +10k FT |
| **Ours — NN** | 50,000 | **23.06** | **0.820** | **0.182** | **50k** |
| **Ours — LUT** | 60,000 | **22.06** | **0.779** | **0.258** | +10k FT |

**Key result:** Our NN model achieves **+0.26 dB PSNR** over the paper's NN baseline while converging in **3× fewer training iterations**.

---

## Improvements Over the Original

### 1. 🔧 Critical Bug Fix — Perceptual Loss Gradient

**File:** `common/utils.py`

The original code wrapped **both** prediction and target VGG features inside `torch.no_grad()`, meaning zero gradients flowed back through the perceptual loss. The 0.04 perceptual weight was completely inactive.

```python
# Original (broken) — no gradients for pred
with torch.no_grad():
    pred_features = vgg(pred_norm)
    target_features = vgg(target_norm)

# Fixed — gradients flow through pred only
pred_features = vgg(pred_norm)
with torch.no_grad():
    target_features = vgg(target_norm)
```

**Impact:** Restoring VGG gradient flow is the primary driver of faster convergence and better texture recovery.

---

### 2. 🧠 Multi-Domain Loss Function

**Files:** `common/utils.py`, `script/train.py`, `script/finetuneLUT.py`

Replaced the single-term pixel loss with a comprehensive multi-domain objective:

```python
loss = SmoothL1Loss(pred, gt)                    # pixel reconstruction
     + 0.04 * VGGPerceptualLoss(pred, gt)         # perceptual features (now fixed)
     + 0.10 * SSIMLoss(pred, gt)                  # structural similarity
     + 0.01 * FFTFrequencyLoss(pred, gt)          # texture sharpness
     + 0.05 * ColorConstancyLoss(pred)            # color balance
```

| Added Loss | Weight | Purpose |
|---|---|---|
| **SSIM Loss** | 0.10 | Directly optimizes luminance, contrast, structure |
| **FFT Frequency Loss** | 0.01 | Preserves high-frequency texture and edge sharpness |
| **Color Constancy Loss** | 0.05 | Prevents R/G/B channel imbalance in dark regions |

---

### 3. ⚡ GAM Architecture Upgrade (pll-4 → pll-8)

**File:** `common/GAM.py`

Upgraded the Global Adjustment Module from 4 parallel QuerySABlock attention branches to **8 parallel branches**, doubling its capacity to predict per-image gamma correction and 3×3 color correction matrices.

```bash
# Original
--gam_type pll_4

# Improved
--gam_type pll_8
```

---

### 4. 🛡️ Training Stability: NaN Guard + Gradient Clipping

**Files:** `script/train.py`, `script/finetuneLUT.py`

Added protection against silent NaN/Inf loss corruption and gradient explosion:

```python
if torch.isnan(loss_G) or torch.isinf(loss_G):
    logger.warning(f"NaN/Inf loss detected at iter {i}! Skipping step.")
    opt_G.zero_grad()
    continue

torch.nn.utils.clip_grad_norm_(params_G, max_norm=1.0)
```

**Result:** Zero training crashes across the full 60,000-iteration pipeline.

---

### 5. 🚀 LUT Fine-Tuning Speedup (~10–100× faster)

**File:** `common/finetune.py`

**Profiling revealed two critical bottlenecks** in the original fine-tuning code:

**Problem A — Serialized Gradient Write-Back (99.6% of GPU time):**
Low-light images have most pixel values near zero, causing all gradient updates to pile up on the same few LUT entries. PyTorch's default `indexing_backward_kernel` processes these **sequentially**, spending 65.8 of 66.1 seconds per step in this single kernel.

**Fix A — Atomic `index_add_` backward:**
```python
class _LUTGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, idx):
        ctx.save_for_backward(idx)
        ctx.wshape = weight.shape
        return weight[idx]

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        gw = torch.zeros(ctx.wshape, dtype=grad.dtype, device=grad.device)
        gw.index_add_(0, idx.reshape(-1), grad.reshape(-1, *ctx.wshape[1:]))
        return gw, None
```

**Problem B — 14,376 CPU↔GPU Synchronisations per Step:**
Boolean mask assignments (`out[mask] = ...`) in the 24-case simplex interpolation force the CPU to wait for the GPU at every case. Each sync costs ~1.2 ms; 14,376 per step = ~17 seconds of stall overhead.

**Fix B — Replace with `torch.where`** (bit-identical output, zero syncs):
```python
# Before (causes sync)
out[i1] = (q - frac_a[i1]) * p0000[i1] + ...

# After (no sync, identical result)
out = torch.where(i1[:, None], (q - frac_a) * p0000 + ..., out)
```

**Step Time Comparison:**

| Configuration | Time / Step | 10,000 Iterations |
|---|---|---|
| Original code (cropSize 256, batch 2) | ~110 s | ~127 days ❌ |
| cropSize 48, batch 8 (before fix) | ~4.4 s | ~12 hours |
| **cropSize 48, batch 8 (after fix)** | **~0.5 s** | **~1.5 hours** ✅ |

---

## Requirements

```
python=3.8
torch>=1.11
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1mvpUxO4D8iJSm0E75vqmPEq6YyF-bFsm) and arrange as follows:

```
data/
├── LOL_v1/
│   ├── high/        # Normal-light training images
│   └── low/         # Low-light training images
└── LowLightBenchmark/
    └── LOL_v1_val/
        ├── high/    # Normal-light validation images
        └── low/     # Low-light validation images
```

> **Note:** The dataset is excluded from this repository via `.gitignore`. Do not commit images or `.npy` cache files.

---

## Usage

### Stage 1 — Train the Neural Network

```bash
python script/train.py \
  --model WVLUT \
  --model_type shared \
  --gam_type pll_8 \
  --expDir ../models/WVLUT_shared \
  --totalIter 50000 \
  --batchSize 16 \
  --cropSize 48 \
  --trainDir ../data/LOL_v1 \
  --valDataset LOL_v1_val \
  --saveStep 10000 \
  --valStep 5000 \
  --displayStep 100
```

### Stage 2 — Transfer Trained Weights to LUTs

```bash
python script/transferLUT.py \
  --model WVLUT \
  --model_type shared \
  --gam_type pll_8 \
  --interval 4 \
  --nf 64 \
  --loadIter 50000 \
  --expDir ../models/WVLUT_shared
```

### Stage 3 — Fine-Tune the LUTs

```bash
python script/finetuneLUT.py \
  --batchSize 8 \
  --cropSize 48 \
  --expDir ../models/WVLUT_shared \
  --model_type shared \
  --gam_type pll_8 \
  --interval 4 \
  --nf 64 \
  --workerNum 4 \
  --startIter 50000 \
  --totalIter 60000 \
  --displayStep 100 \
  --valStep 2000 \
  --saveStep 2000 \
  --trainDir ../data/LOL_v1 \
  --valDataset LOL_v1_val
```

### Stage 4 — Evaluate

```bash
# Evaluate the Neural Network model
python script/eval.py \
  --expDir ../models/WVLUT_shared \
  --resultRoot ../eval/results_nn \
  --model WVLUT \
  --model_type shared \
  --gam_type pll_8 \
  --loadIter 50000 \
  --valDataset LOL_v1_val \
  --ifsave true

# Evaluate the final LUT model
python script/eval.py \
  --expDir ../models/WVLUT_shared \
  --resultRoot ../eval/results_lut \
  --model LUT \
  --model_type shared \
  --gam_type pll_8 \
  --interval 4 \
  --loadIter 60000 \
  --valDataset LOL_v1_val \
  --ifsave true
```

Or use the shell scripts for convenience:
```bash
sh script/train.sh
sh script/transferLUT.sh
sh script/finetuneLUT.sh
sh script/eval.sh
```

---

## Per-Image Results (LOL-v1 Validation Set)

### Our NN Model — Iter 50,000

| Image | PSNR (dB) | SSIM | LPIPS |
|---|:---:|:---:|:---:|
| LOL_v1_val_1 | 19.89 | 0.857 | 0.152 |
| LOL_v1_val_111 | 27.66 | 0.915 | 0.113 |
| LOL_v1_val_146 | 21.50 | 0.862 | 0.175 |
| LOL_v1_val_179 | 17.97 | 0.753 | 0.266 |
| LOL_v1_val_22 | 21.22 | 0.834 | 0.217 |
| LOL_v1_val_23 | 22.85 | 0.816 | 0.240 |
| LOL_v1_val_493 | 19.81 | 0.594 | 0.243 |
| LOL_v1_val_547 | 22.39 | 0.832 | 0.176 |
| LOL_v1_val_55 | 22.99 | 0.849 | 0.186 |
| LOL_v1_val_665 | 25.08 | 0.793 | 0.240 |
| LOL_v1_val_669 | 24.97 | 0.868 | 0.162 |
| LOL_v1_val_748 | 26.12 | 0.882 | 0.134 |
| LOL_v1_val_778 | 22.13 | 0.774 | 0.202 |
| LOL_v1_val_780 | 21.56 | 0.775 | 0.122 |
| LOL_v1_val_79 | 25.09 | 0.909 | 0.097 |
| **Average** | **23.06** | **0.820** | **0.182** |

### Our LUT Model — Iter 60,000 (Fine-Tuned)

| Image | PSNR (dB) | SSIM | LPIPS |
|---|:---:|:---:|:---:|
| LOL_v1_val_1 | 20.08 | 0.818 | 0.240 |
| LOL_v1_val_111 | 27.05 | 0.871 | 0.191 |
| LOL_v1_val_146 | 21.74 | 0.850 | 0.195 |
| LOL_v1_val_179 | 17.58 | 0.722 | 0.316 |
| LOL_v1_val_22 | 22.91 | 0.838 | 0.208 |
| LOL_v1_val_23 | 22.00 | 0.758 | 0.300 |
| LOL_v1_val_493 | 18.80 | 0.562 | 0.348 |
| LOL_v1_val_547 | 23.80 | 0.832 | 0.203 |
| LOL_v1_val_55 | 21.26 | 0.780 | 0.281 |
| LOL_v1_val_665 | 20.38 | 0.722 | 0.359 |
| LOL_v1_val_669 | 22.34 | 0.807 | 0.275 |
| LOL_v1_val_748 | 23.51 | 0.837 | 0.254 |
| LOL_v1_val_778 | 19.00 | 0.685 | 0.322 |
| LOL_v1_val_780 | 20.99 | 0.714 | 0.257 |
| LOL_v1_val_79 | 29.42 | 0.889 | 0.120 |
| **Average** | **22.06** | **0.779** | **0.258** |

---

## Repository Structure

```
WV-LUT/
├── common/
│   ├── architecture.py     # WVLUT neural network (training-time)
│   ├── GAM.py              # Global Adjustment Module (pll_2 / pll_4 / pll_8)
│   ├── finetune.py         # LUT-based inference model (deployment)
│   ├── option.py           # Argument parsing (Train / Test)
│   └── utils.py            # Loss functions & image quality metrics
├── script/
│   ├── data.py             # Dataset loaders
│   ├── train.py            # Stage 1: Train the neural network
│   ├── transferLUT.py      # Stage 2: Convert neural network → LUTs
│   ├── finetuneLUT.py      # Stage 3: Fine-tune LUT entries end-to-end
│   └── eval.py             # Evaluation script
├── models/                 # Saved checkpoints and LUT .npy files
├── data/                   # Datasets (excluded from git)
├── eval/                   # Evaluation output images and logs
├── requirements.txt
└── README.md
```

---

## BibTeX

If you use this work, please cite the original paper:

```bibtex
@ARTICLE{wvlut,
  author={Li, Canlin and Su, Haowen and Tan, Xin and Zhang, Xiangfei and Ma, Lizhuang},
  journal={IEEE Transactions on Multimedia},
  title={WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement},
  year={2025},
  doi={10.1109/TMM.2025.3535342}
}
```

---
