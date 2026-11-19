# Reproducing WV-LUT: A Perceptual-Loss Fix, a Multi-Domain Objective, and Synchronization-Free LUT Fine-Tuning

Code for our reproduction study of
[WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement](https://doi.org/10.1109/TMM.2025.3535342)
(Li *et al.*, IEEE Transactions on Multimedia, 2025).

**Authors:** Yukti Varshney, Harkirat Singh Takhi
Department of Computer Science and Engineering, Netaji Subhas University of Technology, New Delhi

> This repository is built on the
> [official WV-LUT implementation](https://github.com/ssw219/WV-LUT) by Canlin Li, Haowen Su, Xin Tan,
> Xiangfei Zhang and Lizhuang Ma. The model architecture, data pipeline and evaluation code are theirs.
> Our changes are listed below. If you use the WV-LUT method, please cite the original paper.

---

## What we changed

| # | Change | File(s) | Effect |
|---|--------|---------|--------|
| 1 | **Perceptual-loss gradient fix.** The released code computed the prediction's VGG features inside `torch.no_grad()`, so the perceptual term contributed no gradient. Only the target branch is now detached. | `common/utils.py` | The VGG perceptual loss actually trains the model |
| 2 | **Multi-domain objective.** Smooth-L1 + 0.04 VGG + 0.10 SSIM + 0.01 FFT-magnitude + 0.05 colour-constancy. | `common/utils.py`, `script/train.py`, `script/finetuneLUT.py` | Adds structure, frequency and colour terms |
| 3 | **Eight-branch GAM** (`GAM_pll_8` instead of `GAM_pll_4`). | `common/GAM.py`, `common/architecture.py` | Slightly fewer parameters and FLOPs |
| 4 | **Numerical safeguards.** Skip steps with NaN/Inf loss; clip the gradient norm to 1.0. | `script/train.py`, `script/finetuneLUT.py` | No diverged runs in 60k iterations |
| 5 | **Faster LUT fine-tuning.** A custom autograd function (`_LUTGather`) with an `index_add_` backward replaces PyTorch's serialized indexing backward, and `torch.where` replaces boolean-mask assignment in simplex interpolation. | `common/finetune.py` | **8.8× faster** fine-tuning step at a fixed configuration |

### The perceptual-loss fix

```python
# Released code: no gradient reaches the prediction
with torch.no_grad():
    pred_features   = vgg(pred_norm)
    target_features = vgg(target_norm)

# Fixed
pred_features = vgg(pred_norm)            # gradients flow to the model
with torch.no_grad():
    target_features = vgg(target_norm)    # target needs no gradient
```

### The fine-tuning speedup

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

In the 4D simplex interpolation, every `out[m] = f(x[m])` became
`out = torch.where(m[:, None], f(x), out)`, which removes about 14,000 host–device synchronizations per step.

| Configuration (crop / batch) | Seconds per step | 10k iterations |
|---|---|---|
| Released code, 256 / 2 | ~110 | ~12.7 days |
| Released code, 48 / 8 | ~4.4 | ~12.2 hours |
| **Our code, 48 / 8** | **~0.5** | **~1.4 hours** |

Measured on one NVIDIA A100 (40 GB).

---

## Results on LOL-v1 (15 test images)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| WV-LUT network, reported (150k + 10k iterations) | 23.37 | 0.835 | 0.129 |
| WV-LUT 4-bit LUT, reported | 23.31 | 0.833 | 0.129 |
| **Ours, network (50k iterations)** | **23.06** | **0.820** | **0.182** |
| **Ours, 4-bit LUT (10k fine-tuning iterations)** | **22.06** | **0.779** | **0.258** |

"Reported" values are from Table IX of the original paper. Our network reaches 98.7% of the
reported PSNR with 31% of the training iterations. Our deployed LUT is 1.25 dB below the reported
WV-LUT. The paper analyses this gap: most of it comes from five test scenes and shows up as
colour and hue errors.

Metrics are computed per image at full resolution and averaged over the 15 test images.
SSIM uses `skimage.metrics.structural_similarity` on RGB with default settings; LPIPS uses the
AlexNet backbone. No GT-mean correction is applied.

---

## Requirements

- Python 3.8
- PyTorch 1.11.0, CUDA 11.3

```bash
pip install -r requirements.txt
```

## Data preparation

Download LOL-v1 and the evaluation sets from the
[original authors' Google Drive folder](https://drive.google.com/drive/folders/1mvpUxO4D8iJSm0E75vqmPEq6YyF-bFsm)
and arrange them as:

```
data/
├── LOL_v1
│   ├── high
│   └── low
└── LowLightBenchmark
    └── LOL_v1_val
        ├── high
        └── low
```

## Usage

Run from inside `script/`. The settings below reproduce the results in our paper.

**1. Train the network (50k iterations)**

```bash
cd script
python train.py \
  --batchSize 16 --cropSize 96 \
  -e ../models/WVLUT_shared --model_type shared \
  --totalIter 50000 --startIter 0 \
  --lr0 1e-3 --lr1 1e-4 \
  --displayStep 100 --valStep 500 --saveStep 2000 \
  --trainDir ../data/LOL_v1 --valDataset LOL_v1_val
```

**2. Convert the network to 4-bit LUTs**

```bash
python transferLUT.py \
  --expDir ../models/WVLUT_shared --model_type shared \
  --loadIter 50000 --nf 64
```

**3. Fine-tune the LUTs (10k iterations, GAM frozen)**

```bash
python finetuneLUT.py \
  --batchSize 8 --cropSize 96 \
  --expDir ../models/WVLUT_shared --nf 64 \
  --startIter 50000 --totalIter 60000 \
  --lr0 1e-4 --lr1 1e-6 \
  --displayStep 50 --valStep 200 --saveStep 1000 \
  --trainDir ../data/LOL_v1 --valDataset LOL_v1_val
```

**4. Evaluate**

```bash
python eval.py \
  --testDir ../data/LowLightBenchmark \
  --expDir ../models/WVLUT_shared \
  --resultRoot ../eval/results \
  --model WVLUT --model_type shared --nf 64 \
  --loadIter 60000 \
  --valDataset LOL_v1_val \
  --ifsave true --ifGT false
```

---

## Citation

If you use our fixes or results, please cite our paper:

```bibtex
@misc{varshney2026reproducing,
  author = {Varshney, Yukti and Takhi, Harkirat Singh},
  title  = {Reproducing {WV-LUT}: A Perceptual-Loss Fix, a Multi-Domain Objective, and
            Synchronization-Free {LUT} Fine-Tuning for Real-Time Low-Light Image Enhancement},
  year   = {2026},
  note   = {Manuscript}
}
```

and the original WV-LUT paper:

```bibtex
@article{wvlut,
  author  = {Li, Canlin and Su, Haowen and Tan, Xin and Zhang, Xiangfei and Ma, Lizhuang},
  journal = {IEEE Transactions on Multimedia},
  title   = {{WV-LUT}: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement},
  year    = {2025},
  volume  = {27},
  pages   = {4441--4453},
  doi     = {10.1109/TMM.2025.3535342}
}
```

## Acknowledgements

This code builds on the [official WV-LUT implementation](https://github.com/ssw219/WV-LUT),
which in turn builds on [MuLUT](https://github.com/ddlee-cn/MuLUT) and the
[Illumination-Adaptive Transformer](https://github.com/cuiziteng/Illumination-Adaptive-Transformer).
We thank all of their authors for releasing their code.

## Contact

Yukti Varshney: yukti.varshney.ug24@nsut.ac.in