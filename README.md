# spectrum2image – ComfyUI Phase Retrieval Nodes

A ComfyUI custom node pack for **AI-based Fourier phase retrieval**.
Given only an amplitude (magnitude) spectrum image, reconstruct the corresponding phase spectrum using deep learning — no pre-trained weights required.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_PhaseRetrieval/blob/main/ScrShot%206.png)

---

## Overview

In many imaging and signal-processing tasks (holography, X-ray diffraction, astronomy, audio), only the Fourier amplitude `|F(u,v)|` is measurable while the phase `∠F(u,v)` is lost. Recovering the phase from the amplitude alone is the **phase retrieval problem**.

This node pack implements:

| Approach | Node | Notes |
|---|---|---|
| **Deep Prior (AI)** | `Phase Retrieval (Deep Prior AI)` | Untrained U-Net as implicit regularizer. Works out of the box — no dataset needed. |
| **Supervised (AI)** | `Phase Retrieval (Supervised AI)` | Load a checkpoint trained with the Trainer node. Falls back to Deep Prior if none found. |
| **Supervised Training** | `Phase Retrieval Trainer` | Train a PhaseRetrievalNet on amplitude / phase image pairs. |
| **Gerchberg-Saxton** | `Phase Retrieval (Gerchberg-Saxton)` | Classical iterative algorithm. Fast baseline for comparison. |
| **Spectrum utilities** | `Image → Spectrum`, `Spectrum → Image`, `Spectrum Visualizer` | Decompose / reconstruct / visualize FFT spectra. |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI_spectrum2image
pip install -r ComfyUI_spectrum2image/requirements.txt
```

Restart ComfyUI — all nodes will appear under the **spectrum2image** category.

**Requirements:** `torch >= 2.0`, `numpy >= 1.24`, `matplotlib >= 3.7`
GPU acceleration is used automatically (CUDA → MPS → CPU).

---

## Nodes

### Image → Spectrum
Decomposes a spatial image into its Fourier amplitude and phase spectra.

| Pin | Type | Description |
|---|---|---|
| `image` | IMAGE | Input spatial-domain image (grayscale or RGB) |
| → `amplitude_spectrum` | IMAGE | Log-normalized amplitude `[0, 1]` |
| → `phase_spectrum` | IMAGE | Phase mapped from `[-π, π]` to `[0, 1]` |
| → `amplitude_data` | AMPLITUDE_DATA | Raw amplitude metadata for lossless reconstruction |

> Phase encoding: pixel value `0.5` = zero phase, `0.0` = `−π`, `1.0` = `+π`.

---

### Spectrum → Image
Reconstructs a spatial image from amplitude and phase spectra.

| Pin | Type | Description |
|---|---|---|
| `amplitude_spectrum` | IMAGE | Log-normalized amplitude image |
| `phase_spectrum` | IMAGE | Phase image `[0, 1]` |
| `amplitude_data` *(optional)* | AMPLITUDE_DATA | Pass from `Image → Spectrum` for exact amplitude reconstruction |
| → `reconstructed_image` | IMAGE | Recovered spatial image |

---

### Phase Retrieval (Deep Prior AI)

Recovers the phase spectrum from amplitude alone using the **Deep Image Prior** technique.

**Algorithm:**
1. A randomly initialized U-Net maps fixed noise → spatial image.
2. Adam optimizer minimizes `‖log|FFT(output)| − log|target_amplitude|‖₁`.
3. The network's implicit bias acts as a regularizer, preventing trivial solutions.
4. Phase is extracted from the converged FFT.

| Parameter | Default | Description |
|---|---|---|
| `amplitude_spectrum` | — | Log-normalized amplitude image `[0, 1]` |
| `iterations` | 500 | Optimization steps (300–2000 recommended) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `reg_weight` | 1e-4 | Total-variation regularization weight |
| `noise_sigma` | 0.1 | Amplitude of the fixed noise input |
| `base_channels` | 32 | U-Net base channel count (16 = fast, 64 = high quality) |
| `seed` | -1 | RNG seed; `-1` for random |
| → `phase_spectrum` | IMAGE | Recovered phase `[0, 1]` |
| → `debug_reconstruction` | IMAGE | Spatial image reconstructed from output phase |

---

### Phase Retrieval (Supervised AI)

Uses a pre-trained `PhaseRetrievalNet` checkpoint for fast, single-pass inference.
Falls back to Deep Prior automatically when no checkpoint is found.

| Parameter | Default | Description |
|---|---|---|
| `amplitude_spectrum` | — | Input amplitude image |
| `checkpoint_path` | `models/phase_retrieval.pt` | Path to `.pt` checkpoint (absolute or relative to node root) |
| `fallback_iterations` | 300 | Deep Prior iterations used when checkpoint is missing |
| → `phase_spectrum` | IMAGE | Recovered phase |
| → `info` | STRING | Status message (loaded path or fallback reason) |

---

### Phase Retrieval Trainer

Fine-tunes or trains a supervised `PhaseRetrievalNet` on a single amplitude / phase pair.
For multi-image training, loop this node across a dataset using ComfyUI's batch tools.

| Parameter | Default | Description |
|---|---|---|
| `amplitude_spectrum` | — | Input amplitude image |
| `target_phase` | — | Ground-truth phase image (from `Image → Spectrum`) |
| `save_path` | `models/phase_retrieval.pt` | Where to save the checkpoint |
| `epochs` | 200 | Training steps |
| `learning_rate` | 2e-4 | Adam learning rate |
| `base_channels` | 64 | Network capacity |
| `load_existing` | true | Resume from existing checkpoint if present |
| → `checkpoint_path` | STRING | Absolute path of saved checkpoint |
| → `loss_info` | STRING | Final training loss summary |

---

### Phase Retrieval (Gerchberg-Saxton)

Classical alternating-projection phase retrieval. Fast and deterministic; useful as a baseline.

| Parameter | Default | Description |
|---|---|---|
| `amplitude_spectrum` | — | Input amplitude image |
| `iterations` | 200 | GS iteration count |
| `init_phase` | `random` | Phase initialization: `random`, `zero`, or `uniform_random` |
| `seed` | 42 | RNG seed |
| → `phase_spectrum` | IMAGE | Recovered phase |
| → `debug_reconstruction` | IMAGE | Spatial reconstruction |

---

### Spectrum Visualizer

Enhances amplitude spectrum visibility with adjustable gain and colormap.

| Parameter | Default | Description |
|---|---|---|
| `amplitude_spectrum` | — | Input amplitude image |
| `gain` | 1.0 | Brightness multiplier `[0.1, 10.0]` |
| `colormap` | `gray` | `gray` / `hot` / `viridis` / `plasma` |

---

## Example Workflows

### Workflow A – Analysis & Reconstruction
```
[Load Image]
     │
     ▼
[Image → Spectrum]
     ├── amplitude_spectrum ──────────────────────────────────────┐
     ├── phase_spectrum (ground truth, for comparison)            │
     └── amplitude_data ──────────────────────────────────────────┤
                                                                   │
[Deep Prior Phase Retrieval]  ◄── amplitude_spectrum              │
     └── phase_spectrum (predicted)                               │
                │                                                  │
                ▼                                                  │
     [Spectrum → Image] ◄── amplitude_data ──────────────────────┘
           └── [Preview]
```

### Workflow B – Train Then Infer
```
[Load Image A]
     ├── [Image → Spectrum] ──► amplitude ─┐
     │                                     ├─► [Phase Retrieval Trainer]
     └────────────────────► phase ─────────┘         │
                                               checkpoint_path
                                                      │
[Load Image B]                                        ▼
     └── [Image → Spectrum] ──► amplitude ──► [Supervised Phase Retrieval]
                                                      │
                                               phase_spectrum
```

### Workflow C – GS vs. Deep Prior Comparison
```
[Load Image] → [Image → Spectrum] → amplitude_spectrum
                                          │
                       ┌──────────────────┴──────────────────┐
                       ▼                                      ▼
        [GS Phase Retrieval]              [Deep Prior Phase Retrieval]
               │                                              │
        [Spectrum → Image]                        [Spectrum → Image]
               │                                              │
           [Preview]                                      [Preview]
```

---

## Model Architecture

Both networks are U-Net variants defined in `models/phase_net.py`.

```
PhaseUNet (Deep Prior)
  Input:  noise  (1 × H × W)
  Output: phase  (1 × H × W)  ∈ [−π, π]   via tanh(·) × π

  Encoder: 4× DownBlock (Conv-BN-LeakyReLU × 2 + MaxPool)
  Bottleneck: ConvBlock with Dropout(0.3)
  Decoder: 4× UpBlock (ConvTranspose + skip concat + ConvBlock)

PhaseRetrievalNet (Supervised)
  Input:  log-normalized amplitude  (1 × H × W)
  Output: phase                     (1 × H × W)  ∈ [−π, π]

  Same U-Net backbone + residual skip from input to output.
  base_ch=64 by default for higher capacity.
```

---

## File Structure

```
ComfyUI_spectrum2image/
├── __init__.py                  # Node registration (NODE_CLASS_MAPPINGS)
├── requirements.txt
├── models/
│   └── phase_net.py             # PhaseUNet, PhaseRetrievalNet
└── nodes/
    ├── spectrum_utils.py        # FFT helpers, normalization
    ├── image_spectrum_nodes.py  # ImageToSpectrum, SpectrumToImage, Visualizer
    ├── phase_retrieval_node.py  # DeepPriorPhaseRetrieval, Supervised, Trainer
    └── gs_phase_retrieval.py    # GSPhaseRetrieval
```

---

## Tips

- **Starting point:** Use Deep Prior with `iterations=500`, `base_channels=32`. Good balance of quality and speed.
- **High-quality result:** Increase to `iterations=1500`, `base_channels=64`. Expect ~2–5 min on CPU.
- **Faster iteration:** Set `base_channels=16`, `iterations=200` for quick previews.
- **Supervised speedup:** Train on a few image pairs with the Trainer node, then use Supervised for instant inference on similar images.
- **GS baseline:** Always compare Deep Prior output against GS — if GS is already good, Deep Prior may not be necessary.
- **Lossless reconstruction:** Always pass `amplitude_data` from `Image → Spectrum` directly into `Spectrum → Image` to avoid approximation errors in amplitude de-normalization.

---

## License

MIT
