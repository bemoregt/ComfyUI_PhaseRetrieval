"""
AI-based phase retrieval nodes for ComfyUI.

Implements two strategies:
  1. Deep Prior Phase Retrieval  - untrained U-Net as implicit regularizer
  2. Supervised Phase Retrieval  - pre-trained model (load from checkpoint)

Both use the same PhaseUNet architecture.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .spectrum_utils import (
    image_tensor_to_numpy_gray,
    numpy_gray_to_image_tensor,
    image_to_phase,
    normalize_phase_to_image,
)

# ── lazy import to keep ComfyUI startup fast ─────────────────────────────────
def _get_model():
    try:
        from ..models.phase_net import PhaseUNet
    except ImportError:
        import importlib, os, sys
        _dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "phase_net", os.path.join(_dir, "models", "phase_net.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        PhaseUNet = mod.PhaseUNet
    return PhaseUNet


def _get_retrieval_net():
    try:
        from ..models.phase_net import PhaseRetrievalNet
    except ImportError:
        import importlib, os
        _dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "phase_net", os.path.join(_dir, "models", "phase_net.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        PhaseRetrievalNet = mod.PhaseRetrievalNet
    return PhaseRetrievalNet


# ── helpers ───────────────────────────────────────────────────────────────────

def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    """Pad (B,C,H,W) so H and W are multiples of `multiple`."""
    _, _, h, w = tensor.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return tensor, (h, w)
    return nn.functional.pad(tensor, (0, pw, 0, ph), mode='reflect'), (h, w)


def _unpad(tensor: torch.Tensor, orig_hw) -> torch.Tensor:
    h, w = orig_hw
    return tensor[:, :, :h, :w]


def _amplitude_loss(predicted_img: torch.Tensor,
                    target_amp: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Core phase-retrieval loss: |FFT(predicted_img)| should equal target_amp.

    predicted_img: (1,1,H,W) real spatial image
    target_amp:    (1,1,H,W) target Fourier amplitude

    Note: torch.fft.rfft2 does not support autograd on MPS, so the FFT
    is computed on CPU and the loss is moved back to the original device.
    The gradient flows back through the .cpu() device transfer correctly.
    """
    original_device = predicted_img.device

    # Move to CPU for FFT (MPS does not support fft autograd)
    pred_cpu = predicted_img.cpu()
    ta_cpu   = target_amp.detach().cpu()

    fft = torch.fft.rfft2(pred_cpu)               # (1,1,H,W//2+1) complex
    fft_amp = torch.abs(fft) + eps

    _, _, H, W = pred_cpu.shape
    ta = torch.roll(ta_cpu,
                    shifts=(-H // 2, -W // 2),
                    dims=(-2, -1))
    ta_half = ta[:, :, :, :W // 2 + 1]            # rfft2 layout

    # Log-amplitude L1 loss
    loss = torch.mean(torch.abs(torch.log(fft_amp + eps) - torch.log(ta_half + eps)))
    return loss.to(original_device)


def _extract_phase_from_network_output(net_output: torch.Tensor,
                                       target_amp: torch.Tensor) -> np.ndarray:
    """
    Given a spatial image produced by the network, replace its amplitude with
    target_amp and return the phase (as numpy, H,W, in [-pi,pi]).
    """
    img = net_output.squeeze().detach().cpu().numpy()  # (H,W)
    H, W = img.shape

    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)

    amp_np = target_amp.squeeze().cpu().numpy()    # (H,W)
    # Project: keep phase, replace amplitude
    phase = np.angle(f_shift)
    # Optional: consistency check (not mandatory)
    return phase


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Deep Prior Phase Retrieval (no training required)
# ─────────────────────────────────────────────────────────────────────────────

class DeepPriorPhaseRetrieval:
    """
    Recover the phase spectrum from an amplitude spectrum image using the
    Deep Image Prior technique.

    Algorithm:
      1. Initialize a random U-Net (no pre-training needed).
      2. Optimize its weights so that FFT(network_output) has amplitude
         matching the input amplitude spectrum.
      3. The network's implicit bias acts as a regularizer, preventing
         trivial phase solutions and favoring natural-looking phase maps.
      4. Extract phase from the converged FFT.

    Inputs:
      amplitude_spectrum  – [0,1] log-normalized amplitude image
      iterations          – optimization steps (300–2000)
      learning_rate       – Adam LR (1e-3 is usually fine)
      reg_weight          – TV regularization strength
      noise_sigma         – initial input noise level
      seed                – RNG seed (-1 = random)

    Output:
      phase_spectrum      – [0,1] normalized phase image (0.5 = zero phase)
    """

    CATEGORY = "spectrum2image/phase_retrieval"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 500,
                    "min": 50,
                    "max": 5000,
                    "step": 50,
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-3,
                    "min": 1e-5,
                    "max": 1e-1,
                    "step": 1e-5,
                    "display": "number",
                }),
                "reg_weight": ("FLOAT", {
                    "default": 1e-4,
                    "min": 0.0,
                    "max": 1e-1,
                    "step": 1e-5,
                    "display": "number",
                }),
                "noise_sigma": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "base_channels": ("INT", {
                    "default": 32,
                    "min": 16,
                    "max": 128,
                    "step": 16,
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("phase_spectrum", "debug_reconstruction")
    FUNCTION = "retrieve_phase"

    def retrieve_phase(self,
                       amplitude_spectrum: torch.Tensor,
                       iterations: int,
                       learning_rate: float,
                       reg_weight: float,
                       noise_sigma: float,
                       base_channels: int,
                       seed: int):

        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = _best_device()
        PhaseUNet = _get_model()

        # ── Prepare target amplitude ──────────────────────────────────────────
        amp_norm = image_tensor_to_numpy_gray(amplitude_spectrum)  # (H,W) [0,1]
        H, W = amp_norm.shape

        # Convert log-normalized amplitude back to raw scale for loss
        # We keep it in log scale for numerical stability
        log_amp_target = amp_norm  # already log-normalized [0,1]
        amp_target = torch.tensor(log_amp_target, dtype=torch.float32,
                                  device=device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # ── Build network ─────────────────────────────────────────────────────
        net = PhaseUNet(in_channels=1, out_channels=1,
                        base_ch=base_channels).to(device)
        net.train()

        # Fixed noise input (deep prior: network maps fixed noise to image)
        noise_input = torch.randn(1, 1, H, W, device=device) * noise_sigma
        noise_input_padded, orig_hw = _pad_to_multiple(noise_input, multiple=16)

        # Also pad amplitude target
        amp_target_padded, _ = _pad_to_multiple(amp_target, multiple=16)

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        best_loss = float('inf')
        best_phase = None

        print(f"[DeepPriorPhaseRetrieval] Starting optimization: {iterations} iters on {device}")

        for i in range(iterations):
            optimizer.zero_grad()

            # Perturb noise slightly each step (helps escape local minima)
            net_input = noise_input_padded + 0.02 * torch.randn_like(noise_input_padded)

            out_padded = net(net_input)                       # (1,1,H',W') phase image
            out = _unpad(out_padded, orig_hw)                 # (1,1,H,W)

            # Convert network phase output to spatial image: exp(i*phase)*amp
            # But we don't have target image — we build a spatial image whose
            # FFT amplitude matches target. Use network output as a direct spatial image.
            # Map tanh output [-pi,pi] to a plausible spatial image via ifft:
            spatial_img = out / math.pi  # [-1, 1] treat as spatial intensity proxy

            loss = _amplitude_loss(spatial_img, amp_target_padded[:, :, :orig_hw[0], :orig_hw[1]])

            # Total-variation regularization on the output (encourages smoothness)
            if reg_weight > 0:
                tv = (torch.mean(torch.abs(out[:, :, 1:, :] - out[:, :, :-1, :])) +
                      torch.mean(torch.abs(out[:, :, :, 1:] - out[:, :, :, :-1])))
                loss = loss + reg_weight * tv

            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_phase = out.detach().clone()

            if (i + 1) % 100 == 0:
                print(f"  iter {i+1:4d}/{iterations}  loss={loss.item():.6f}  best={best_loss:.6f}")

        print(f"[DeepPriorPhaseRetrieval] Done. Best loss: {best_loss:.6f}")

        # ── Extract phase ─────────────────────────────────────────────────────
        phase_np = best_phase.squeeze().cpu().numpy()  # (H,W) in [-pi, pi]
        phase_norm = normalize_phase_to_image(phase_np)

        phase_img = numpy_gray_to_image_tensor(phase_norm)

        # Debug: reconstruct spatial image using raw amplitude + predicted phase
        amp_raw = image_tensor_to_numpy_gray(amplitude_spectrum)
        # Rough reversal: amp_norm → raw amp (expm1 of scaled value)
        amp_for_recon = np.expm1(amp_raw * 10.0)  # approximate
        from .spectrum_utils import reconstruct_from_amplitude_phase
        recon = reconstruct_from_amplitude_phase(amp_for_recon, phase_np)
        rmin, rmax = recon.min(), recon.max()
        if rmax - rmin > 1e-12:
            recon = (recon - rmin) / (rmax - rmin)
        debug_img = numpy_gray_to_image_tensor(np.clip(recon, 0, 1))

        return (phase_img, debug_img)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Supervised Phase Retrieval (load pre-trained checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class SupervisedPhaseRetrieval:
    """
    Phase retrieval using a pre-trained PhaseRetrievalNet (supervised).

    Loads a checkpoint saved by PhaseRetrievalTrainer.
    If no checkpoint is found, falls back to DeepPriorPhaseRetrieval
    with default settings.

    Checkpoint format: torch.save({'model': net.state_dict(), ...}, path)
    """

    CATEGORY = "spectrum2image/phase_retrieval"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "checkpoint_path": ("STRING", {
                    "default": "models/phase_retrieval.pt",
                    "multiline": False,
                }),
                "fallback_iterations": ("INT", {
                    "default": 300,
                    "min": 50,
                    "max": 2000,
                    "step": 50,
                }),
                "base_channels": ("INT", {
                    "default": 32,
                    "min": 16,
                    "max": 128,
                    "step": 16,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("phase_spectrum", "info")
    FUNCTION = "retrieve_phase"

    def retrieve_phase(self,
                       amplitude_spectrum: torch.Tensor,
                       checkpoint_path: str,
                       fallback_iterations: int,
                       base_channels: int):
        device = _best_device()

        # Try absolute path first, then relative to ComfyUI root
        ckpt_path = checkpoint_path
        if not os.path.isabs(ckpt_path):
            # Relative to this file's location (custom node root)
            node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_path = os.path.join(node_dir, checkpoint_path)

        if os.path.exists(ckpt_path):
            return self._supervised_inference(
                amplitude_spectrum, ckpt_path, base_channels, device)
        else:
            info = (f"Checkpoint not found at '{ckpt_path}'. "
                    f"Falling back to Deep Prior ({fallback_iterations} iters).")
            print(f"[SupervisedPhaseRetrieval] {info}")
            node = DeepPriorPhaseRetrieval()
            phase_img, _ = node.retrieve_phase(
                amplitude_spectrum,
                iterations=fallback_iterations,
                learning_rate=1e-3,
                reg_weight=1e-4,
                noise_sigma=0.1,
                base_channels=base_channels,
                seed=-1,
            )
            return (phase_img, info)

    def _supervised_inference(self, amplitude_spectrum, ckpt_path, base_channels, device):
        PhaseRetrievalNet = _get_retrieval_net()

        net = PhaseRetrievalNet(in_channels=1, out_channels=1,
                                base_ch=base_channels).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model', ckpt)
        net.load_state_dict(state)
        net.eval()

        amp_norm = image_tensor_to_numpy_gray(amplitude_spectrum)
        H, W = amp_norm.shape
        inp = torch.tensor(amp_norm, dtype=torch.float32, device=device
                           ).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        inp_padded, orig_hw = _pad_to_multiple(inp, multiple=16)

        with torch.no_grad():
            out_padded = net(inp_padded)
            out = _unpad(out_padded, orig_hw)           # (1,1,H,W) phase in [-pi,pi]

        phase_np = out.squeeze().cpu().numpy()
        phase_norm = normalize_phase_to_image(phase_np)
        phase_img = numpy_gray_to_image_tensor(phase_norm)

        info = f"Loaded checkpoint: {ckpt_path}"
        return (phase_img, info)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Phase Retrieval Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PhaseRetrievalTrainer:
    """
    Train a supervised PhaseRetrievalNet on a single amplitude/phase pair.

    For real training, feed image pairs from the ImageToSpectrum node and
    accumulate gradient steps. This node trains on a single sample (fine-tuning
    or few-shot regime).

    Outputs:
      checkpoint_path  – where the trained model was saved
      loss_info        – final training loss (string)
    """

    CATEGORY = "spectrum2image/phase_retrieval"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "target_phase": ("IMAGE",),
                "save_path": ("STRING", {
                    "default": "models/phase_retrieval.pt",
                    "multiline": False,
                }),
                "epochs": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 2000,
                    "step": 10,
                }),
                "learning_rate": ("FLOAT", {
                    "default": 2e-4,
                    "min": 1e-6,
                    "max": 1e-2,
                    "step": 1e-6,
                    "display": "number",
                }),
                "base_channels": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 128,
                    "step": 16,
                }),
                "load_existing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("checkpoint_path", "loss_info")
    FUNCTION = "train"

    def train(self,
              amplitude_spectrum: torch.Tensor,
              target_phase: torch.Tensor,
              save_path: str,
              epochs: int,
              learning_rate: float,
              base_channels: int,
              load_existing: bool):
        PhaseRetrievalNet = _get_retrieval_net()
        device = _best_device()

        net = PhaseRetrievalNet(in_channels=1, out_channels=1,
                                base_ch=base_channels).to(device)

        # Resolve save path
        abs_save = save_path
        if not os.path.isabs(abs_save):
            node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_save = os.path.join(node_dir, save_path)

        os.makedirs(os.path.dirname(abs_save), exist_ok=True)

        if load_existing and os.path.exists(abs_save):
            ckpt = torch.load(abs_save, map_location=device)
            net.load_state_dict(ckpt.get('model', ckpt))
            print(f"[PhaseRetrievalTrainer] Loaded existing checkpoint: {abs_save}")

        # Prepare tensors
        amp_np = image_tensor_to_numpy_gray(amplitude_spectrum)
        phase_np = image_tensor_to_numpy_gray(target_phase)

        inp = torch.tensor(amp_np, dtype=torch.float32,
                           device=device).unsqueeze(0).unsqueeze(0)
        # Target phase: [0,1] image → [-pi,pi]
        tgt_phase = image_to_phase(phase_np)
        tgt = torch.tensor(tgt_phase, dtype=torch.float32,
                           device=device).unsqueeze(0).unsqueeze(0)

        inp_padded, orig_hw = _pad_to_multiple(inp, multiple=16)
        tgt_padded, _ = _pad_to_multiple(tgt, multiple=16)

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.L1Loss()

        net.train()
        last_loss = 0.0
        print(f"[PhaseRetrievalTrainer] Training {epochs} epochs on {device}")

        for ep in range(epochs):
            optimizer.zero_grad()
            out = net(inp_padded)
            out_unpad = _unpad(out, orig_hw)
            loss = criterion(out_unpad, tgt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = loss.item()

            if (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1:4d}/{epochs}  loss={last_loss:.6f}")

        torch.save({'model': net.state_dict(), 'loss': last_loss}, abs_save)
        info = f"epochs={epochs}  final_loss={last_loss:.6f}"
        print(f"[PhaseRetrievalTrainer] Saved to {abs_save}  {info}")
        return (abs_save, info)
