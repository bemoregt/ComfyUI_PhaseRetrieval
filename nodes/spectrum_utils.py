"""
Utility functions for spectrum manipulation shared across nodes.
"""

import numpy as np
import torch


def image_tensor_to_numpy_gray(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert ComfyUI IMAGE tensor (B,H,W,C) float32 [0,1] to (H,W) float64.
    Handles both grayscale (C=1) and RGB (C=3) by converting to luminance.
    """
    img = image_tensor[0].cpu().numpy()  # (H,W,C)
    if img.shape[2] == 1:
        return img[:, :, 0].astype(np.float64)
    # Luminance from RGB
    return (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(np.float64)


def numpy_gray_to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert (H,W) float array to ComfyUI IMAGE tensor (1,H,W,3) float32 [0,1].
    Values are clipped to [0,1].
    """
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    arr_rgb = np.stack([arr, arr, arr], axis=-1)      # (H,W,3)
    return torch.from_numpy(arr_rgb).unsqueeze(0)      # (1,H,W,3)


def normalize_amplitude_spectrum(amp: np.ndarray) -> np.ndarray:
    """
    Log-scale normalize amplitude spectrum to [0,1].
    amp: raw FFT amplitude (non-negative)
    """
    log_amp = np.log1p(amp)
    mn, mx = log_amp.min(), log_amp.max()
    if mx - mn < 1e-12:
        return np.zeros_like(log_amp)
    return (log_amp - mn) / (mx - mn)


def denormalize_amplitude_spectrum(norm_amp: np.ndarray,
                                   original_min: float,
                                   original_max: float) -> np.ndarray:
    """Reverse of normalize_amplitude_spectrum."""
    log_amp = norm_amp * (original_max - original_min) + original_min
    return np.expm1(log_amp)


def normalize_phase_to_image(phase: np.ndarray) -> np.ndarray:
    """
    Map phase in [-pi, pi] to [0, 1].
    """
    return (phase + np.pi) / (2.0 * np.pi)


def image_to_phase(img: np.ndarray) -> np.ndarray:
    """
    Map image in [0, 1] back to phase in [-pi, pi].
    """
    return img * 2.0 * np.pi - np.pi


def fftshift_amplitude_phase(image: np.ndarray):
    """
    Compute 2D FFT of image and return shifted amplitude and phase.
    Returns:
        amp: (H,W) non-negative amplitude
        phase: (H,W) phase in [-pi, pi]
        fft_complex: (H,W) complex array (for reconstruction)
    """
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)
    amp = np.abs(f_shift)
    phase = np.angle(f_shift)
    return amp, phase, f_shift


def reconstruct_from_amplitude_phase(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Reconstruct real image from amplitude and phase.
    """
    f_complex = amp * np.exp(1j * phase)
    f_ishift = np.fft.ifftshift(f_complex)
    img = np.fft.ifft2(f_ishift).real
    return img
