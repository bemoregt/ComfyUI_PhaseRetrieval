"""
Classical Gerchberg-Saxton (GS) phase retrieval node.
Useful as a fast baseline / comparison against deep-prior approach.
"""

import numpy as np
import torch

from .spectrum_utils import (
    image_tensor_to_numpy_gray,
    numpy_gray_to_image_tensor,
    normalize_phase_to_image,
)


class GSPhaseRetrieval:
    """
    Gerchberg-Saxton iterative phase retrieval (classical algorithm).

    Given only the Fourier amplitude |F(u,v)|, iteratively estimates the
    phase using alternating projections between spatial and frequency domains.

    Convergence is not guaranteed for all inputs; typically 100–500 iterations
    is sufficient for a stable estimate.

    This node is provided for comparison with the AI-based nodes.
    """

    CATEGORY = "spectrum2image/phase_retrieval"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 2000,
                    "step": 10,
                }),
                "init_phase": (["random", "zero", "uniform_random"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("phase_spectrum", "debug_reconstruction")
    FUNCTION = "retrieve_phase"

    def retrieve_phase(self,
                       amplitude_spectrum: torch.Tensor,
                       iterations: int,
                       init_phase: str,
                       seed: int):
        np.random.seed(seed)

        # Decode input amplitude (log-normalized [0,1] → raw amplitude)
        amp_norm = image_tensor_to_numpy_gray(amplitude_spectrum)  # (H,W) [0,1]
        # Approximate reversal of log normalization
        # (exact only if amplitude_data is passed; here we use the image directly)
        log_amp = amp_norm * 10.0      # rough scale factor
        amp_raw = np.expm1(log_amp)    # non-negative raw amplitudes

        H, W = amp_raw.shape

        # ── Initialize phase ──────────────────────────────────────────────────
        if init_phase == "zero":
            phase = np.zeros((H, W), dtype=np.float64)
        elif init_phase == "uniform_random":
            phase = np.random.uniform(-np.pi, np.pi, (H, W))
        else:  # "random" – Gaussian noise
            phase = np.random.randn(H, W)

        # ── GS iterations ─────────────────────────────────────────────────────
        # Work in shifted frequency domain
        f_current = amp_raw * np.exp(1j * phase)   # initial estimate in freq domain

        prev_error = np.inf
        for i in range(iterations):
            # 1. Inverse FFT → spatial domain
            f_ishift = np.fft.ifftshift(f_current)
            img_est = np.fft.ifft2(f_ishift).real

            # 2. Apply spatial constraint: keep real part (no negative values for images)
            img_est = np.abs(img_est)

            # 3. Forward FFT → frequency domain
            f_new = np.fft.fftshift(np.fft.fft2(img_est))

            # 4. Replace amplitude with target, keep estimated phase
            est_phase = np.angle(f_new)
            f_current = amp_raw * np.exp(1j * est_phase)

            # Monitor convergence
            if (i + 1) % 50 == 0:
                error = np.mean(np.abs(np.abs(f_new) - amp_raw))
                print(f"[GS iter {i+1:4d}] amplitude error={error:.6f}")
                if abs(prev_error - error) < 1e-8:
                    print(f"[GS] Converged at iteration {i+1}")
                    break
                prev_error = error

        final_phase = np.angle(f_current)  # (H,W) in [-pi, pi]

        # ── Encode output ─────────────────────────────────────────────────────
        phase_norm = normalize_phase_to_image(final_phase)
        phase_img = numpy_gray_to_image_tensor(phase_norm)

        # Debug reconstruction
        from .spectrum_utils import reconstruct_from_amplitude_phase
        recon = reconstruct_from_amplitude_phase(amp_raw, final_phase)
        rmin, rmax = recon.min(), recon.max()
        if rmax - rmin > 1e-12:
            recon = (recon - rmin) / (rmax - rmin)
        debug_img = numpy_gray_to_image_tensor(np.clip(recon, 0, 1))

        return (phase_img, debug_img)
