"""
ComfyUI nodes for image <-> spectrum conversion.
"""

import numpy as np
import torch

from .spectrum_utils import (
    image_tensor_to_numpy_gray,
    numpy_gray_to_image_tensor,
    normalize_amplitude_spectrum,
    normalize_phase_to_image,
    image_to_phase,
    fftshift_amplitude_phase,
    reconstruct_from_amplitude_phase,
)


class ImageToSpectrum:
    """
    Decompose an image into its Fourier amplitude and phase spectra.

    Output amplitude image: log-normalized to [0,1] for visualization.
    Output phase image: mapped from [-pi,pi] to [0,1] (0.5 = zero phase).
    """

    CATEGORY = "spectrum2image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AMPLITUDE_DATA")
    RETURN_NAMES = ("amplitude_spectrum", "phase_spectrum", "amplitude_data")
    FUNCTION = "decompose"

    def decompose(self, image: torch.Tensor):
        gray = image_tensor_to_numpy_gray(image)

        amp, phase, _ = fftshift_amplitude_phase(gray)

        # Save raw amplitude stats for later reconstruction
        log_amp = np.log1p(amp)
        amp_min = float(log_amp.min())
        amp_max = float(log_amp.max())

        # Normalize for visualization
        norm_amp = normalize_amplitude_spectrum(amp)
        norm_phase = normalize_phase_to_image(phase)

        amp_img = numpy_gray_to_image_tensor(norm_amp)
        phase_img = numpy_gray_to_image_tensor(norm_phase)

        # Pass raw amplitude as metadata for downstream reconstruction
        amplitude_data = {
            "raw_amplitude": amp,
            "log_min": amp_min,
            "log_max": amp_max,
            "shape": gray.shape,
        }

        return (amp_img, phase_img, amplitude_data)


class SpectrumToImage:
    """
    Reconstruct a spatial-domain image from amplitude and phase spectra.

    Expects:
      - amplitude_spectrum: [0,1] log-normalized amplitude image
      - phase_spectrum: [0,1] mapped phase image (0.5 = zero phase)
      - amplitude_data (optional): raw amplitude metadata from ImageToSpectrum
        If provided, uses raw amplitude for exact reconstruction.
    """

    CATEGORY = "spectrum2image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "phase_spectrum": ("IMAGE",),
            },
            "optional": {
                "amplitude_data": ("AMPLITUDE_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reconstructed_image",)
    FUNCTION = "reconstruct"

    def reconstruct(self,
                    amplitude_spectrum: torch.Tensor,
                    phase_spectrum: torch.Tensor,
                    amplitude_data=None):
        # Decode phase
        phase_norm = image_tensor_to_numpy_gray(phase_spectrum)
        phase = image_to_phase(phase_norm)

        # Decode amplitude
        if amplitude_data is not None:
            amp = amplitude_data["raw_amplitude"]
        else:
            amp_norm = image_tensor_to_numpy_gray(amplitude_spectrum)
            # Reverse log normalization using image stats as approximation
            log_amp = amp_norm * 10.0  # rough scale; exact only with amplitude_data
            amp = np.expm1(log_amp)

        img = reconstruct_from_amplitude_phase(amp, phase)

        # Normalize output image to [0,1]
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-12:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        return (numpy_gray_to_image_tensor(img),)


class SpectrumVisualizer:
    """
    Visualize amplitude spectrum with adjustable log gain.
    Useful for inspecting high-frequency components.
    """

    CATEGORY = "spectrum2image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amplitude_spectrum": ("IMAGE",),
                "gain": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "colormap": (["gray", "hot", "viridis", "plasma"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"

    def visualize(self, amplitude_spectrum: torch.Tensor, gain: float, colormap: str):
        amp = image_tensor_to_numpy_gray(amplitude_spectrum)
        amp_boosted = np.clip(amp * gain, 0.0, 1.0)

        if colormap == "gray":
            return (numpy_gray_to_image_tensor(amp_boosted),)

        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
            colored = cmap(amp_boosted)[:, :, :3]  # (H,W,3)
            colored = torch.from_numpy(colored.astype(np.float32)).unsqueeze(0)
            return (colored,)
        except ImportError:
            return (numpy_gray_to_image_tensor(amp_boosted),)
