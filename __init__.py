"""
spectrum2image – ComfyUI custom node pack for Fourier phase retrieval.

Nodes:
  • ImageToSpectrum          – image → amplitude + phase spectra
  • SpectrumToImage          – amplitude + phase → image
  • SpectrumVisualizer       – visualize amplitude with color mapping
  • DeepPriorPhaseRetrieval  – AI phase retrieval (no training required)
  • SupervisedPhaseRetrieval – AI phase retrieval with pre-trained checkpoint
  • PhaseRetrievalTrainer    – train a supervised model
  • GSPhaseRetrieval         – classical Gerchberg-Saxton (comparison)
"""

from .nodes.image_spectrum_nodes import (
    ImageToSpectrum,
    SpectrumToImage,
    SpectrumVisualizer,
)
from .nodes.phase_retrieval_node import (
    DeepPriorPhaseRetrieval,
    SupervisedPhaseRetrieval,
    PhaseRetrievalTrainer,
)
from .nodes.gs_phase_retrieval import GSPhaseRetrieval

NODE_CLASS_MAPPINGS = {
    "ImageToSpectrum":            ImageToSpectrum,
    "SpectrumToImage":            SpectrumToImage,
    "SpectrumVisualizer":         SpectrumVisualizer,
    "DeepPriorPhaseRetrieval":    DeepPriorPhaseRetrieval,
    "SupervisedPhaseRetrieval":   SupervisedPhaseRetrieval,
    "PhaseRetrievalTrainer":      PhaseRetrievalTrainer,
    "GSPhaseRetrieval":           GSPhaseRetrieval,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToSpectrum":            "Image → Spectrum",
    "SpectrumToImage":            "Spectrum → Image",
    "SpectrumVisualizer":         "Spectrum Visualizer",
    "DeepPriorPhaseRetrieval":    "Phase Retrieval (Deep Prior AI)",
    "SupervisedPhaseRetrieval":   "Phase Retrieval (Supervised AI)",
    "PhaseRetrievalTrainer":      "Phase Retrieval Trainer",
    "GSPhaseRetrieval":           "Phase Retrieval (Gerchberg-Saxton)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
