"""Detector package — exports all available pitch detectors."""

from .pyin_detector import PyinDetector
from .torchcrepe_detector import TorchCrepeDetector

__all__ = ["PyinDetector", "TorchCrepeDetector"]
