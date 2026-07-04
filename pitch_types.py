"""Shared types and utilities for the Pitch Please detector framework.

Defines the common interface that all pitch detectors implement, plus
shared helpers for audio loading and JSON output that both detectors
reuse to avoid code duplication.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional

import librosa
import numpy as np


@dataclass
class PitchFrame:
    """A single pitch estimate at a point in time.

    Attributes
    ----------
    time_ms : int
        Timestamp of the frame in milliseconds.
    frequency : float | None
        Detected fundamental frequency in Hz, or None if unvoiced.
    confidence : float
        Detector-specific confidence score in the range [0.0, 1.0].
    """

    time_ms: int
    frequency: float | None
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    """Container for the full output of a detector run.

    Bundles the pitch frames with metadata needed for validation
    and downstream processing.
    """

    frames: List[PitchFrame]
    sample_rate: int
    duration_s: float
    processing_time_s: float
    device: str

    @property
    def voiced_frames(self) -> List[PitchFrame]:
        return [f for f in self.frames if f.frequency is not None]

    @property
    def average_confidence(self) -> float:
        return (
            sum(f.confidence for f in self.frames) / len(self.frames)
            if self.frames
            else 0.0
        )


class PitchDetector(ABC):
    """Abstract base class for all pitch detectors.

    Every detector exposes the same public API so the caller does not
    need to know which algorithm is being used:

        detector = PyinDetector()
        result = detector.detect("assets/solo.mp3")
        frames = result.frames
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name for validation output."""
        ...

    @property
    def device(self) -> str:
        """Compute device used by this detector (e.g. 'cpu' or 'cuda')."""
        return "cpu"

    @abstractmethod
    def _run_detection(self, audio: np.ndarray, sr: int) -> List[PitchFrame]:
        """Internal per-detector pitch extraction on already-loaded audio.

        Subclasses implement this with their specific algorithm.
        """
        ...

    def detect(self, audio_path: str) -> DetectionResult:
        """Run pitch detection on an audio file.

        Loads the audio, times the detection, and returns a
        DetectionResult with frames and metadata.
        """
        import time

        y, sr = self.load_audio(audio_path)
        duration_s = len(y) / sr

        t0 = time.perf_counter()
        frames = self._run_detection(y, sr)
        processing_time_s = time.perf_counter() - t0

        return DetectionResult(
            frames=frames,
            sample_rate=sr,
            duration_s=duration_s,
            processing_time_s=processing_time_s,
            device=self.device,
        )

    # ---- Shared helpers (used by all concrete detectors) ----

    @staticmethod
    def load_audio(path: str) -> tuple[np.ndarray, int]:
        """Load an audio file as mono float32 at its native sample rate.

        Returns
        -------
        (np.ndarray, int)
            Mono audio samples (float32) and the sample rate in Hz.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input audio file not found: {path}")

        try:
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception as exc:
            raise RuntimeError(
                f"Unsupported or unreadable audio file: {path}"
            ) from exc

        if y.size == 0:
            raise ValueError("Audio file loaded but contains no samples (empty).")

        y = y.astype(np.float32, copy=False)
        return y, sr

    @staticmethod
    def save_output(frames: List[PitchFrame], output_path: str) -> None:
        """Write pitch frames as a JSON array, creating dirs as needed."""
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        records = [f.to_dict() for f in frames]
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=4)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write output file: {output_path}"
            ) from exc

    def print_validation(self, result: DetectionResult, output_path: str) -> None:
        """Print a standardised validation summary for any detector."""
        voiced = result.voiced_frames
        avg_conf = result.average_confidence

        print(f"---- {self.name} pitch extraction summary ----")
        print(f"Detector              : {self.name}")
        print(f"Device                : {result.device}")
        print(f"Sample rate           : {result.sample_rate} Hz")
        print(f"Audio duration        : {result.duration_s:.3f} s")
        print(f"Frames processed      : {len(result.frames)}")
        print(f"Voiced frames         : {len(voiced)}")
        print(f"Unvoiced frames       : {len(result.frames) - len(voiced)}")
        print(f"Average confidence    : {avg_conf:.4f}")
        print(f"Processing time       : {result.processing_time_s:.3f} s")
        print(f"Output file           : {output_path}")
        print("-" * 52)
