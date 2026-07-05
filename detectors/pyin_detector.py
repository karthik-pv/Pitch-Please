"""pYIN-based pitch detector.

Wraps librosa's pyin() in the common PitchDetector interface so it is
interchangeable with the torchcrepe detector.

In addition to pitch, this detector computes RMS energy per frame using
the exact same frame_length and hop_length as pYIN, ensuring a perfect
one-to-one correspondence between frequency, confidence, voiced flag,
and energy for every frame.
"""

from __future__ import annotations

from typing import List

import librosa
import numpy as np

from pitch_types import PitchDetector, PitchFrame


# Guitar-friendly pYIN configuration.
# Standard tuning E2 ~ 82 Hz; lead guitar can reach ~ 1500 Hz.
FMIN = 80.0
FMAX = 1500.0
TARGET_HOP_MS = 10  # one pitch estimate every ~10 ms


class PyinDetector(PitchDetector):
    """Pitch detector using librosa's probabilistic YIN algorithm."""

    @property
    def name(self) -> str:
        return "pYIN"

    def _run_detection(self, audio: np.ndarray, sr: int) -> List[PitchFrame]:
        hop_length = max(1, int(round(sr * TARGET_HOP_MS / 1000.0)))

        # frame_length: 2048 is a good default for music signals; for very
        # high sample rates a larger window improves low-frequency resolution.
        frame_length = 2048 if sr <= 48000 else 4096

        # --- Pitch detection (unchanged) ---
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=np.nan,  # keep NaN so unvoiced frames are explicit
        )

        # --- RMS energy extraction ---
        # Uses the exact same frame_length and hop_length as pYIN so that
        # every RMS value corresponds to the same audio frame as the pitch
        # estimate.  This guarantees a one-to-one alignment by index.
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]  # shape: (frames,)

        # --- Frame alignment verification ---
        # All arrays must have identical length.  If they differ, the data
        # is corrupt and must not be silently truncated.
        if len(f0) != len(rms):
            raise RuntimeError(
                f"Frame alignment mismatch: pYIN produced {len(f0)} frames "
                f"but RMS produced {len(rms)} frames. "
                f"This indicates a parameter inconsistency."
            )

        # --- Build output frames ---
        frames: List[PitchFrame] = []
        for i in range(len(f0)):
            time_ms = int(round(i * hop_length / sr * 1000.0))
            is_voiced = bool(voiced_flag[i]) if voiced_flag is not None else False
            freq = float(f0[i]) if is_voiced and not np.isnan(f0[i]) else None
            confidence = float(voiced_prob[i]) if voiced_prob is not None else 0.0
            # Energy is always present, even for unvoiced frames.
            # It represents signal strength, not pitch confidence.
            energy = float(rms[i]) if i < len(rms) else 0.0
            frames.append(
                PitchFrame(
                    time_ms=time_ms,
                    frequency=freq,
                    confidence=round(confidence, 4),
                    voiced=is_voiced,
                    energy=round(energy, 4),
                )
            )

        return frames
