"""torchcrepe-based pitch detector.

Wraps the torchcrepe package (PyTorch CREPE) in the common PitchDetector
interface so it is interchangeable with the pYIN detector.

torchcrepe is a modern PyTorch reimplementation of CREPE that runs on
GPU when available and is orders of magnitude faster than the original
TensorFlow-based crepe package.

Key design points:
  * Automatically selects CUDA if available, otherwise CPU.
  * Resamples audio to 16 kHz (torchcrepe's required sample rate)
    using librosa's high-quality resampler.
  * Uses torchcrepe's batched predict() pipeline — no per-frame loops.
  * Produces one prediction every ~10 ms to match pYIN's resolution.
  * Applies a confidence threshold to mark unvoiced frames as null.
"""

from __future__ import annotations

from typing import List

import librosa
import numpy as np
import torch

from pitch_types import PitchDetector, PitchFrame


# torchcrepe requires audio at exactly 16 kHz.
CREPE_SR = 16000

# Hop length in samples at 16 kHz for ~10 ms resolution.
# 16000 * 0.01 = 160 samples → one prediction every 10 ms.
HOP_LENGTH = 160

# Confidence threshold below which a frame is treated as unvoiced.
# torchcrepe's confidence is the max activation of the output layer (0–1).
# 0.6 is a commonly used threshold: below this the model is uncertain
# and the frequency estimate is unreliable.
CONFIDENCE_THRESHOLD = 0.6

# CREPE model capacity: 'full' gives the best accuracy.
# Other options: 'tiny', 'small', 'medium', 'large'.
MODEL_CAPACITY = "full"

# Target prediction interval — one prediction every ~10 ms.
TARGET_HOP_MS = 10


class TorchCrepeDetector(PitchDetector):
    """Pitch detector using torchcrepe (PyTorch CREPE model)."""

    def __init__(self) -> None:
        self._device = self._select_device()

    @staticmethod
    def _select_device() -> str:
        """Auto-detect CUDA availability and return the device name."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"Using device: {device}")
        return device

    @property
    def name(self) -> str:
        return "torchcrepe"

    @property
    def device(self) -> str:
        return self._device

    def _run_detection(self, audio: np.ndarray, sr: int) -> List[PitchFrame]:
        # Import here so the module can be imported without torchcrepe
        # installed (e.g. when only using the pYIN detector).
        import torchcrepe

        # Resample to 16 kHz if the source rate differs.
        if sr != CREPE_SR:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=CREPE_SR)
        else:
            audio_16k = audio

        # torchcrepe expects a float tensor of shape (batch, samples).
        audio_tensor = torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0)
        audio_tensor = audio_tensor.to(self._device)

        # Run batched inference — torchcrepe handles all frames in one call.
        # With return_periodicity=True, predict() returns (pitch, periodicity)
        # tensors of shape (1, frames).  periodicity is our confidence.
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            sample_rate=CREPE_SR,
            hop_length=HOP_LENGTH,
            model=MODEL_CAPACITY,
            batch_size=1024,
            device=self._device,
            pad=True,
            return_periodicity=True,
        )

        # Move results to CPU as numpy arrays.
        pitch_np = pitch.squeeze(0).cpu().numpy()        # (frames,)
        conf_np = periodicity.squeeze(0).cpu().numpy()    # (frames,)

        # --- RMS energy extraction ---
        # Compute RMS on the original audio using a frame length and hop
        # length that produce the same number of frames as torchcrepe.
        # torchcrepe uses HOP_LENGTH=160 at 16 kHz (10 ms).  We match this
        # on the original sample rate so energy aligns with pitch frames.
        orig_hop = max(1, int(round(sr * TARGET_HOP_MS / 1000.0)))
        orig_frame_length = 2048 if sr <= 48000 else 4096
        rms = librosa.feature.rms(
            y=audio,
            frame_length=orig_frame_length,
            hop_length=orig_hop,
        )[0]

        # --- Frame alignment verification ---
        n_pitch = len(pitch_np)
        if len(rms) != n_pitch:
            raise RuntimeError(
                f"Frame alignment mismatch: torchcrepe produced {n_pitch} "
                f"frames but RMS produced {len(rms)} frames."
            )

        frames: List[PitchFrame] = []
        for i in range(n_pitch):
            # Timestamp from frame index, hop length, and sample rate.
            time_ms = int(round(i * HOP_LENGTH / CREPE_SR * 1000.0))
            conf = float(conf_np[i])

            # Apply confidence threshold: below it, treat as unvoiced.
            is_voiced = conf >= CONFIDENCE_THRESHOLD
            freq = float(pitch_np[i]) if is_voiced else None

            # Energy is always present, even for unvoiced frames.
            energy = float(rms[i]) if i < len(rms) else 0.0

            frames.append(
                PitchFrame(
                    time_ms=time_ms,
                    frequency=freq,
                    confidence=round(conf, 4),
                    voiced=is_voiced,
                    energy=round(energy, 4),
                )
            )

        return frames
