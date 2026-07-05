"""Pitch Trajectory Recovery Engine.

Orchestrates all recovery stages:
  1. Load JSON
  2. Segment detection
  3. Frame classification
  4. Outlier detection
  5. Tiny gap recovery
  6. Continuous curve fitting (PCHIP)
  7. Confidence-weighted recovery
  8. Guitar physics constraints
  9. Output

This module does NOT improve pitch detection.  It estimates the most
likely continuous guitar pitch trajectory from noisy detector
observations, preserving musical expression while removing detector
artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np

from recovery_types import RecoveredFrame, VALID, UNCERTAIN, OUTLIER, SILENT, RECOVERED
from segment_detector import detect_segments
from frame_classifier import classify_frames
from curve_fitter import recover_segment


# Maximum allowed instantaneous frequency ratio between consecutive
# recovered frames.  If the ratio exceeds this, the jump is considered
# physically impossible for a guitar and is smoothed.
MAX_FREQ_RATIO = 2.5  # ~1.3 octaves — beyond a slide, likely an artifact


def load_frames(path: str) -> List[dict]:
    """Stage 1 — Load the detector JSON, keeping every field."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input JSON is empty or not a list of records.")

    return data


def apply_physics_constraints(
    times: np.ndarray,
    recovered: np.ndarray,
    states: List[str],
) -> np.ndarray:
    """Stage 8 — Enforce simple guitar physics constraints.

    Prevents instantaneous octave jumps that are physically impossible.
    If two consecutive recovered frames have a frequency ratio exceeding
    MAX_FREQ_RATIO, the second frame is adjusted toward the first to
    maintain continuity.  This only applies within segments (SILENT
    frames are skipped).
    """
    n = len(times)
    result = recovered.copy()

    for i in range(1, n):
        if states[i] == SILENT or states[i - 1] == SILENT:
            continue

        prev = result[i - 1]
        curr = result[i]

        if np.isnan(prev) or np.isnan(curr) or prev <= 0 or curr <= 0:
            continue

        ratio = curr / prev
        if ratio > MAX_FREQ_RATIO:
            # Downward jump too large — bring closer to previous.
            result[i] = prev * MAX_FREQ_RATIO
        elif ratio < 1.0 / MAX_FREQ_RATIO:
            # Upward jump too large — bring closer to previous.
            result[i] = prev / MAX_FREQ_RATIO

    return result


def run_recovery(input_path: str) -> List[RecoveredFrame]:
    """Run the full trajectory recovery pipeline on a detector JSON file."""
    # Stage 1 — Load
    raw_frames = load_frames(input_path)

    n = len(raw_frames)
    times_ms = np.array([f["time_ms"] for f in raw_frames], dtype=np.float64)
    frequencies = [f.get("frequency") for f in raw_frames]
    confidences = [f.get("confidence", 0.0) for f in raw_frames]
    voiced_flags = [f.get("voiced", False) for f in raw_frames]
    energies = [f.get("energy", 0.0) for f in raw_frames]

    # Stage 2 — Segment detection
    segments = detect_segments(frequencies, voiced_flags, energies)

    # Stage 3 & 4 — Frame classification (includes outlier detection)
    states = classify_frames(
        frequencies, confidences, voiced_flags, energies, segments
    )

    # Mark frames outside segments as SILENT.
    in_segment = [False] * n
    for start, end in segments:
        for i in range(start, end + 1):
            in_segment[i] = True
    for i in range(n):
        if not in_segment[i]:
            states[i] = SILENT

    # Prepare arrays for recovery.
    freq_arr = np.array(
        [f if f is not None else np.nan for f in frequencies], dtype=np.float64
    )
    times_s = times_ms / 1000.0  # convert to seconds for interpolation

    # Stages 5–7 — Recovery per segment
    recovered_freq = freq_arr.copy()
    recovered_states = list(states)

    for seg_start, seg_end in segments:
        seg_times = times_s[seg_start:seg_end + 1]
        seg_freqs = freq_arr[seg_start:seg_end + 1]
        seg_confs = np.array(confidences[seg_start:seg_end + 1], dtype=np.float64)
        seg_states = states[seg_start:seg_end + 1]

        seg_recovered, seg_updated = recover_segment(
            seg_times, seg_freqs, seg_confs, seg_states,
            energy_threshold=0.0,  # already handled by classification
        )

        recovered_freq[seg_start:seg_end + 1] = seg_recovered
        recovered_states[seg_start:seg_end + 1] = seg_updated

    # Stage 8 — Physics constraints
    recovered_freq = apply_physics_constraints(times_s, recovered_freq, recovered_states)

    # Stage 9 — Build output
    result: List[RecoveredFrame] = []
    for i in range(n):
        raw_f = frequencies[i]
        rec_f = float(recovered_freq[i]) if not np.isnan(recovered_freq[i]) else None

        # For SILENT frames, recovered_frequency is None.
        if recovered_states[i] == SILENT:
            rec_f = None

        # For VALID frames, recovered_frequency equals raw (unchanged).
        if recovered_states[i] == VALID:
            rec_f = raw_f

        result.append(RecoveredFrame(
            time_ms=int(times_ms[i]),
            raw_frequency=raw_f,
            recovered_frequency=rec_f,
            confidence=round(confidences[i], 4),
            voiced=voiced_flags[i],
            energy=round(energies[i], 4),
            state=recovered_states[i],
        ))

    return result


def save_recovered(frames: List[RecoveredFrame], output_path: str) -> None:
    """Write the recovered frames as a JSON array."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    records = [f.to_dict() for f in frames]
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=4)
    except OSError as exc:
        raise RuntimeError(f"Failed to write output file: {output_path}") from exc


def print_statistics(frames: List[RecoveredFrame], num_segments: int) -> None:
    """Print recovery statistics for validation."""
    n = len(frames)
    state_counts = {}
    for f in frames:
        state_counts[f.state] = state_counts.get(f.state, 0) + 1

    valid = state_counts.get(VALID, 0)
    uncertain = state_counts.get(UNCERTAIN, 0)
    outlier = state_counts.get(OUTLIER, 0)
    silent = state_counts.get(SILENT, 0)
    recovered = state_counts.get(RECOVERED, 0)

    avg_conf = sum(f.confidence for f in frames) / n if n else 0.0
    avg_energy = sum(f.energy for f in frames) / n if n else 0.0

    print("---- Trajectory Recovery summary ----")
    print(f"Total frames          : {n}")
    print(f"Total segments        : {num_segments}")
    print(f"Valid frames          : {valid}")
    print(f"Uncertain frames      : {uncertain}")
    print(f"Outliers removed      : {outlier}")
    print(f"Recovered frames      : {recovered}")
    print(f"Silent frames         : {silent}")
    print(f"Average confidence    : {avg_conf:.4f}")
    print(f"Average energy        : {avg_energy:.4f}")
    print("-" * 40)
