"""Stage 3 & 4 — Frame Classification and Outlier Detection.

Classifies each frame into one of four states:
  VALID    — high confidence, voiced, good energy, consistent with neighbors
  UNCERTAIN — low confidence or null frequency, but high energy and within a segment
  OUTLIER  — frequency exists but deviates sharply from local trend
  SILENT   — low energy, unvoiced, represents an intentional gap

Outlier detection uses a ±5 frame local window and checks whether a
frame's frequency deviates significantly from the median of its
neighbors while those neighbors are themselves consistent.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from recovery_types import VALID, UNCERTAIN, OUTLIER, SILENT


# --- Thresholds ---

# Minimum confidence for a frame to be considered VALID.
CONFIDENCE_THRESHOLD = 0.5

# Minimum energy for a frame to be considered active (not silent).
# Uses the same adaptive approach as segment detection.
ENERGY_SILENCE_FACTOR = 0.15

# Outlier detection window (±N frames).
OUTLIER_WINDOW = 5

# A frame is an outlier if its frequency deviates from the local median
# by more than this factor (relative).  E.g. 0.5 = 50% deviation.
OUTLIER_DEVIATION_FACTOR = 0.5

# Minimum number of consistent neighbors required to call something an outlier.
OUTLIER_MIN_NEIGHBORS = 3

# Maximum relative deviation among neighbors for them to be "consistent".
NEIGHBOR_CONSISTENCY_FACTOR = 0.3


def _local_median(values: List[Optional[float]], center: int, window: int) -> tuple[float, float]:
    """Compute the median frequency and consistency of neighbors around center.

    Returns (median_freq, neighbor_consistency_ratio) where consistency
    is the fraction of neighbors within NEIGHBOR_CONSISTENCY_FACTOR of
    the median.
    """
    n = len(values)
    start = max(0, center - window)
    end = min(n, center + window + 1)

    neighbor_freqs = []
    for i in range(start, end):
        if i == center:
            continue
        f = values[i]
        if f is not None and f > 0:
            neighbor_freqs.append(f)

    if len(neighbor_freqs) < OUTLIER_MIN_NEIGHBORS:
        return (0.0, 0.0)

    med = float(np.median(neighbor_freqs))
    if med <= 0:
        return (0.0, 0.0)

    consistent = sum(
        1 for f in neighbor_freqs if abs(f - med) / med <= NEIGHBOR_CONSISTENCY_FACTOR
    )
    consistency_ratio = consistent / len(neighbor_freqs)

    return (med, consistency_ratio)


def classify_frames(
    frequencies: List[Optional[float]],
    confidences: List[float],
    voiced_flags: List[bool],
    energies: List[float],
    segment_ranges: List[tuple[int, int]],
) -> List[str]:
    """Classify every frame into VALID, UNCERTAIN, OUTLIER, or SILENT.

    segment_ranges is used to know which frames belong to a musical
    segment (frames outside segments are SILENT).
    """
    n = len(frequencies)
    states = [SILENT] * n

    # Build a set of indices that belong to a segment.
    in_segment = [False] * n
    for start, end in segment_ranges:
        for i in range(start, end + 1):
            in_segment[i] = True

    # Compute adaptive energy threshold.
    energy_arr = np.array(energies, dtype=np.float64)
    positive = energy_arr[energy_arr > 0]
    median_energy = float(np.median(positive)) if positive.size > 0 else 0.0
    energy_threshold = median_energy * ENERGY_SILENCE_FACTOR

    # First pass: classify as SILENT, UNCERTAIN, or VALID (pre-outlier).
    for i in range(n):
        if not in_segment[i]:
            states[i] = SILENT
            continue

        if energies[i] < energy_threshold and not voiced_flags[i]:
            states[i] = SILENT
            continue

        freq = frequencies[i]
        conf = confidences[i]
        voiced = voiced_flags[i]

        if freq is None or not voiced or conf < CONFIDENCE_THRESHOLD:
            states[i] = UNCERTAIN
        else:
            states[i] = VALID

    # Second pass: detect outliers among VALID frames.
    for i in range(n):
        if states[i] != VALID:
            continue

        freq = frequencies[i]
        if freq is None or freq <= 0:
            continue

        med, consistency = _local_median(frequencies, i, OUTLIER_WINDOW)

        if med <= 0 or consistency < 0.6:
            continue  # Not enough consistent neighbors to judge

        deviation = abs(freq - med) / med
        if deviation > OUTLIER_DEVIATION_FACTOR:
            # Also check that this frame's confidence is lower than
            # the neighbor average — a true outlier is typically less
            # confident.
            states[i] = OUTLIER

    return states
