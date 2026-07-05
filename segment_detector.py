"""Stage 2 — Segment Detection.

Splits the recording into continuous musical segments separated by
silence or unvoiced regions.  A new segment begins when the signal
has been continuously silent/unvoiced for a sustained period
(~50–100 ms).  Single bad frames do not cause a split.

Each segment is represented as a (start_index, end_index) pair
(inclusive) referring to indices in the original frame list.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


# Minimum consecutive silent/unvoiced frames to trigger a segment break.
# At 10 ms per frame, 5 frames = 50 ms, 10 frames = 100 ms.
# We use 5 frames (50 ms) as the minimum gap to split on.
MIN_GAP_FRAMES = 5

# Energy threshold for silence detection.
# We use an adaptive threshold: a fraction of the median energy.
# Frames below this are considered "low energy".
SILENCE_ENERGY_FACTOR = 0.15


def compute_silence_threshold(energies: np.ndarray) -> float:
    """Compute an adaptive silence threshold from the energy distribution.

    Uses a fraction of the median energy so the threshold adapts to
    recordings with different overall levels.
    """
    if energies.size == 0:
        return 0.0
    median_energy = float(np.median(energies[energies > 0])) if np.any(energies > 0) else 0.0
    return median_energy * SILENCE_ENERGY_FACTOR


def is_silent_frame(freq, voiced, energy, threshold) -> bool:
    """Check if a frame qualifies as silent/unvoiced for segmentation."""
    if energy < threshold:
        return True
    if not voiced:
        return True
    if freq is None:
        return True
    return False


def detect_segments(
    frequencies: List[Optional[float]],
    voiced_flags: List[bool],
    energies: List[float],
) -> List[Tuple[int, int]]:
    """Split the recording into continuous musical segments.

    Returns a list of (start_idx, end_idx) inclusive index pairs.
    """
    n = len(frequencies)
    if n == 0:
        return []

    energy_arr = np.array(energies, dtype=np.float64)
    threshold = compute_silence_threshold(energy_arr)

    # Mark each frame as silent or active.
    silent = []
    for i in range(n):
        s = is_silent_frame(frequencies[i], voiced_flags[i], energies[i], threshold)
        silent.append(s)

    # Find segment boundaries: a break occurs after a run of MIN_GAP_FRAMES
    # or more consecutive silent frames.
    segments: List[Tuple[int, int]] = []
    seg_start = 0
    silent_run = 0

    for i in range(n):
        if silent[i]:
            silent_run += 1
        else:
            if silent_run >= MIN_GAP_FRAMES:
                # Close the current segment before the silent run.
                seg_end = i - silent_run
                if seg_end >= seg_start:
                    segments.append((seg_start, seg_end))
                seg_start = i
            silent_run = 0

    # Close the final segment.
    if silent_run >= MIN_GAP_FRAMES:
        seg_end = n - 1 - silent_run
        if seg_end >= seg_start:
            segments.append((seg_start, seg_end))
    else:
        if n - 1 >= seg_start:
            segments.append((seg_start, n - 1))

    return segments
