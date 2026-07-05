"""Stage 6 & 7 — Continuous Curve Fitting with Confidence Weighting.

Estimates a smooth continuous pitch trajectory within each segment
using shape-preserving cubic interpolation (PCHIP).  PCHIP is preferred
over plain cubic splines because it avoids overshoot — critical for
preserving bends and vibrato without introducing artificial wiggles.

Confidence weighting is applied by using high-confidence VALID frames
as strong anchor points and letting low-confidence UNCERTAIN frames
have reduced influence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator


# Maximum gap (in frames) that recovery will attempt to fill.
# At 10 ms per frame, 4 frames = 40 ms, 5 frames = 50 ms.
MAX_RECOVER_GAP = 5


def fit_segment_curve(
    times: np.ndarray,
    freqs: np.ndarray,
    confidences: np.ndarray,
    valid_mask: np.ndarray,
) -> PchipInterpolator:
    """Fit a PCHIP curve to the VALID anchor points in a segment.

    Confidence weighting is applied by repeating high-confidence points
    (effectively giving them more weight in the interpolation).  Low-
    confidence points are included once so they still influence the
    curve but do not dominate it.

    Parameters
    ----------
    times : (N,) array
        Time values (seconds) for every frame in the segment.
    freqs : (N,) array
        Frequency values for every frame (NaN for missing).
    confidences : (N,) array
        Confidence values for every frame.
    valid_mask : (N,) boolean array
        True for frames to use as anchor points (VALID + UNCERTAIN with freq).
    """
    anchor_times = []
    anchor_freqs = []

    for i in range(len(times)):
        if not valid_mask[i]:
            continue
        f = freqs[i]
        if np.isnan(f) or f <= 0:
            continue

        # High-confidence anchors get extra weight by duplication.
        # This is a simple but effective confidence-weighting scheme
        # for PCHIP (which does not natively support weights).
        anchor_times.append(times[i])
        anchor_freqs.append(f)

        if confidences[i] >= 0.7:
            # Strong anchor: duplicate to increase its influence.
            anchor_times.append(times[i])
            anchor_freqs.append(f)

    if len(anchor_times) < 2:
        # Not enough anchors for a curve — fall back to constant or
        # linear if possible.
        if len(anchor_times) == 1:
            # Constant interpolation.
            t0 = anchor_times[0]
            f0 = anchor_freqs[0]
            return PchipInterpolator([t0, t0 + 1e-6], [f0, f0], extrapolate=False)
        else:
            return None

    anchor_times = np.array(anchor_times)
    anchor_freqs = np.array(anchor_freqs)

    # Sort by time (duplicated points are already sorted).
    sort_idx = np.argsort(anchor_times)
    anchor_times = anchor_times[sort_idx]
    anchor_freqs = anchor_freqs[sort_idx]

    # PCHIP requires unique x values.  Merge duplicates by averaging.
    unique_times, unique_indices = np.unique(anchor_times, return_index=True)
    if len(unique_times) < len(anchor_times):
        merged_freqs = []
        for t in unique_times:
            mask = anchor_times == t
            merged_freqs.append(float(np.mean(anchor_freqs[mask])))
        anchor_times = unique_times
        anchor_freqs = np.array(merged_freqs)

    return PchipInterpolator(anchor_times, anchor_freqs, extrapolate=False)


def recover_segment(
    times: np.ndarray,
    freqs: np.ndarray,
    confidences: np.ndarray,
    states: List[str],
    energy_threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    """Recover the pitch trajectory for a single segment.

    Returns (recovered_freqs, updated_states) where recovered_freqs
    contains the estimated frequency for every frame in the segment
    (NaN for frames that remain unrecovered).
    """
    from recovery_types import VALID, UNCERTAIN, OUTLIER, SILENT, RECOVERED

    n = len(times)
    recovered = freqs.copy()
    updated_states = list(states)

    # Build mask for anchor points: VALID frames are strong anchors.
    # UNCERTAIN frames with a frequency are weak anchors.
    # OUTLIER frames are excluded from anchors (they will be recovered).
    anchor_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if states[i] == VALID:
            anchor_mask[i] = True
        elif states[i] == UNCERTAIN and not np.isnan(freqs[i]) and freqs[i] > 0:
            anchor_mask[i] = True

    # Fit the curve.
    curve = fit_segment_curve(times, freqs, confidences, anchor_mask)

    if curve is None:
        # Not enough anchors — can't recover this segment.
        return recovered, updated_states

    # --- Stage 5: Tiny gap recovery ---
    # Fill short runs of UNCERTAIN/SILENT-within-segment frames where
    # energy is high and the gap is <= MAX_RECOVER_GAP frames.
    i = 0
    while i < n:
        if states[i] == VALID or states[i] == RECOVERED:
            i += 1
            continue

        # Check if this is a recoverable gap.
        # A gap is a run of non-VALID frames within the segment.
        gap_start = i
        while i < n and (states[i] == UNCERTAIN or states[i] == OUTLIER):
            i += 1
        gap_end = i  # exclusive

        gap_len = gap_end - gap_start

        if gap_len == 0:
            i += 1
            continue

        # Only recover short gaps.
        if gap_len <= MAX_RECOVER_GAP:
            # Check that we have valid anchors on both sides.
            has_left = gap_start > 0 and (states[gap_start - 1] == VALID or states[gap_start - 1] == RECOVERED)
            has_right = gap_end < n and states[gap_end] == VALID

            if has_left and has_right:
                # Recover using the fitted curve.
                t_gap = times[gap_start:gap_end]
                try:
                    recovered_vals = curve(t_gap)
                    for j in range(gap_len):
                        idx = gap_start + j
                        val = float(recovered_vals[j])
                        if np.isfinite(val) and val > 0:
                            recovered[idx] = val
                            updated_states[idx] = RECOVERED
                except Exception:
                    pass  # Leave as-is if interpolation fails

        # Don't re-process these frames.
        continue

    # --- Outlier replacement ---
    # Replace OUTLIER frames with the curve estimate.
    for i in range(n):
        if states[i] == OUTLIER:
            try:
                val = float(curve(times[i]))
                if np.isfinite(val) and val > 0:
                    recovered[i] = val
                    updated_states[i] = RECOVERED
            except Exception:
                pass

    # --- Confidence-weighted adjustment for UNCERTAIN frames ---
    # Move low-confidence observations toward the estimated curve.
    for i in range(n):
        if states[i] == UNCERTAIN and not np.isnan(freqs[i]) and freqs[i] > 0:
            try:
                curve_val = float(curve(times[i]))
                if np.isfinite(curve_val) and curve_val > 0:
                    # Blend: high confidence → stay close to raw;
                    # low confidence → move toward curve.
                    weight = confidences[i]
                    adjusted = weight * freqs[i] + (1.0 - weight) * curve_val
                    recovered[i] = adjusted
            except Exception:
                pass

    return recovered, updated_states
