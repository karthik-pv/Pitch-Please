"""Data types for the Pitch Trajectory Recovery Engine.

Defines the RecoveredFrame dataclass that extends the detector's raw
output with recovery metadata: the recovered frequency and a state
classification.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


# Frame state classification.
VALID = "VALID"
UNCERTAIN = "UNCERTAIN"
OUTLIER = "OUTLIER"
SILENT = "SILENT"
RECOVERED = "RECOVERED"


@dataclass
class RecoveredFrame:
    """A single frame enriched with recovery information.

    Attributes
    ----------
    time_ms : int
        Timestamp of the frame in milliseconds (from detector).
    raw_frequency : float | None
        Original detector frequency, never modified.
    recovered_frequency : float | None
        Estimated true frequency after trajectory recovery.
        May differ from raw_frequency for OUTLIER and RECOVERED frames.
        None for SILENT frames.
    confidence : float
        Original detector confidence (unchanged).
    voiced : bool
        Original detector voiced flag (unchanged).
    energy : float
        Original RMS energy (unchanged).
    state : str
        Classification: VALID, UNCERTAIN, OUTLIER, SILENT, or RECOVERED.
    """

    time_ms: int
    raw_frequency: Optional[float]
    recovered_frequency: Optional[float]
    confidence: float
    voiced: bool
    energy: float
    state: str

    def to_dict(self) -> dict:
        return asdict(self)
