# Pitch Please - Milestone 1: pYIN Pitch Extraction

A standalone Python script that extracts a continuous pitch timeline from an
isolated guitar solo MP3 using librosa's pYIN algorithm.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python librosa_detector.py
```

By default it reads `assets/comfortably_numb_solo.mp3` and writes
`output/pitch.json`. Paths can be overridden:

```bash
python librosa_detector.py --input assets/solo.mp3 --output output/pitch.json
```

## Output

`output/pitch.json` is a JSON array of records, one per ~10 ms frame:

```json
[
    { "time_ms": 0,   "frequency": 329.63, "confidence": 0.99 },
    { "time_ms": 10,  "frequency": 329.82, "confidence": 0.98 },
    { "time_ms": 250, "frequency": null,   "confidence": 0.07 }
]
```

Unvoiced frames emit `frequency: null` (no interpolation).

## Configuration

Guitar-tuned pYIN parameters (in `librosa_detector.py`):

| Parameter | Value |
|-----------|-------|
| `fmin` | 80 Hz |
| `fmax` | 1500 Hz |
| Hop length | `sr * 10 / 1000` (~10 ms) |
| `frame_length` | 2048 (4096 for sr > 48 kHz) |

## Validation

After running, a summary is printed: sample rate, duration, frame count,
voiced/unvoiced counts, and average confidence.
