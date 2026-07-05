# Pitch Please

A guitar solo pitch analysis toolkit: extract pitch from audio, recover a
continuous trajectory, and validate by ear with sine-wave reconstruction.

## Project Structure

```
pitch-please/
├── assets/              # MP3 files for pYIN
├── assets_wav/          # WAV files for torchcrepe
├── output/              # All generated JSON and WAV files
├── detectors/
│   ├── pyin_detector.py       # librosa pYIN detector
│   └── torchcrepe_detector.py # PyTorch CREPE detector (GPU)
├── pitch_types.py             # Shared detector interface & data types
├── main.py                    # Detection entry point
├── recovery_types.py          # RecoveredFrame data type
├── segment_detector.py        # Stage 2: musical segment splitting
├── frame_classifier.py        # Stages 3-4: VALID/UNCERTAIN/OUTLIER/SILENT
├── curve_fitter.py            # Stages 5-7: PCHIP curve fitting & gap recovery
├── trajectory_recovery.py     # Full recovery pipeline orchestrator
├── recovery_main.py           # Recovery CLI entry point
├── reconstruct.py             # Sine-wave reconstruction tool
├── server.py                  # Flask API for the frontend
├── frontend/                  # React + Plotly visualization
└── requirements.txt
```

## Setup

```bash
# Create a virtual environment (Python 3.12 required for torchcrepe)
py -3.12 -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12.6), install torch from the PyTorch index
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Running the Full Pipeline for Your Own MP3

### 1. Add your audio file

Drop an MP3 into `assets/`:

```
assets/my_solo.mp3
```

For torchcrepe, also drop a WAV version into `assets_wav/`:

```
assets_wav/my_solo.wav
```

### 2. Run pitch detection

```bash
# pYIN only (uses the MP3 from assets/)
python main.py --detector pyin --input assets/my_solo.mp3

# torchcrepe only (uses the WAV from assets_wav/, requires GPU)
python main.py --detector crepe --input assets_wav/my_solo.wav

# Both detectors on their respective default files
python main.py --all
```

This produces:
- `output/my_solo_pyin_out.json` — raw pYIN pitch frames
- `output/my_solo_pyin_reconstructed.wav` — sine-wave reconstruction
- `output/my_solo_crepe_out.json` — raw torchcrepe pitch frames
- `output/my_solo_crepe_reconstructed.wav` — sine-wave reconstruction

### 3. Run trajectory recovery

```bash
python recovery_main.py --input output/my_solo_pyin_out.json
```

This produces:
- `output/my_solo_pyin_out_recovered.json` — recovered trajectory
- `output/my_solo_pyin_out_recovered_reconstructed.wav` — smoother sine-wave

### 4. (Optional) Start the frontend

```bash
# Terminal 1 — backend API
python server.py

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

## Output Schema

### Detector output (`{stem}_{detector}_out.json`)

```json
[
    {
        "time_ms": 120,
        "frequency": 438.72,
        "confidence": 0.81,
        "voiced": true,
        "energy": 0.72
    },
    {
        "time_ms": 130,
        "frequency": null,
        "confidence": 0.18,
        "voiced": false,
        "energy": 0.61
    }
]
```

### Recovered output (`{stem}_recovered.json`)

```json
[
    {
        "time_ms": 120,
        "raw_frequency": 438.72,
        "recovered_frequency": 439.05,
        "confidence": 0.81,
        "voiced": true,
        "energy": 0.72,
        "state": "VALID"
    }
]
```

States: `VALID`, `UNCERTAIN`, `OUTLIER`, `SILENT`, `RECOVERED`.

## CLI Commands Summary

```bash
# --- Detection ---
python main.py --detector pyin --input assets/my_solo.mp3
python main.py --detector crepe --input assets_wav/my_solo.wav
python main.py --all

# --- Recovery ---
python recovery_main.py --input output/my_solo_pyin_out.json
python recovery_main.py --input output/my_solo_pyin_out.json --no-reconstruct

# --- Manual reconstruction (any pitch JSON) ---
python reconstruct.py --input output/my_solo_pyin_out.json --output output/my_solo.wav

# --- Frontend ---
python server.py
cd frontend && npm run dev
```

## Configuration

### pYIN (`detectors/pyin_detector.py`)

| Parameter | Value |
|-----------|-------|
| `fmin` | 80 Hz |
| `fmax` | 1500 Hz |
| Hop length | `sr * 10 / 1000` (~10 ms) |
| `frame_length` | 2048 (4096 for sr > 48 kHz) |

### torchcrepe (`detectors/torchcrepe_detector.py`)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz (resampled from source) |
| Hop length | 160 samples (10 ms) |
| Model | `full` |
| Confidence threshold | 0.6 |
| Device | Auto (CUDA if available, else CPU) |

### Recovery (`frame_classifier.py`, `curve_fitter.py`)

| Parameter | Value |
|-----------|-------|
| Min gap for segment split | 5 frames (50 ms) |
| Confidence threshold (VALID) | 0.5 |
| Outlier window | ±5 frames |
| Outlier deviation factor | 0.5 (50% from local median) |
| Max recoverable gap | 5 frames (50 ms) |
| Max freq ratio (physics) | 2.5x between consecutive frames |
| Curve fitting | PCHIP (shape-preserving cubic) |
