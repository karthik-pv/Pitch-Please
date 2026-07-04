"""Pitch Please — unified pitch detection entry point.

Runs either the pYIN or torchcrepe detector on an audio file, saves the
pitch timeline JSON, and optionally reconstructs a sine-wave WAV for
auditory validation.

Output naming convention
------------------------
    JSON:   output/{stem}_{detector}_out.json
    WAV:    output/{stem}_{detector}_reconstructed.wav

where {stem} is the input filename without extension and {detector}
is 'pyin' or 'crepe'.

Usage
-----
    # Run a single detector
    python main.py --detector pyin
    python main.py --detector crepe

    # Run both detectors and reconstruct both
    python main.py --all

    # Custom input file
    python main.py --detector pyin --input assets/solo.mp3
"""

from __future__ import annotations

import argparse
import os
import sys

from pitch_types import PitchDetector, DetectionResult
from detectors import PyinDetector, TorchCrepeDetector


DETECTORS: dict[str, type[PitchDetector]] = {
    "pyin": PyinDetector,
    "crepe": TorchCrepeDetector,
}

# Default input files per detector.
DEFAULT_INPUTS: dict[str, str] = {
    "pyin": "assets/comfortably_numb_solo.mp3",
    "crepe": "assets_wav/comfortably_numb_solo.wav",
}


def stem_from_path(path: str) -> str:
    """Return the filename without its extension."""
    return os.path.splitext(os.path.basename(path))[0]


def output_paths(input_path: str, detector_key: str) -> tuple[str, str]:
    """Build the output JSON and reconstructed WAV paths from naming convention."""
    stem = stem_from_path(input_path)
    json_path = f"output/{stem}_{detector_key}_out.json"
    wav_path = f"output/{stem}_{detector_key}_reconstructed.wav"
    return json_path, wav_path


def run_detector(detector_key: str, input_path: str, reconstruct: bool) -> bool:
    """Run a single detector on the given input. Returns True on success."""
    detector_cls = DETECTORS[detector_key]
    detector = detector_cls()

    json_path, wav_path = output_paths(input_path, detector_key)

    try:
        result: DetectionResult = detector.detect(input_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error during {detector_key} detection: {exc}", file=sys.stderr)
        return False

    if not result.frames:
        print(f"Error: {detector_key} produced no pitch frames.", file=sys.stderr)
        return False

    try:
        detector.save_output(result.frames, json_path)
    except RuntimeError as exc:
        print(f"Error saving {detector_key} output: {exc}", file=sys.stderr)
        return False

    detector.print_validation(result, json_path)
    print(f"Saved {len(result.frames)} records to {json_path}")

    if reconstruct:
        if not reconstruct_wav(json_path, wav_path):
            return False

    return True


def reconstruct_wav(json_path: str, wav_path: str) -> bool:
    """Run the reconstruction tool on a pitch JSON to produce a sine-wave WAV."""
    from reconstruct import load_pitch_json, reconstruct_audio, write_wav, \
        print_validation as print_recon_validation, SAMPLE_RATE

    try:
        records = load_pitch_json(json_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error loading pitch JSON for reconstruction: {exc}", file=sys.stderr)
        return False

    try:
        audio = reconstruct_audio(records, sr=SAMPLE_RATE)
    except ValueError as exc:
        print(f"Error during reconstruction: {exc}", file=sys.stderr)
        return False

    try:
        write_wav(audio, wav_path, sr=SAMPLE_RATE)
    except RuntimeError as exc:
        print(f"Error writing reconstructed WAV: {exc}", file=sys.stderr)
        return False

    print_recon_validation(records, audio, SAMPLE_RATE, wav_path)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pitch detector on an audio file and save the pitch timeline."
    )
    parser.add_argument(
        "--detector",
        choices=list(DETECTORS.keys()),
        default=None,
        help="Pitch detector to use (pyin or crepe)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the input audio file (default: per-detector default)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all detectors on their default inputs and reconstruct both",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        default=True,
        help="Also produce a reconstructed sine-wave WAV (default: true)",
    )
    parser.add_argument(
        "--no-reconstruct",
        dest="reconstruct",
        action="store_false",
        help="Skip sine-wave reconstruction",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        success = True
        for key in DETECTORS:
            inp = DEFAULT_INPUTS[key]
            print(f"\n{'='*60}")
            print(f"  Running {key} on {inp}")
            print(f"{'='*60}\n")
            ok = run_detector(key, inp, reconstruct=args.reconstruct)
            if not ok:
                success = False
        sys.exit(0 if success else 1)

    if args.detector is None:
        parser = argparse.ArgumentParser(description="Pitch Please detector")
        parser.error("either --detector or --all is required")

    detector_key = args.detector
    input_path = args.input or DEFAULT_INPUTS[detector_key]

    ok = run_detector(detector_key, input_path, reconstruct=args.reconstruct)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
