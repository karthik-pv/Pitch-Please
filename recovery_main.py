"""Pitch Trajectory Recovery Engine — CLI entry point.

Reads a detector output JSON, runs the full recovery pipeline, and
writes the recovered pitch trajectory to output.

Usage
-----
    python recovery_main.py --input output/comfortably_numb_solo_pyin_out.json
    python recovery_main.py --input output/comfortably_numb_solo_pyin_out.json --reconstruct

The output JSON is named {stem}_recovered.json and the optional
reconstructed WAV is named {stem}_recovered_reconstructed.wav.
"""

from __future__ import annotations

import argparse
import os
import sys

from trajectory_recovery import run_recovery, save_recovered, print_statistics, load_frames
from segment_detector import detect_segments


def stem_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover a continuous pitch trajectory from detector output."
    )
    parser.add_argument(
        "--input",
        default="output/comfortably_numb_solo_pyin_out.json",
        help="Path to the detector output JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the recovered JSON (default: output/{stem}_recovered.json)",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        default=True,
        help="Also produce a reconstructed sine-wave WAV from the recovered data",
    )
    parser.add_argument(
        "--no-reconstruct",
        dest="reconstruct",
        action="store_false",
        help="Skip sine-wave reconstruction",
    )
    return parser.parse_args()


def reconstruct_recovered(json_path: str, wav_path: str) -> bool:
    """Reconstruct a sine-wave WAV from the recovered JSON.

    The reconstruct.py tool expects 'frequency' and 'time_ms' fields.
    The recovered JSON uses 'recovered_frequency' instead, so we adapt
    the records before passing them to the reconstruction pipeline.
    """
    from reconstruct import reconstruct_audio, write_wav, print_validation, SAMPLE_RATE

    try:
        raw = load_frames(json_path)
    except Exception as exc:
        print(f"Error loading recovered JSON: {exc}", file=sys.stderr)
        return False

    # Adapt: use recovered_frequency as the frequency for reconstruction.
    adapted = []
    for r in raw:
        adapted.append({
            "time_ms": r["time_ms"],
            "frequency": r.get("recovered_frequency"),
        })

    try:
        audio = reconstruct_audio(adapted, sr=SAMPLE_RATE)
    except ValueError as exc:
        print(f"Error during reconstruction: {exc}", file=sys.stderr)
        return False

    try:
        write_wav(audio, wav_path, sr=SAMPLE_RATE)
    except RuntimeError as exc:
        print(f"Error writing WAV: {exc}", file=sys.stderr)
        return False

    print_validation(adapted, audio, SAMPLE_RATE, wav_path)
    return True


def main():
    args = parse_args()
    input_path = args.input
    stem = stem_from_path(input_path)

    output_path = args.output or f"output/{stem}_recovered.json"
    wav_path = f"output/{stem}_recovered_reconstructed.wav"

    # Run the recovery pipeline.
    try:
        # Re-extract segment count for statistics.
        raw_frames = load_frames(input_path)
        frequencies = [f.get("frequency") for f in raw_frames]
        voiced_flags = [f.get("voiced", False) for f in raw_frames]
        energies = [f.get("energy", 0.0) for f in raw_frames]
        segments = detect_segments(frequencies, voiced_flags, energies)

        recovered = run_recovery(input_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error during recovery: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        save_recovered(recovered, output_path)
    except RuntimeError as exc:
        print(f"Error saving recovered output: {exc}", file=sys.stderr)
        sys.exit(1)

    print_statistics(recovered, len(segments))
    print(f"Saved {len(recovered)} records to {output_path}")

    if args.reconstruct:
        reconstruct_recovered(output_path, wav_path)


if __name__ == "__main__":
    main()
