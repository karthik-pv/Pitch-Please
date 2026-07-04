"""Pitch Please - Pitch reconstruction utility.

Reads the extracted pitch timeline (output/pitch.json) and synthesises a
continuous sine-wave audio file that follows the detected frequencies.

This is a validation/debugging tool: the output sounds like a theremin or
laboratory oscillator, not a guitar, but it preserves the melodic contour
(bends, vibrato, slides) so the accuracy of the pitch extraction can be
verified by ear.

Key design points:
  * A single phase accumulator runs across the entire recording so the
    oscillator never resets between frames (no clicks).
  * Each frame produces exactly the right number of samples for its
    duration — no timing drift.
  * Unvoiced frames (frequency === null) produce silence but the phase
    accumulator keeps advancing so resumption is click-free.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100


def load_pitch_json(path):
    """Load the pitch timeline JSON file and return the list of records."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pitch JSON not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Pitch JSON is empty or not a list of records.")

    return data


def _frame_sample_count(time_ms, prev_time_ms, sr):
    """Compute how many samples belong to this frame.

    Uses the difference between the current and previous timestamps so
    that the total sample count matches the recording duration exactly
    with no drift.
    """
    duration_ms = time_ms - prev_time_ms
    return max(0, int(round(duration_ms * sr / 1000.0)))


def generate_sine(freq, n_samples, phase):
    """Generate n_samples of a sine wave at *freq* starting from *phase*.

    Returns (samples, new_phase).  The phase accumulator is advanced
    continuously so consecutive calls produce a seamless waveform.
    """
    if freq is None or n_samples == 0:
        # Silence: still advance phase by the equivalent amount so the
        # oscillator resumes smoothly when voiced audio returns.
        if freq is not None:
            phase += 2.0 * np.pi * freq * n_samples / SAMPLE_RATE
            phase = phase % (2.0 * np.pi)
        return np.zeros(n_samples, dtype=np.float32), phase

    # Angular frequencies per sample.
    omega = 2.0 * np.pi * freq / SAMPLE_RATE
    t = np.arange(n_samples, dtype=np.float64)
    samples = np.sin(phase + omega * t).astype(np.float32)

    # Advance the phase accumulator to the end of this frame.
    phase = (phase + omega * n_samples) % (2.0 * np.pi)
    return samples, phase


def reconstruct_audio(records, sr=SAMPLE_RATE):
    """Walk every frame and build the full audio buffer.

    A single phase accumulator is threaded through all frames so the
    oscillator runs continuously — only its frequency changes per frame.
    """
    total_samples = 0
    audio_chunks = []
    phase = 0.0
    prev_time_ms = 0

    for record in records:
        time_ms = record["time_ms"]
        freq = record.get("frequency")
        n = _frame_sample_count(time_ms, prev_time_ms, sr)

        if n > 0:
            samples, phase = generate_sine(freq, n, phase)
            audio_chunks.append(samples)
            total_samples += n

        prev_time_ms = time_ms

    if total_samples == 0:
        raise ValueError("No audio samples were generated (all frames zero-length?).")

    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
    return audio


def write_wav(audio, path, sr=SAMPLE_RATE):
    """Write the audio buffer to a 16-bit PCM WAV file."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Normalise to avoid clipping, then convert to int16.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio_int16 = (audio * 32767.0).astype(np.int16)

    try:
        sf.write(path, audio_int16, sr, subtype="PCM_16")
    except OSError as exc:
        raise RuntimeError(f"Failed to write WAV file: {path}") from exc


def print_validation(records, audio, sr, output_path):
    """Print a short summary for debugging."""
    total_ms = records[-1]["time_ms"] if records else 0
    duration_s = len(audio) / sr
    voiced = sum(1 for r in records if r.get("frequency") is not None)

    print("---- Reconstruction summary ----")
    print(f"Total frames         : {len(records)}")
    print(f"Voiced frames        : {voiced}")
    print(f"Unvoiced frames      : {len(records) - voiced}")
    print(f"Last timestamp       : {total_ms} ms ({total_ms / 1000:.3f} s)")
    print(f"Sample rate          : {sr} Hz")
    print(f"Samples generated    : {len(audio)}")
    print(f"Audio duration       : {duration_s:.3f} s")
    print(f"Output file          : {output_path}")
    print("--------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct a sine-wave audio file from a pitch timeline JSON."
    )
    parser.add_argument(
        "--input",
        default="output/pitch.json",
        help="Path to the pitch JSON (default: output/pitch.json)",
    )
    parser.add_argument(
        "--output",
        default="output/reconstructed.wav",
        help="Path to the output WAV (default: output/reconstructed.wav)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        records = load_pitch_json(args.input)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error loading pitch JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        audio = reconstruct_audio(records, sr=SAMPLE_RATE)
    except ValueError as exc:
        print(f"Error during reconstruction: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        write_wav(audio, args.output, sr=SAMPLE_RATE)
    except RuntimeError as exc:
        print(f"Error writing WAV: {exc}", file=sys.stderr)
        sys.exit(1)

    print_validation(records, audio, SAMPLE_RATE, args.output)


if __name__ == "__main__":
    main()
