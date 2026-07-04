import os
import subprocess
import sys

BASE_DIR = os.path.dirname(__file__)

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
ASSETS_WAV_DIR = os.path.join(BASE_DIR, "assets_wav")


def main():
    os.makedirs(ASSETS_WAV_DIR, exist_ok=True)

    if not os.path.isdir(ASSETS_DIR):
        print(f"Assets directory not found: {ASSETS_DIR}")
        sys.exit(1)

    for filename in os.listdir(ASSETS_DIR):
        filepath = os.path.join(ASSETS_DIR, filename)

        if not os.path.isfile(filepath):
            continue

        if not filename.lower().endswith(".mp3"):
            continue

        base = os.path.splitext(filename)[0]
        wav_path = os.path.join(ASSETS_WAV_DIR, base + ".wav")

        print(f"Converting {filename} -> assets_wav/{base}.wav")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                filepath,
                wav_path,
            ],
            check=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()