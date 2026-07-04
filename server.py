"""Pitch Please - Flask API server.

Serves the pitch JSON files stored in the output/ directory to the frontend.
Intended for local development only.
"""

import json
import os

from flask import Flask, jsonify, send_from_directory

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

app = Flask(__name__, static_folder=None)


@app.get("/api/files")
def list_files():
    """Return the list of available pitch JSON files in output/."""
    if not os.path.isdir(OUTPUT_DIR):
        return jsonify([])

    files = sorted(
        f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")
    )
    return jsonify(files)


@app.get("/api/pitch/<path:filename>")
def get_pitch(filename):
    """Return the contents of a single pitch JSON file."""
    if not filename.endswith(".json"):
        return jsonify({"error": "Only .json files are supported"}), 400

    full_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(full_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        with open(full_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return jsonify({"error": f"Failed to read {filename}: {exc}"}), 500

    return jsonify(data)


@app.after_request
def add_cors_headers(response):
    """Allow the Vite dev server to call this API during development."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Serving pitch data from: {OUTPUT_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=True)
