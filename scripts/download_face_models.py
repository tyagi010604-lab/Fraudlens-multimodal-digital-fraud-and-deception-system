"""One-time, OPTIONAL download of the two small OpenCV Zoo ONNX models used for face verification.

    python scripts/download_face_models.py

Files (~40 MB total) are saved to models/face/. Nothing else is downloaded. If you are offline, download the
files manually from https://github.com/opencv/opencv_zoo (models/face_detection_yunet and
models/face_recognition_sface) and place them in models/face/ with the names below.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = {config.FACE_DETECTOR_ONNX: f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
         config.FACE_RECOGNIZER_ONNX: f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx"}

config.FACE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
for dest, url in FILES.items():
    if dest.exists() and dest.stat().st_size > 100_000:
        print("already present:", dest.name)
        continue
    print("downloading", dest.name, "...")
    try:
        urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size < 100_000:       # an LFS pointer/HTML error page instead of the model
            dest.unlink()
            raise RuntimeError("downloaded file is too small to be an ONNX model")
        print("  ok", round(dest.stat().st_size / 1e6, 1), "MB")
    except Exception as e:
        print("  FAILED:", e, "\n  Download it manually (see this script's docstring).")
