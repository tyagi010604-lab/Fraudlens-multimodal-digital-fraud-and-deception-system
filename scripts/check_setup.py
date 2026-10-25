"""Diagnose a FraudLens installation (Windows-friendly).

    python scripts/check_setup.py
    python scripts/check_setup.py --text "Hey! are we meeting at 5?"     # show exactly how the text model scores a message
"""
import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--text", help="explain the text model's decision for this message")
a = ap.parse_args()

print(f"Python {sys.version.split()[0]} on {platform.platform()}")
for mod in ("streamlit", "numpy", "pandas", "sklearn", "xgboost", "cv2", "librosa", "soundfile", "pytesseract", "PIL", "joblib"):
    try:
        m = __import__(mod)
        print(f"  [ok]   {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  [MISSING] {mod:9s} -> pip install -r requirements.txt  ({type(e).__name__})")

print("\nModels (trained models are loaded, never trained, by the app):")
import joblib  # noqa: E402
import sklearn  # noqa: E402
for name, path in (("text", config.TEXT_MODEL_PATH), ("video", config.VIDEO_MODEL_PATH), ("voice ID", config.VOICE_ID_PATH),
                   ("url hint", config.URL_MODEL_PATH), ("audio deepfake", config.AUDIO_MODEL_PATH)):
    if not path.exists():
        note = " (expected: no suitable labelled data)" if name == "audio deepfake" else "  -> python scripts/train_models.py --only " + \
            {"text": "text", "video": "video", "voice ID": "voice", "url hint": "url"}.get(name, "audio")
        print(f"  [not trained] {name:15s}{note}")
        continue
    try:
        b = joblib.load(path)
        info = b.get("info", {})
        v = info.get("versions", {}).get("scikit-learn")
        warn = f"  (trained with scikit-learn {v}, you have {sklearn.__version__} - retrain if you see errors)" if v and v != sklearn.__version__ else ""
        print(f"  [loads] {name:15s} model={info.get('model', info.get('scorer', '-'))} trained={str(info.get('trained_at', '-'))[:10]} "
              f"experimental={info.get('experimental')}{warn}")
    except Exception as e:
        print(f"  [LOAD ERROR] {name}: {type(e).__name__}: {e}\n               -> retrain: python scripts/train_models.py --only ...")

from src import identity_checks, text_ocr  # noqa: E402
from src import preprocess_audio as pa  # noqa: E402
from src import data_access as da  # noqa: E402
ok, msg = text_ocr.check_tesseract()
print("\nTesseract OCR:", "OK - " + msg if ok else "NOT FOUND\n" + msg)
print("ffmpeg:", "found" if pa.have_ffmpeg() else "not found (only needed for a video's own audio track / MP3)",
      "| ffprobe:", "found" if __import__("shutil").which("ffprobe") else "not found")
print("Face models:", identity_checks.face_backend_status()[1])
print("\nDatasets (searched in:", ", ".join(str(d) for d in da.search_dirs()), ")")
for k in config.DATASET_ARCHIVES:
    p = da.find_archive(k)
    print(f"  {'[found]' if p else '[missing]':10s} {k:14s} {p.name if p else config.DATASET_ARCHIVES[k][0]}")

if a.text:
    print("\n--- text model decision ---")
    r = text_ocr.analyze_text(a.text)
    m = r["model"]
    print("cleaned text :", text_ocr.clean_for_model(a.text))
    print("model label  :", m.get("label"), "| score", m.get("risk_score"), "| threshold", m.get("threshold"),
          "| vocab coverage", m.get("vocab_coverage"))
    b = text_ocr.load_text_model()
    if b:
        u = b["unknown"]
        print(f"uncertain band: {u['uncertain_low']} .. {u['uncertain_high']}   min vocab coverage: {u['min_vocab_coverage']}")
    for rsn in m.get("reasons", []):
        print("  reason:", rsn)
    print("rule cues    :", r["rules"]["level"], r["rules"]["score"], [c["category"] for c in r["rules"]["categories"]])
    print("ASSESSMENT   :", r["assessment"]["label"], "-", r["assessment"]["explanation"])
