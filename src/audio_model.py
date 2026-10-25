"""Module 2: AI-generated / cloned voice detection.

The full pipeline (load -> resample -> features -> classifier) is implemented. A classifier is only
used if one has been trained on suitable labelled real-vs-synthetic speech. With the datasets
supplied for this project none qualified, so by default this module reports:
    "Experimental - suitable labelled training data required."
and returns NO score (no fabricated predictions).
"""
import json

import joblib

import config
from src import preprocess_audio as pa
from src import unknown_detection as ud

UNTRAINED_MSG = "Experimental — suitable labelled training data required."
_B = {"obj": None, "err": None, "loaded": False}


def load_model(force=False):
    if _B["loaded"] and not force:
        return _B["obj"]
    _B["loaded"] = True
    try:
        _B["obj"] = joblib.load(config.AUDIO_MODEL_PATH) if config.AUDIO_MODEL_PATH.exists() else None
    except Exception as e:
        _B["obj"], _B["err"] = None, f"Audio model could not be loaded ({type(e).__name__}); retrain it."
    return _B["obj"]


def training_status():
    if config.AUDIO_STATUS_PATH.exists():
        try:
            return json.loads(config.AUDIO_STATUS_PATH.read_text())
        except Exception:
            pass
    return {"status": "untrained", "message": UNTRAINED_MSG}


def analyse_audio(path, source="uploaded file"):
    try:
        y, sr, info = pa.load_audio(path)
    except pa.AudioError as e:
        return {"status": "error", "label": "Not assessed", "reason": str(e)}
    out = {"audio_info": info, "audio_summary": pa.audio_summary(y, sr), "source": source,
           "_mel": pa.mel_spectrogram(y, sr).astype("float16")}
    issues = ud.audio_scope_issues(info)
    b = load_model()
    if b is None:
        out.update({"status": "experimental_untrained", "label": "Not assessed", "risk_score": None,
                    "reason": UNTRAINED_MSG, "detail": training_status().get("datasets_checked"),
                    "scope_issues": issues,
                    "note": "The pipeline ran (loading, resampling, MFCC/Mel features) but no classifier exists, so no "
                            "authenticity prediction is made. Silence in this field is not evidence the audio is genuine."})
        return out
    feats = pa.extract_features(y, sr).reshape(1, -1)
    p = float(b["model"].predict_proba(feats)[0, 1])
    thr = b["threshold"]
    label = "Unknown / Review" if issues else ("Possible synthetic/cloned speech" if p >= thr else "No synthetic pattern detected")
    out.update({"status": "experimental" if b["info"].get("experimental") else "ready", "label": label, "risk_score": round(p, 4),
                "threshold": round(thr, 4), "unknown": bool(issues), "reasons": issues, "calibrated": False,
                "model_info": b["info"]})
    return out
