"""Module 1: video deepfake detection (EXPERIMENTAL baseline).

Pipeline: validate -> sample frames -> classic forensic features -> scikit-learn/XGBoost classifier
trained by scripts/train_models.py on the small SDFVD set. Output is a clip-level score. The method
cannot localise manipulated frames and does not analyse lip-sync; "unstable frames" are only
feature-jump indicators.
"""
import joblib

import config
from src import preprocess_video as pv
from src import unknown_detection as ud

_B = {"obj": None, "err": None, "loaded": False}


def load_model(force=False):
    if _B["loaded"] and not force:
        return _B["obj"]
    _B["loaded"] = True
    try:
        if config.VIDEO_MODEL_PATH.exists():
            b = joblib.load(config.VIDEO_MODEL_PATH)
            if list(b["feature_names"]) != pv.feature_names():
                raise ValueError("feature set changed - retrain")
            _B["obj"] = b
        else:
            _B["err"] = "Video model not trained yet. Run: python scripts/train_models.py --only video"
    except Exception as e:
        _B["obj"], _B["err"] = None, f"Video model could not be loaded ({type(e).__name__}: {e}). Retrain it."
    return _B["obj"]


def model_error():
    load_model()
    return _B["err"]


def analyse_video(path):
    """Return a result dict for the video evidence. Never raises for bad media (returns status 'error')."""
    try:
        feats = pv.extract_video_features(path)
    except pv.VideoError as e:
        return {"status": "error", "label": "Not assessed", "reason": str(e)}
    meta = feats["meta"]
    out = {"analysed_frames": feats["n_frames_used"], "faces_found_frames": feats["faces_found"], "video_meta": meta,
           "temporal_instability_frames": pv.temporal_outliers(feats["per_frame"])[:10],
           "timestamp_note": ("Listed timestamps are frames whose image statistics jump away from the clip median "
                              "(blur, compression or detection changes). They are NOT locations of manipulation."),
           "per_frame_summary": [{"t": r["timestamp_s"], "face": bool(r["face_found"])} for r in feats["per_frame"]]}
    b = load_model()
    if b is None:
        out.update({"status": "unavailable", "label": "Not assessed", "risk_score": None, "reason": model_error(),
                    "note": "Frame extraction and features worked, but no trained classifier is available."})
        return out
    info = b["info"]
    p = float(b["model"].predict_proba(feats["vector"].reshape(1, -1))[0, 1])
    reasons = ud.video_scope_issues(meta, feats["faces_found"], feats["n_frames_used"])
    ood = b["ood"].check(feats["vector"])
    if ood["out_of_distribution"]:
        reasons.append(f"Feature distance {ood['distance']} exceeds the validation-calibrated limit {ood['threshold']}: "
                       "the clip looks unlike the training clips.")
    lo, hi = b["unknown"]["uncertain_low"], b["unknown"]["uncertain_high"]
    if lo <= p <= hi:
        reasons.append(f"Score {p:.2f} is inside the uncertain band [{lo:.2f}, {hi:.2f}].")
    thr = b["threshold"]
    if reasons:
        label = "Unknown / Review"
    elif p >= thr:
        label = "Possible manipulation (experimental)"
    else:
        label = "No manipulation pattern detected (experimental)"
    out.update({"status": "experimental", "label": label, "risk_score": round(p, 4), "threshold": round(thr, 4),
                "unknown": bool(reasons), "reasons": reasons, "ood": ood,
                "calibrated": False, "score_note": "Raw classifier score, not a calibrated probability.",
                "model_info": {k: info[k] for k in ("model", "dataset", "n_train", "n_test", "experimental_reason", "trained_at")},
                "limitations": info["limitations"]})
    return out
