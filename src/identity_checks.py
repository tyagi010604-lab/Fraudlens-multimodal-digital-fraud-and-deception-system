"""Modules 3 & 4: reference-based identity checks.

IMPORTANT
  * A face/voice "Match" never proves that media is authentic or that a person is who they claim to be.
  * Identity verification is NOT deepfake detection (a cloned voice may match; a real voice may not).
  * Both checks are skipped ("Not assessed") unless the user supplies a reference sample.

Face: OpenCV YuNet detector + SFace embeddings (ONNX, ~40 MB, fetched once by
      scripts/download_face_models.py). Without those files the result is "Unavailable" (nothing is faked).
Voice: MFCC statistics + scorer calibrated on speaker-disjoint DeepfakeTIMIT data (EXPERIMENTAL).
"""
import joblib
import numpy as np

import config
from src import preprocess_audio as pa
from src import preprocess_video as pv

FACE_DISCLAIMER = ("A face match does not establish that the media is authentic or prove a person's identity; "
                   "a mismatch can also result from pose, lighting, age or image quality.")
VOICE_DISCLAIMER = ("Voice comparison is experimental and is NOT synthetic-speech detection. Neither a match nor a "
                    "mismatch proves who is speaking.")


# ------------------------------------------------------------------------------- FACE ------
_FACE = {"det": None, "rec": None, "tried": False, "err": None}


def face_backend_status():
    """(available, message)."""
    if not (config.FACE_DETECTOR_ONNX.exists() and config.FACE_RECOGNIZER_ONNX.exists()):
        return False, ("Face embedding models are not installed. Run `python scripts/download_face_models.py` once "
                       f"(downloads ~40 MB of OpenCV Zoo ONNX files into {config.FACE_MODEL_DIR}). "
                       "Until then face verification returns 'Unavailable'.")
    if not (hasattr(__import__("cv2"), "FaceDetectorYN") and hasattr(__import__("cv2"), "FaceRecognizerSF")):
        return False, "Your OpenCV build lacks FaceDetectorYN/FaceRecognizerSF (need opencv-python >= 4.5.4)."
    return True, "OpenCV YuNet + SFace available."


def _load_face():
    import cv2
    if not _FACE["tried"]:
        _FACE["tried"] = True
        ok, msg = face_backend_status()
        if not ok:
            _FACE["err"] = msg
        else:
            try:
                _FACE["det"] = cv2.FaceDetectorYN.create(str(config.FACE_DETECTOR_ONNX), "", (320, 320), 0.8, 0.3, 5000)
                _FACE["rec"] = cv2.FaceRecognizerSF.create(str(config.FACE_RECOGNIZER_ONNX), "")
            except Exception as e:
                _FACE["err"] = f"Face models failed to load ({type(e).__name__}: {e})."
    return _FACE["det"], _FACE["rec"], _FACE["err"]


def _embed(img_bgr):
    """Embedding of the largest detected face or (None, n_faces)."""
    det, rec, _ = _load_face()
    h, w = img_bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None, 0
    best = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
    feat = rec.feature(rec.alignCrop(img_bgr, best))
    return feat, len(faces)


def _read_image(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("Image could not be read.")
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def verify_face(reference_image, probe_path, probe_is_video=False):
    """Compare the face in `reference_image` with faces in `probe_path` (image or video)."""
    import cv2
    base = {"module": "face_identity", "disclaimer": FACE_DISCLAIMER}
    det, rec, err = _load_face()
    if det is None:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None, "reason": err}
    try:
        ref = _read_image(reference_image)
    except Exception as e:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None, "reason": f"Reference: {e}"}
    f_ref, n_ref = _embed(ref)
    if f_ref is None:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None,
                "reason": "No face was detected in the reference image."}
    notes = [f"{n_ref} faces in the reference image; the largest was used."] if n_ref > 1 else []
    try:
        if probe_is_video:
            frames = [f for _, _, f in pv.sample_frames(probe_path, config.FACE_VIDEO_FRAMES)]
        else:
            frames = [_read_image(probe_path)]
    except Exception as e:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None, "reason": f"Probe: {e}"}
    sims = []
    for fr in frames:
        f, _ = _embed(fr)
        if f is not None:
            sims.append(float(rec.match(f_ref, f, cv2.FaceRecognizerSF_FR_COSINE)))
    if not sims:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None,
                "reason": "No face was detected in the image/video being checked."}
    sim = float(np.median(sims))
    thr = config.FACE_COSINE_THRESHOLD
    return {**base, "status": "ready", "label": "Match" if sim >= thr else "Mismatch", "similarity": round(sim, 4),
            "similarity_metric": "SFace cosine similarity", "threshold": thr,
            "threshold_note": "OpenCV-documented SFace threshold; not re-calibrated on your data.",
            "frames_with_face": len(sims), "frames_checked": len(frames), "notes": notes}


# ------------------------------------------------------------------------------- VOICE ------
_V = {"obj": None, "loaded": False, "err": None}


def load_voice_model():
    if not _V["loaded"]:
        _V["loaded"] = True
        try:
            _V["obj"] = joblib.load(config.VOICE_ID_PATH) if config.VOICE_ID_PATH.exists() else None
            if _V["obj"] is None:
                _V["err"] = "Voice verifier not calibrated yet. Run: python scripts/train_models.py --only voice"
        except Exception as e:
            _V["err"] = f"Voice verifier could not be loaded ({type(e).__name__}); retrain it."
    return _V["obj"]


def _voice_embed(b, feat):
    z = feat.reshape(1, -1)
    if b["kind"] == "raw_cosine":
        return z[0]
    z = b["scaler"].transform(z)
    return z[0] if b["kind"] == "standardised_cosine" else b["lda"].transform(z)[0]


def compare_voices(reference_path, probe_path):
    base = {"module": "voice_identity", "disclaimer": VOICE_DISCLAIMER}
    b = load_voice_model()
    if b is None:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None, "reason": _V["err"]}
    try:
        yr, sr, ir = pa.load_audio(reference_path)
        yp, _, ip = pa.load_audio(probe_path)
    except pa.AudioError as e:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None, "reason": str(e)}
    short = [n for n, i in (("reference", ir), ("compared", ip)) if i["analysed_duration_s"] < config.VOICE_MIN_SECONDS]
    if short:
        return {**base, "status": "unavailable", "label": "Unavailable", "similarity": None,
                "reason": f"The {' and '.join(short)} recording is shorter than {config.VOICE_MIN_SECONDS:.0f} s."}
    er, ep = _voice_embed(b, pa.speaker_features(yr, sr)), _voice_embed(b, pa.speaker_features(yp, sr))
    sim = float(np.dot(er, ep) / (np.linalg.norm(er) * np.linalg.norm(ep) + 1e-9))
    thr = b["threshold"]
    margin = 0.05
    if abs(sim - thr) < margin:
        label = "Unknown / Review"
        why = f"Similarity {sim:.3f} is within ±{margin} of the decision threshold {thr:.3f}: inconclusive."
    else:
        label, why = ("Match" if sim >= thr else "Mismatch"), None
    return {**base, "status": "experimental", "label": label, "similarity": round(sim, 4),
            "similarity_metric": f"cosine ({b['kind']})", "threshold": round(thr, 4), "reasons": [why] if why else [],
            "threshold_note": "Threshold tuned on validation speakers from DeepfakeTIMIT (clean studio speech).",
            "model_info": {k: b["info"][k] for k in ("dataset", "n_speakers", "experimental_reason")},
            "durations_s": [ir["analysed_duration_s"], ip["analysed_duration_s"]]}
