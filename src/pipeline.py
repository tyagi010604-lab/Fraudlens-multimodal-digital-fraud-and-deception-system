"""End-to-end workflow: validate -> preprocess -> run available modules -> consistency -> fusion.

Uploaded bytes are written to a private temporary folder that is ALWAYS deleted when the analysis ends
(success or failure). Nothing is logged except derived results.
"""
import datetime as dt
import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path

import config
from src import audio_model, consistency, fusion, identity_checks, text_ocr, video_model
from src import preprocess_audio as pa


def module_status():
    """Readiness of the nine modules for the dashboard. status in ready | experimental | unavailable."""
    vm = video_model.load_model()
    am = audio_model.load_model()
    tm = text_ocr.load_text_model()
    vi = identity_checks.load_voice_model()
    face_ok, face_msg = identity_checks.face_backend_status()
    tess_ok, tess_msg = text_ocr.check_tesseract()
    S = []
    S.append(("1. Video deepfake detection", "experimental" if vm else "unavailable",
              "Classic forensic features + classifier trained on 106 SDFVD clips; clip-level score only." if vm else
              video_model.model_error()))
    if am:
        S.append(("2. AI/cloned voice detection", "experimental" if am["info"].get("experimental") else "ready",
                  "Classifier trained on user-supplied labelled audio."))
    else:
        S.append(("2. AI/cloned voice detection", "experimental",
                  audio_model.UNTRAINED_MSG + " Pipeline works; no score is produced."))
    S.append(("3. Face identity verification", "ready" if face_ok else "unavailable", face_msg))
    S.append(("4. Voice identity verification", "experimental" if vi else "unavailable",
              "MFCC similarity calibrated on 32 TIMIT speakers (clean studio audio)." if vi else identity_checks._V["err"]))
    S.append(("5. Text scam detection", "ready" if tm else "unavailable",
              "ML model = SMS SPAM classifier (TF-IDF + linear SVM, de-duplicated SMS data); scam cues (credentials, account threats, links, payments) come from separate transparent rules."
              if tm else (text_ocr.text_model_error() + " (rule-based cues still work)")))
    S.append(("6. OCR & payment evidence", "ready" if tess_ok else "unavailable", tess_msg))
    ff = "ffmpeg found" if pa.have_ffmpeg() else "ffmpeg NOT found (needed to use a video's own audio track)"
    S.append(("7. Audio-video consistency", "ready", f"Transparent rules; no lip-sync analysis. Needs two kinds of evidence. {ff}."))
    S.append(("8. Multimodal risk fusion", "ready", "Weighted-score baseline over available evidence (weights in config.py)."))
    S.append(("9. Unknown-case detection", "ready", "Validation-calibrated distance / uncertainty bands plus scope checks."))
    return [{"module": a, "status": b, "detail": c} for a, b, c in S]


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:                     # a failing module must not break the whole analysis
        return {"status": "error", "label": "Not assessed", "risk_score": None,
                "reason": f"{type(e).__name__}: {e}"}


def _save(tmp, upload, kind, exts, evidence):
    name, data = upload
    ext = Path(name).suffix.lower()
    if ext not in exts:
        raise ValueError(f"Unsupported {kind} type '{ext}'. Supported: {', '.join(exts)}.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"{kind} file exceeds {config.MAX_UPLOAD_MB} MB.")
    if len(data) == 0:
        raise ValueError(f"{kind} file is empty.")
    path = tmp / f"{kind}_{len(evidence)}{ext}"
    path.write_bytes(data)
    evidence.append({"type": kind, "extension": ext, "size_mb": round(len(data) / 1e6, 3),
                     "sha256_prefix": hashlib.sha256(data).hexdigest()[:12]})
    return path


def run_analysis(inputs, ocr_video_frames=True, extract_video_audio=True):
    """inputs: dict(text, transcript, video=(name, bytes), audio=(name, bytes), images=[(name, bytes)],
    ref_face=(name, bytes), ref_voice=(name, bytes)). Missing keys mean 'not supplied'."""
    tmp = Path(tempfile.mkdtemp(prefix="fraudlens_"))
    evidence, errors, not_assessed = [], {}, {}
    results = {"ocr": []}
    try:
        text = (inputs.get("text") or "").strip()
        transcript = (inputs.get("transcript") or "").strip()
        paths = {}
        for key, kind, exts in (("video", "video", config.VIDEO_EXTS), ("audio", "audio", config.AUDIO_EXTS),
                                ("ref_face", "reference_face", config.IMAGE_EXTS),
                                ("ref_voice", "reference_voice", config.AUDIO_EXTS)):
            if inputs.get(key):
                try:
                    paths[key] = _save(tmp, inputs[key], kind, exts, evidence)
                except ValueError as e:
                    errors[key] = str(e)
        img_paths = []
        for up in inputs.get("images") or []:
            try:
                img_paths.append((up[0], _save(tmp, up, "image", config.IMAGE_EXTS, evidence)))
            except ValueError as e:
                errors[f"image:{up[0]}"] = str(e)

        # --- Module 5: text -------------------------------------------------------------------
        if text:
            results["text"] = _safe(text_ocr.analyze_text, text)
        else:
            not_assessed["5. Text scam detection"] = "No text was entered."
        if transcript and not text:
            results["transcript_text"] = _safe(text_ocr.analyze_text, transcript, "call transcript")

        # --- Module 6: OCR on images (and selected video frames) -----------------------------------
        for i, (orig, p) in enumerate(img_paths):
            r = _safe(text_ocr.analyze_image, p, f"image {i + 1}")
            if r.get("status") == "error" and "reason" in r and "text_analysis" not in r:
                r.setdefault("source", f"image {i + 1}")
            results["ocr"].append(r)
        if not img_paths and not (paths.get("video") and ocr_video_frames):
            not_assessed["6. OCR & payment evidence"] = "No image or screenshot was uploaded."

        # --- Module 1: video -----------------------------------------------------------------------
        if "video" in paths:
            results["video"] = _safe(video_model.analyse_video, paths["video"])
            if ocr_video_frames and results["video"].get("status") != "error":
                try:
                    from PIL import Image
                    import cv2
                    from src import preprocess_video as pv
                    frames = pv.sample_frames(paths["video"], 4)
                    for _, ts, fr in [frames[0], frames[len(frames) // 2], frames[-1]]:
                        o = _safe(text_ocr.analyze_image, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)),
                                  f"video frame at {ts:.1f}s")
                        o_ocr = o.get("ocr", {})
                        # frames without genuine on-screen text yield OCR noise; keep only credible text
                        if o.get("status") == "ok" and o_ocr.get("n_words", 0) >= 4 and (o_ocr.get("mean_confidence") or 0) >= 60:
                            results["ocr"].append(o)
                except Exception:
                    pass
        elif "video" in errors:
            results["video"] = {"status": "error", "label": "Not assessed", "reason": errors["video"]}
        else:
            not_assessed["1. Video deepfake detection"] = "No video was uploaded."

        # --- Module 2: audio (separate file, or the video's own track) ----------------------------------
        audio_path, audio_source = paths.get("audio"), "separate" if "audio" in paths else None
        has_track = None
        if "video" in paths:
            has_track = pa.video_has_audio_track(paths["video"])
        if audio_path is None and "video" in paths and extract_video_audio:
            ex = pa.extract_audio_from_video(paths["video"], tmp)
            if ex:
                audio_path, audio_source = ex, "extracted"
        if audio_path is not None:
            results["audio"] = _safe(audio_model.analyse_audio, audio_path,
                                     "audio track extracted from the video" if audio_source == "extracted" else "uploaded audio file")
        elif "audio" in errors:
            results["audio"] = {"status": "error", "label": "Not assessed", "reason": errors["audio"]}
        else:
            not_assessed["2. AI/cloned voice detection"] = (
                "No audio was uploaded" + ("" if "video" not in paths else
                                           " and no audio track could be extracted from the video "
                                           "(needs ffmpeg and a video with sound)") + ".")

        # --- Module 3: face identity --------------------------------------------------------------------
        if "ref_face" in paths:
            if "video" in paths:
                results["face"] = _safe(identity_checks.verify_face, paths["ref_face"], paths["video"], True)
            elif img_paths:
                results["face"] = _safe(identity_checks.verify_face, paths["ref_face"], img_paths[0][1], False)
            else:
                not_assessed["3. Face identity verification"] = "A reference face was given but there is no image/video to check."
        else:
            not_assessed["3. Face identity verification"] = "No reference face image supplied."

        # --- Module 4: voice identity -------------------------------------------------------------------
        if "ref_voice" in paths:
            if audio_path is not None:
                results["voice"] = _safe(identity_checks.compare_voices, paths["ref_voice"], audio_path)
            else:
                not_assessed["4. Voice identity verification"] = "A reference voice was given but there is no recording to compare."
        else:
            not_assessed["4. Voice identity verification"] = "No reference voice recording supplied."

        types_supplied = [t for t, ok in (("text", bool(text)), ("transcript", bool(transcript)), ("video", "video" in paths),
                                          ("audio", audio_path is not None), ("image", bool(img_paths))) if ok]

        # --- Module 7: consistency ----------------------------------------------------------------------
        ctx = {"video": results.get("video"), "audio": results.get("audio"), "audio_source": audio_source,
               "face": results.get("face"), "voice": results.get("voice"), "transcript": transcript,
               "video_has_audio_track": has_track}
        results["consistency"] = _safe(consistency.run_consistency, ctx)
        if results["consistency"].get("status") == "not_assessed":
            have = ", ".join(types_supplied) or "none"
            hint = ""
            if "video" in paths and audio_path is None:
                hint = (" The video's audio track could not be extracted (no audio stream, or ffmpeg/ffprobe not installed) "
                        "- upload the audio separately to enable audio-video checks.")
            not_assessed["7. Audio-video consistency"] = f"{results['consistency']['reason']} Evidence supplied: {have}.{hint}"

        # --- Module 8 + 9: fusion (Unknown/Review is handled inside each module and respected here) -------------
        fuse_in = dict(results)
        if "text" not in fuse_in and "transcript_text" in fuse_in:
            fuse_in["text"] = fuse_in["transcript_text"]
        fusion_res = fusion.fuse(fuse_in)

        for k, v in (("video", results.get("video")), ("audio", results.get("audio"))):
            if v and v.get("status") in ("error", "unavailable", "experimental_untrained"):
                not_assessed[{"video": "1. Video deepfake detection", "audio": "2. AI/cloned voice detection"}[k]] = \
                    v.get("reason", "")
        for k, name in (("face", "3. Face identity verification"), ("voice", "4. Voice identity verification")):
            v = results.get(k)
            if v and v.get("label") == "Unavailable":
                not_assessed[name] = v.get("reason", "")

        types = [t for t, ok in (("text", bool(text)), ("transcript", bool(transcript)), ("video", "video" in paths),
                                 ("audio", "audio" in paths), ("image/screenshot", bool(img_paths)),
                                 ("reference face", "ref_face" in paths), ("reference voice", "ref_voice" in paths)) if ok]
        return {"analysis_id": uuid.uuid4().hex[:10].upper(),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "evidence_types": types, "evidence": evidence, "results": results, "fusion": fusion_res,
                "not_assessed": not_assessed, "input_errors": errors, "module_status": module_status()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)       # privacy: uploaded media never persists
