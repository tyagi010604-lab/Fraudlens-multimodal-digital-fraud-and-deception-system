"""Video preprocessing: validation, uniform frame sampling and lightweight visual features.

The features are classic, interpretable image-forensics statistics (sharpness, high-frequency
energy, noise residual, colour statistics, face-vs-background differences and frame-to-frame
change). They are *not* deep features and do not measure lip-sync or locate manipulated frames.
"""
from pathlib import Path

import cv2
import numpy as np

import config


class VideoError(ValueError):
    """Raised for unreadable / unsupported / oversized video files."""


# ------------------------------------------------------------------------ validation -------
def validate_video(path):
    p = Path(path)
    if not p.exists():
        raise VideoError("File not found.")
    if p.suffix.lower() not in config.VIDEO_EXTS:
        raise VideoError(f"Unsupported video type '{p.suffix}'. Supported: {', '.join(config.VIDEO_EXTS)}.")
    if p.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise VideoError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV could not open this video (corrupted or unsupported codec).")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n <= 0 or w <= 0 or h <= 0:
            raise VideoError("Video has no readable frames.")
        ok, _ = cap.read()
        if not ok:
            raise VideoError("First frame could not be decoded (corrupted file).")
    finally:
        cap.release()
    duration = n / fps if fps > 0 else None
    return {"frames": n, "fps": round(fps, 3), "width": w, "height": h,
            "duration_s": round(duration, 3) if duration else None, "size_mb": round(p.stat().st_size / 1e6, 2)}


# ---------------------------------------------------------------------- frame sampling -----
def _resize(frame, max_w):
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)
    return frame


def sample_frames(path, n_frames=config.VIDEO_N_FRAMES, max_width=config.VIDEO_MAX_WIDTH):
    """Uniformly sample up to n_frames. Returns list of (frame_index, timestamp_s, BGR image)."""
    info = validate_video(path)
    total, fps = info["frames"], info["fps"] or 25.0
    idxs = np.unique(np.linspace(0, max(total - 1, 0), num=min(n_frames, total)).astype(int))
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok and fr is not None:
                out.append((int(i), round(i / fps, 3), _resize(fr, max_width)))
    finally:
        cap.release()
    if not out:
        raise VideoError("No frames could be decoded from this video.")
    return out


# -------------------------------------------------------------------------- face detect ----
_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def detect_largest_face(gray):
    """Haar-cascade detection of the largest frontal face. Returns (x, y, w, h) or None."""
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(max(40, gray.shape[1] // 12),) * 2)
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


# ---------------------------------------------------------------------- image statistics ---
_MASKS = {}


def _hf_mask(n):
    if n not in _MASKS:
        yy, xx = np.mgrid[:n, :n]
        r = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
        _MASKS[n] = r > 0.5
    return _MASKS[n]


def hf_ratio(gray, size=128):
    """Share of spectral power in the upper half of spatial frequencies (GAN/resampling traces)."""
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    total = spec.sum() + 1e-9
    return float(spec[_hf_mask(size)].sum() / total)


def lap_var(gray):
    return float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()))


def noise_std(gray):
    g = gray.astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (5, 5), 0)).std())


PER_FRAME = ["lap_var", "hf_ratio", "noise_std", "sat_mean", "val_mean", "val_std",
             "face_lap", "face_hf", "face_noise", "face_bg_sharp_ratio", "face_chroma_diff"]
POSITION = ["face_cx", "face_cy", "face_area"]


def frame_features(frame):
    """Return dict of per-frame features (face features are NaN when no face is found)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    f = {"lap_var": lap_var(gray), "hf_ratio": hf_ratio(gray), "noise_std": noise_std(gray),
         "sat_mean": float(hsv[..., 1].mean()), "val_mean": float(hsv[..., 2].mean()),
         "val_std": float(hsv[..., 2].std())}
    nan = float("nan")
    f.update({k: nan for k in PER_FRAME[6:] + POSITION})
    box = detect_largest_face(gray)
    f["face_found"] = 0.0
    if box is not None:
        x, y, w, h = box
        face = gray[y:y + h, x:x + w]
        if face.size > 0:
            f["face_found"] = 1.0
            f["face_lap"] = lap_var(face)
            f["face_hf"] = hf_ratio(face, 64)
            f["face_noise"] = noise_std(face)
            H, W = gray.shape
            f["face_cx"], f["face_cy"] = (x + w / 2) / W, (y + h / 2) / H
            f["face_area"] = (w * h) / (W * H)
            # background ring = box enlarged by 50 %, minus the box itself
            x0, y0 = max(0, x - w // 4), max(0, y - h // 4)
            x1, y1 = min(W, x + w + w // 4), min(H, y + h + h // 4)
            ring_mask = np.ones(gray.shape, bool)
            ring_mask[:] = False
            ring_mask[y0:y1, x0:x1] = True
            ring_mask[y:y + h, x:x + w] = False
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            bg = float(np.log1p(lap[ring_mask].var())) if ring_mask.sum() > 50 else nan
            f["face_bg_sharp_ratio"] = f["face_lap"] - bg if bg == bg else nan
            ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            if ring_mask.sum() > 50:
                f["face_chroma_diff"] = float(
                    abs(ycc[y:y + h, x:x + w, 1].mean() - ycc[..., 1][ring_mask].mean()) +
                    abs(ycc[y:y + h, x:x + w, 2].mean() - ycc[..., 2][ring_mask].mean()))
    return f


def feature_names():
    names = []
    for k in PER_FRAME:
        names += [f"{k}_mean", f"{k}_std"]
    names += ["face_found_frac", "face_cx_std", "face_cy_std", "frame_diff_mean", "frame_diff_std"]
    return names


def extract_video_features(path, n_frames=config.VIDEO_N_FRAMES, frames=None):
    """Compute the clip-level feature vector.

    Returns dict(vector, names, per_frame (list of dict with timestamp), meta).
    """
    info = validate_video(path)
    frames = frames or sample_frames(path, n_frames)
    rows, small = [], []
    for idx, ts, fr in frames:
        d = frame_features(fr)
        d["frame_index"], d["timestamp_s"] = idx, ts
        rows.append(d)
        small.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (96, 96)).astype(np.float32))
    vec = []
    with np.errstate(all="ignore"):
        for k in PER_FRAME:
            col = np.array([r[k] for r in rows], dtype=float)
            if np.isnan(col).all():
                vec += [0.0, 0.0]
            else:
                vec += [float(np.nanmean(col)), float(np.nanstd(col))]
        found = np.array([r["face_found"] for r in rows])
        vec.append(float(found.mean()))
        for k in ("face_cx", "face_cy"):
            col = np.array([r[k] for r in rows], dtype=float)
            vec.append(float(np.nanstd(col)) if (~np.isnan(col)).sum() > 1 else 0.0)
    diffs = [float(np.abs(small[i + 1] - small[i]).mean()) for i in range(len(small) - 1)]
    vec += [float(np.mean(diffs)) if diffs else 0.0, float(np.std(diffs)) if diffs else 0.0]
    vec = np.nan_to_num(np.array(vec, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return {"vector": vec, "names": feature_names(), "per_frame": rows, "meta": info,
            "n_frames_used": len(rows), "faces_found": int(found.sum())}


# ----------------------------------------------------------------- temporal outlier frames --
def temporal_outliers(per_frame, z_thresh=config.TEMPORAL_OUTLIER_Z):
    """Frames whose feature values jump far from the clip median (robust z-score).

    This is a transparent *instability indicator* (e.g. blur/compression changes, cuts, face
    re-detection jumps). It does NOT identify manipulated frames.
    """
    out = []
    if len(per_frame) < 6:
        return out
    for k in ("lap_var", "hf_ratio", "noise_std", "face_lap", "face_hf", "face_noise"):
        col = np.array([r[k] for r in per_frame], dtype=float)
        ok = ~np.isnan(col)
        if ok.sum() < 6:
            continue
        med = np.median(col[ok])
        mad = np.median(np.abs(col[ok] - med))
        mad = max(mad, 1e-3 * max(abs(med), 1e-3))
        z = 0.6745 * (col - med) / mad
        for r, zi in zip(per_frame, z):
            if not np.isnan(zi) and abs(zi) > z_thresh:
                out.append({"timestamp_s": r["timestamp_s"], "frame_index": r["frame_index"],
                            "feature": k, "robust_z": round(float(zi), 2)})
    out.sort(key=lambda d: (d["timestamp_s"], d["feature"]))
    return out
