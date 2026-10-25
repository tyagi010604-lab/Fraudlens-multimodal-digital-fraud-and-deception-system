"""Locate and (lazily, minimally) read the downloaded dataset archives.

Rules: never download anything, never extract more than needed, never keep duplicates.
Archives are searched in DATA_DIR, ./data and any FRAUDLENS_EXTRA_DATA_DIRS.
"""
import io
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

import config


class DatasetMissing(RuntimeError):
    pass


def search_dirs():
    dirs = [config.DATA_DIR, config.BASE_DIR / "data", *config.EXTRA_DATA_DIRS]
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def find_archive(key):
    """Return the Path of the archive for dataset `key` or None."""
    for d in search_dirs():
        if not d.exists():
            continue
        for pat in config.DATASET_ARCHIVES[key]:
            hits = [h for h in sorted(d.glob(pat)) if h.is_file()]
            if hits:
                return hits[0]
    return None


def require_archive(key):
    p = find_archive(key)
    if p is None:
        raise DatasetMissing(
            f"Dataset '{key}' not found. Place {config.DATASET_ARCHIVES[key][0]} in "
            f"{config.DATA_DIR} (or set FRAUDLENS_DATA_DIR).")
    return p


# ------------------------------------------------------------------ SMS Spam Collection ----
def load_sms():
    """Return DataFrame[label, text] read directly from the zip (no extraction)."""
    p = require_archive("sms")
    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith("SMSSpamCollection"))
        raw = z.read(name).decode("utf-8", errors="replace")
    rows = [ln.split("\t", 1) for ln in raw.splitlines() if "\t" in ln]
    return pd.DataFrame(rows, columns=["label", "text"])


# --------------------------------------------------------------------------- SDFVD ----------
def ensure_sdfvd():
    """Extract the nested SDFVD videos once into the cache. Returns DataFrame[path,label,pair_id,name]."""
    dest = config.CACHE_DIR / "sdfvd"
    if not (dest / "videos_real").exists():
        outer = require_archive("sdfvd")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(outer) as zo:
            inner_name = next(n for n in zo.namelist() if n.lower().endswith(".zip"))
            inner_bytes = zo.read(inner_name)           # ~180 MB held briefly in RAM, not written
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zi:
            for info in zi.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                    continue
                parts = Path(info.filename).parts
                target = dest / parts[-2] / parts[-1]
                target.parent.mkdir(parents=True, exist_ok=True)
                with zi.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        del inner_bytes
    rows = []
    for label, folder in (("real", "videos_real"), ("fake", "videos_fake")):
        for f in sorted((dest / folder).glob("*.mp4")):
            m = re.search(r"(\d+)", f.stem)
            rows.append({"path": str(f), "label": label, "name": f.name,
                         "pair_id": int(m.group(1)) if m else -1})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- DeepfakeTIMIT ------
def ensure_timit_audio(tier="higher_quality"):
    """Extract only the WAV files of ONE quality tier (the two tiers hold byte-identical audio).

    Returns DataFrame[speaker, utt, path]. The WAVs are the *original, unmanipulated* speech
    of the VidTIMIT subjects (DeepfakeTIMIT only swaps faces), i.e. genuine human speech.
    """
    dest = config.CACHE_DIR / "deepfaketimit_wav"
    if not dest.exists() or not any(dest.rglob("*.wav")):
        src = require_archive("deepfaketimit")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src, "r:*") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".wav") or f"/{tier}/" not in m.name:
                    continue
                parts = Path(m.name).parts
                if ".." in parts:
                    continue
                spk, utt = parts[-2], Path(parts[-1]).stem
                out = dest / spk / f"{utt}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as fsrc, open(out, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
    rows = [{"speaker": p.parent.name, "utt": p.stem, "path": str(p)} for p in sorted(dest.rglob("*.wav"))]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- UyghurDF ---------
def ensure_uyghur():
    """Extract the 47-WAV preview (small) and return its metadata DataFrame (with absolute paths)."""
    dest = config.CACHE_DIR / "uyghurdf"
    root = dest / "UyghurDF_preview_v0.1"
    if not (root / "metadata.csv").exists():
        src = require_archive("uyghurdf")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src, "r:*") as tf:
            safe = [m for m in tf.getmembers() if (m.isfile() or m.isdir())
                    and ".." not in Path(m.name).parts and not Path(m.name).is_absolute()]
            tf.extractall(dest, members=safe)
    md = pd.read_csv(root / "metadata.csv")
    md["abs_path"] = md["audio_path"].map(lambda p: str(root / p))
    return md


# --------------------------------------------------------------------------- QR dataset ------
def load_qr_urls(force=False):
    """Decode the 1000 QR PNGs inside the nested RAR. Returns DataFrame[name, label, url].

    Decoded strings are cached to CACHE_DIR/qr_urls.csv so decoding runs once.
    """
    cache = config.CACHE_DIR / "qr_urls.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, keep_default_na=False)
    import cv2
    import numpy as np
    outer = require_archive("qr")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_rar = config.CACHE_DIR / "qr_tmp.rar"
    with zipfile.ZipFile(outer) as zo:
        rar_name = next(n for n in zo.namelist() if n.lower().endswith(".rar"))
        tmp_rar.write_bytes(zo.read(rar_name))
    det = cv2.QRCodeDetector()
    rows = []

    def decode(name, data):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        text = ""
        if img is not None:
            text, _, _ = det.detectAndDecode(img)
        parent = Path(str(name).replace("\\", "/")).parent.name.lower()   # e.g. benign_qr_images_500
        rows.append({"name": Path(name).name,
                     "label": "benign" if parent.startswith("benign") else "malicious",
                     "url": (text or "").strip()})

    try:
        from unrar.cffi import rarfile as cffi_rar        # pip install unrar-cffi
        rf = cffi_rar.RarFile(str(tmp_rar))
        for n in rf.namelist():
            if n.lower().endswith(".png"):
                decode(n, rf.read(n))
    except ImportError:
        tool = shutil.which("unrar") or shutil.which("7z") or shutil.which("7za")
        if tool is None:
            tmp_rar.unlink(missing_ok=True)
            raise DatasetMissing("Reading the QR dataset needs `pip install unrar-cffi` or an unrar/7-Zip binary.")
        out_dir = config.CACHE_DIR / "qr_png"
        out_dir.mkdir(parents=True, exist_ok=True)
        if "7z" in tool:
            cmd = [tool, "x", "-y", str(tmp_rar), f"-o{out_dir}"]
        else:
            cmd = [tool, "x", "-y", str(tmp_rar), str(out_dir) + "/"]
        subprocess.run(cmd, check=True, capture_output=True)
        for p in out_dir.rglob("*.png"):
            decode(str(p), p.read_bytes())
        shutil.rmtree(out_dir, ignore_errors=True)
    tmp_rar.unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# ------------------------------------------------------------ optional extra labelled text ----
_POS = {"1", "spam", "scam", "fraud", "phishing", "smishing", "malicious", "fake"}
_NEG = {"0", "ham", "legit", "legitimate", "benign", "normal", "genuine", "real"}


def load_extra_text():
    """Optional additional labelled text: CSV files in <data dir>/extra_text/ with columns `text,label`.

    label: scam/spam/phishing/fraud/1  -> positive ;  ham/legit/benign/0 -> negative. Extra column `category` is kept.
    Returns an (possibly empty) DataFrame[text,label(0/1),category,source]. Rows with unknown labels are dropped.
    """
    frames = []
    for d in search_dirs():
        for f in sorted((d / "extra_text").glob("*.csv")) if (d / "extra_text").exists() else []:
            df = pd.read_csv(f)
            if not {"text", "label"} <= set(df.columns):
                print(f"[extra_text] skipping {f.name}: needs columns text,label")
                continue
            lab = df["label"].astype(str).str.strip().str.lower()
            y = lab.map(lambda v: 1 if v in _POS else (0 if v in _NEG else None))
            keep = y.notna() & df["text"].notna()
            out = pd.DataFrame({"text": df.loc[keep, "text"].astype(str), "y": y[keep].astype(int)})
            out["category"] = df.loc[keep, "category"] if "category" in df.columns else ""
            out["source"] = f"extra:{f.name}"
            frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["text", "y", "category", "source"])
