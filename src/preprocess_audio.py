"""Audio preprocessing: validation, resampling to a common format, feature extraction.

Features:
  * `extract_features`   - MFCC / delta / spectral statistics for authenticity classifiers.
  * `mel_spectrogram`    - log-Mel spectrogram (display, or input for a CNN if one is trained).
  * `speaker_features`   - voiced-frame MFCC statistics used by the (experimental) voice-ID module.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import config

HOP, NFFT = 160, 400      # 10 ms hop / 25 ms window at 16 kHz


class AudioError(ValueError):
    """Raised for unreadable / unsupported audio files."""


def validate_audio_path(path):
    p = Path(path)
    if not p.exists():
        raise AudioError("File not found.")
    if p.suffix.lower() not in config.AUDIO_EXTS:
        raise AudioError(f"Unsupported audio type '{p.suffix}'. Supported: {', '.join(config.AUDIO_EXTS)}.")
    if p.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    return p


def load_audio(path, sr=config.AUDIO_SR, max_seconds=config.AUDIO_MAX_SECONDS, check_ext=True):
    """Load any supported audio as mono float32 at `sr`, truncated to max_seconds.

    Returns (y, sr, info). `info` records the ORIGINAL properties plus what was done.
    """
    import librosa
    import soundfile as sf
    p = validate_audio_path(path) if check_ext else Path(path)
    orig_sr = orig_ch = None
    full_dur = None
    try:
        si = sf.info(str(p))
        orig_sr, orig_ch, full_dur = si.samplerate, si.channels, si.duration
    except Exception:
        pass                                    # mp3/m4a may be decoded via librosa/audioread only
    try:
        y, _ = librosa.load(str(p), sr=sr, mono=True, duration=max_seconds)
    except Exception as e:                      # corrupted / unsupported codec
        raise AudioError(f"Audio could not be decoded ({type(e).__name__}). "
                         "For MP3/M4A install ffmpeg or convert to WAV.") from e
    if y.size == 0:
        raise AudioError("Audio file contains no samples.")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.95                     # peak-normalise (level carries no authenticity info here)
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    # share of 25 ms frames more than ~26 dB below the loudest frame
    silence_ratio = float(np.mean(rms < 0.05 * rms.max())) if rms.size and rms.max() > 0 else 1.0
    info = {"original_sample_rate": orig_sr, "original_channels": orig_ch,
            "original_duration_s": round(full_dur, 3) if full_dur else None,
            "analysed_duration_s": round(dur, 3), "analysis_sample_rate": sr,
            "truncated": bool(full_dur and full_dur > max_seconds),
            "silence_ratio": round(silence_ratio, 3), "peak_before_norm": round(peak, 4),
            "size_mb": round(p.stat().st_size / 1e6, 3)}
    return y.astype(np.float32), sr, info


# ------------------------------------------------------------------------------- features --
def _stats(a):
    return [float(np.mean(a)), float(np.std(a))]


def feature_names():
    names = []
    for grp in ("mfcc", "dmfcc"):
        for i in range(20):
            names += [f"{grp}{i}_mean", f"{grp}{i}_std"]
    for k in ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms"):
        names += [f"{k}_mean", f"{k}_std"]
    names += [f"contrast{i}_mean" for i in range(7)]
    return names


def extract_features(y, sr=config.AUDIO_SR):
    """Fixed-length (99-D) feature vector: MFCC + delta stats, spectral stats, spectral contrast."""
    import librosa
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=NFFT, hop_length=HOP)
    d = librosa.feature.delta(mf) if mf.shape[1] >= 9 else np.zeros_like(mf)
    v = []
    for m in (mf, d):
        for row in m:
            v += _stats(row)
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    for f in (librosa.feature.spectral_centroid(S=S, sr=sr)[0], librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
              librosa.feature.spectral_rolloff(S=S, sr=sr)[0], librosa.feature.spectral_flatness(S=S)[0],
              librosa.feature.zero_crossing_rate(y, frame_length=NFFT, hop_length=HOP)[0],
              librosa.feature.rms(S=S, frame_length=NFFT)[0]):
        v += _stats(f)
    con = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=100.0)
    v += [float(r.mean()) for r in con]
    return np.nan_to_num(np.array(v, dtype=np.float64))


def mel_spectrogram(y, sr=config.AUDIO_SR, n_mels=64):
    import librosa
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=HOP, n_mels=n_mels)
    return librosa.power_to_db(M, ref=np.max)


def speaker_features(y, sr=config.AUDIO_SR):
    """Voiced-frame MFCC mean+std (c1..c23, c0 dropped, per-utterance mean removed) -> 46-D."""
    import librosa
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, n_fft=NFFT, hop_length=HOP)[1:]
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    n = min(mf.shape[1], rms.shape[0])
    mf, rms = mf[:, :n], rms[:n]
    keep = rms >= np.percentile(rms, 30)           # drop the quietest 30 % (pauses / noise floor)
    mf = mf[:, keep] if keep.sum() > 20 else mf
    return np.concatenate([mf.mean(axis=1), mf.std(axis=1)])


def audio_summary(y, sr=config.AUDIO_SR):
    """Small human-readable descriptors for display."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=NFFT, hop_length=HOP)[0]
    return {"rms_mean": round(float(rms.mean()), 4), "spectral_centroid_hz": round(float(cent.mean()), 1),
            "zero_crossing_rate": round(float(librosa.feature.zero_crossing_rate(y, frame_length=NFFT,
                                                                                  hop_length=HOP).mean()), 4)}


# --------------------------------------------------------------------- audio from video ----
def have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def video_has_audio_track(path):
    """True/False via ffprobe, or None if ffprobe is unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def extract_audio_from_video(path, out_dir=None):
    """Extract the audio track to a temporary 16 kHz mono WAV with ffmpeg. Returns Path or None."""
    if not have_ffmpeg() or not video_has_audio_track(path):
        return None
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="fraudlens_"))
    out = out_dir / (Path(path).stem + "_audio.wav")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar",
                        str(config.AUDIO_SR), "-t", str(config.AUDIO_MAX_SECONDS), str(out)],
                       capture_output=True)
    return out if r.returncode == 0 and out.exists() and out.stat().st_size > 1000 else None
