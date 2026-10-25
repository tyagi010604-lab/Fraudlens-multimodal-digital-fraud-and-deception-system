"""FraudLens configuration: paths, dataset locations, thresholds and fusion weights.

Everything that a student might want to tweak lives here.
Dataset archives are NOT copied: point FRAUDLENS_DATA_DIR at the folder that holds
your downloaded archives (default: ./data). Only small cache extracts are written
to ./data/_cache (git-ignored).
"""
import os
from pathlib import Path

APP_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FRAUDLENS_DATA_DIR", BASE_DIR / "data"))
EXTRA_DATA_DIRS = [Path(p) for p in os.environ.get("FRAUDLENS_EXTRA_DATA_DIRS", "").split(os.pathsep) if p]
CACHE_DIR = Path(os.environ.get("FRAUDLENS_CACHE_DIR", BASE_DIR / "data" / "_cache"))
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVAL_DIR = OUTPUTS_DIR / "eval"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# --- Dataset archives (exact file names as downloaded, plus glob fallbacks) -------------
DATASET_ARCHIVES = {
    "sdfvd": ["SDFVD_Small-scale_Deepfake_Forgery_Video_Dataset.zip", "*SDFVD*.zip"],
    "deepfaketimit": ["DeepfakeTIMIT_tar.gz", "DeepfakeTIMIT.tar.gz", "DeepfakeTIMIT.tar", "*DeepfakeTIMIT*"],
    "uyghurdf": ["1790257220952-UyghurDF_preview_v0_1_tar.gz", "UyghurDF_preview_v0.1.tar.gz", "*UyghurDF*"],
    "sms": ["sms_spam_collection.zip", "*sms*spam*.zip"],
    "phishing": ["phishing_websites.zip", "*phishing*.zip"],
    "qr": ["Dataset_of_1000_Images_of_Malicious_and_Benign_QR_codes_2025.zip", "*QR*.zip"],
}

# --- Tesseract OCR (Module 6) --------------------------------------------------------------
# Leave empty to auto-detect (environment variable TESSERACT_CMD, PATH, then standard Windows/macOS/Linux install
# folders). To force a location, set e.g.  TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = ""

# --- Model files ---------------------------------------------------------------------------
TEXT_MODEL_PATH = MODELS_DIR / "text_scam_model.joblib"
VIDEO_MODEL_PATH = MODELS_DIR / "video_deepfake_model.joblib"
AUDIO_MODEL_PATH = MODELS_DIR / "audio_deepfake_model.joblib"
AUDIO_STATUS_PATH = MODELS_DIR / "audio_status.json"
VOICE_ID_PATH = MODELS_DIR / "voice_verifier.joblib"
URL_MODEL_PATH = MODELS_DIR / "url_lexical_model.joblib"
FACE_MODEL_DIR = MODELS_DIR / "face"
FACE_DETECTOR_ONNX = FACE_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_ONNX = FACE_MODEL_DIR / "face_recognition_sface_2021dec.onnx"

# --- Upload limits / supported types ------------------------------------------------------
MAX_UPLOAD_MB = 200
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
AUDIO_EXTS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

# --- Video processing ----------------------------------------------------------------------
VIDEO_N_FRAMES = 16          # frames sampled uniformly per clip
VIDEO_MAX_WIDTH = 640        # frames are down-scaled to this width before feature extraction
VIDEO_MAX_SECONDS_WARN = 600
TEMPORAL_OUTLIER_Z = 3.5     # robust z-score for "abnormal feature jump" frames

# --- Audio processing ----------------------------------------------------------------------
AUDIO_SR = 16000
AUDIO_MAX_SECONDS = 30       # audio is truncated to this length for analysis
AUDIO_MIN_SECONDS = 1.0
VOICE_MIN_SECONDS = 2.0      # shorter clips are not compared in voice verification

# --- Face verification (OpenCV SFace; cosine similarity, OpenCV-documented threshold) -------
FACE_COSINE_THRESHOLD = 0.363
FACE_VIDEO_FRAMES = 8

# --- Risk categories (applies to every 0..1 risk score) -----------------------------------
RISK_LOW_MAX = 0.35          # score <  0.35 -> Low
RISK_HIGH_MIN = 0.65         # score >= 0.65 -> High ; otherwise Medium

# --- Fusion weights (documented weighted-score baseline) ------------------------------------
# Only modules that actually produced a usable result take part; weights are re-normalised
# over those modules. Experimental modules have their weight multiplied by EXPERIMENTAL_FACTOR.
FUSION_WEIGHTS = {
    "message_content": 0.35,   # pasted text and OCR text are ONE evidence group (max, not sum)
    "video_authenticity": 0.25,
    "audio_authenticity": 0.20,
    "identity": 0.12,          # face / voice mismatch against a user-supplied reference
    "consistency": 0.08,       # transparent audio-video / cross-module rule findings
}
EXPERIMENTAL_FACTOR = 0.5
RULE_BLEND = 0.25            # share of the rule-based indicator score blended into message_content

# --- Text model ---------------------------------------------------------------------------
TEXT_CHUNK_CHARS = 300       # long texts are scored in chunks (model was trained on SMS-length text)
TEXT_MIN_TOKENS = 3

DISCLAIMER = (
    "FraudLens is a student research prototype intended to support preliminary evidence review. "
    "Its predictions may be incorrect and must not be treated as proof of fraud, identity or "
    "criminal activity. Human verification is required."
)
