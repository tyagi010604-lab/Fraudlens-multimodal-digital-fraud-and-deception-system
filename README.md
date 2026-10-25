# FraudLens — Multimodal AI-Based Digital Fraud and Deception Detection

A Streamlit research prototype that analyses **video, audio, images/screenshots and text** with nine separate modules, fuses the *available*
evidence into a **preliminary** Low/Medium/High risk, flags unfamiliar cases as *Unknown / Review*, and produces a downloadable report.

> FraudLens is a student research prototype intended to support preliminary evidence review. Its predictions may be incorrect and must not be treated
> as proof of fraud, identity or criminal activity. Human verification is required.

## 1. Requirements
* Python **3.10 or 3.11**, VS Code.
* **Tesseract OCR** program (for Module 6) - separate from `pip`. **Windows steps:**
  1. Download the installer from <https://github.com/UB-Mannheim/tesseract/wiki> (64-bit `tesseract-ocr-w64-setup-...exe`).
  2. Run it, keep the default folder `C:\Program Files\Tesseract-OCR`, and make sure **English** language data is ticked.
  3. **Close and reopen VS Code** (and its terminal), then `streamlit run app.py`. FraudLens checks `TESSERACT_CMD`, `config.py`, PATH and the standard
     Windows install folders automatically, so editing PATH is *not* required.
  4. Installed somewhere else? Set `TESSERACT_CMD = r"D:\Tools\Tesseract-OCR\tesseract.exe"` in `config.py`, or run `setx TESSERACT_CMD "D:\Tools\Tesseract-OCR\tesseract.exe"` and reopen VS Code.
  5. Verify: `python scripts/check_setup.py` (should print "Tesseract OCR: OK - ... found at ...").
  macOS - `brew install tesseract`; Ubuntu/Debian - `sudo apt install tesseract-ocr`. Without it the app still runs and shows these instructions.
* **ffmpeg** (optional): needed to use a video's own audio track and to decode MP3/M4A. Windows: `winget install ffmpeg`; macOS: `brew install ffmpeg`; Ubuntu: `sudo apt install ffmpeg`.

## 2. Setup (VS Code terminal, inside the `FraudLens` folder)
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
In VS Code: *Ctrl+Shift+P → Python: Select Interpreter → .venv*.

## 3. Datasets (not copied by the project)
Put the downloaded archives **unchanged** in `FraudLens/data/` (or elsewhere and set `FRAUDLENS_DATA_DIR`, e.g. PowerShell `$env:FRAUDLENS_DATA_DIR="D:\datasets"`):

```
sms_spam_collection.zip
SDFVD_Small-scale_Deepfake_Forgery_Video_Dataset.zip
DeepfakeTIMIT_tar.gz                      (really a plain .tar)
1790257220952-UyghurDF_preview_v0_1_tar.gz
Dataset_of_1000_Images_of_Malicious_and_Benign_QR_codes_2025.zip
phishing_websites.zip
```
Only what is needed is extracted, once, into `data/_cache/` (git-ignored; safe to delete). File names are configured in `config.py`.

## 4. Inspect datasets and train
```bash
python scripts/inspect_datasets.py            # -> outputs/eval/dataset_inspection.json (+ see dataset_notes.md)
python scripts/train_models.py --only text    # seconds
python scripts/train_models.py --only voice   # ~1-2 min  (experimental speaker verification calibration)
python scripts/train_models.py --only url     # seconds   (experimental URL hint model)
python scripts/train_models.py --only video   # ~6 min on 1 CPU (extracts features from 106 clips, cached afterwards)
python scripts/train_models.py --only audio   # records why no synthetic-voice detector can be trained from the attached data
python scripts/evaluate_models.py             # re-evaluates saved models on the saved held-out splits
```
(`python scripts/train_models.py` with no argument runs all.) Models are saved to `models/`, metrics and confusion matrices to `outputs/eval/`.
**Models are never trained when the website starts.** Pre-trained files from the build are included; because pickles depend on library versions, if the app
reports a model "could not be loaded", simply re-run the matching training command.

Optional face verification (Module 3): `python scripts/download_face_models.py` (one-time, ~40 MB OpenCV Zoo ONNX files into `models/face/`).

### Adding labelled scam data later (recommended)
The bundled text model is an **SMS spam** classifier. To teach it scams (UPI/KYC/OTP/phishing, job and e-mail scams) add real, labelled data - do not
invent it. Put one or more CSV files in `data/extra_text/` with columns `text,label[,category]` (`label` = `scam`/`spam`/`phishing`/`fraud`/`1` or `legit`/`ham`/`benign`/`0`),
then run `python scripts/train_models.py --only text`. The data is merged, de-duplicated, split and evaluated with the same code; the metrics file records how many extra rows were used.
Include plenty of **legitimate** messages of the same style (delivery notices, bank alerts, chats), otherwise the model will learn "formal = scam".

### Diagnosing your installation
`python scripts/check_setup.py` lists package versions, whether each model loads, Tesseract/ffmpeg/face-model status and dataset detection.
`python scripts/check_setup.py --text "your message"` prints exactly how the text model and rules judged a message (cleaned text, score, threshold, uncertain band, cues).

## 5. Run the website
```bash
streamlit run app.py
```
It starts even if nothing has been trained; the **Home** page lists each module as Ready / Experimental / Unavailable.

Pages: Home/Dashboard · Analyse Evidence · Individual Results · Overall Risk Assessment · Model Evaluation · Dataset Information · Report (JSON/CSV/TXT).
Use **Clear session** to discard results; uploaded files are processed in a temp folder deleted immediately after analysis, and reports mask phone numbers/UPI IDs/e-mails by default.

Tests (headless): `python tests/test_end_to_end.py` and `python tests/test_app_pages.py` (they use the supplied datasets; set `FRAUDLENS_DATA_DIR`).

## 6. Project layout
```
app.py  config.py  requirements.txt  README.md  dataset_notes.md
src/    preprocess_video.py preprocess_audio.py text_ocr.py video_model.py audio_model.py identity_checks.py
        consistency.py fusion.py unknown_detection.py report.py pipeline.py data_access.py
scripts/ inspect_datasets.py train_models.py evaluate_models.py download_face_models.py
models/ outputs/(eval, reports)  data/  tests/
```

## 7. Module status and honest limitations
| # | Module | Status | Key limitation |
|---|---|---|---|
| 1 | Video deepfake | 🧪 Experimental | 106 SDFVD clips; classic forensic features; held-out test 22 clips (F1 0.73, 3 FP/3 FN). Clip-level only: **no frame localisation, no lip-sync**. |
| 2 | AI/cloned voice | 🧪 Pipeline only | **No suitable labelled data was attached** (UyghurDF preview is listening-only; DeepfakeTIMIT audio is all genuine). Shows *"Experimental — suitable labelled training data required."* No score, no accuracy. Train your own with `--audio-real/--audio-fake`. |
| 3 | Face identity | ⛔ until `download_face_models.py` | OpenCV YuNet+SFace; code written but **not verified end-to-end with real weights in the build environment** (no internet there). A match proves nothing about authenticity/identity. |
| 4 | Voice identity | 🧪 Experimental | MFCC+LDA calibrated on 32 clean TIMIT speakers (test EER 0.18). Not synthetic-speech detection. |
| 5 | Text scam | ✅ Ready (SMS domain) | Trained on 2011 SMS spam; test F1 0.968 on SMS only. Rule-based cues are shown separately from the model. |
| 6 | OCR / payment evidence | ✅ if Tesseract installed | OCR errors on low-quality screenshots; regex hits are not proof of fraud. Also decodes QR codes. |
| 7 | Consistency | ✅ Rules | Duration agreement, authenticity-score conflicts, identity conflicts, transcript-rate plausibility. Lip-sync explicitly *not assessed*. |
| 8 | Fusion | ✅ Weighted baseline | Weights in `config.py` are documented defaults, **not learned**; no labelled integration dataset exists, so no XGBoost fusion and no fusion accuracy is claimed. |
| 9 | Unknown cases | ✅ | k-NN feature distance + uncertainty bands calibrated on validation data, plus scope checks. |

Fusion rules: unavailable/untrained/Unknown modules are *excluded* (never zero risk); experimental modules get half weight; pasted text and OCR text form one evidence group;
a face/voice *match* adds no score; if only experimental or indirect evidence exists the result is capped below *High*.

## 7b. How text results are produced
* **Model prediction** - TF-IDF + calibrated linear SVM on de-duplicated SMS spam data (spam = positive class). Score is *not* a calibrated probability outside that domain.
* **Unknown / Review** - returned when the score lies in the class-overlap band (calibrated on ~4,400 out-of-fold predictions; see *Model Evaluation*) or when too few words are in the training vocabulary.
* **Rule-based warnings** - cue categories (urgency, account threat, payment request, sensitive-info request, impersonation, too-good-to-be-true, link prompt/suspicious link). One category is weak; **"high" needs >= 3 different categories at once**.
* **Headline** - *Suspicious* if the model flags spam, or the rules reach "high"; *Unknown / Review* if the model is uncertain; otherwise *No strong warning signs*. A rule-only "Suspicious" is labelled as such.

## 8. Troubleshooting
* *"Tesseract was not found"* → see §1, then `python scripts/check_setup.py`. *MP3 won't open* → install ffmpeg or convert to WAV.
* *`pip install unrar-cffi` fails* → it is only needed for the QR dataset; install an `unrar`/7-Zip binary instead or skip `--only url`.
* *Model "could not be loaded"* → retrain that model (§4).
* Reports are saved only if you press **Save a copy** (to `outputs/reports/`, git-ignored).
