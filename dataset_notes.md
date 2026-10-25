# Dataset notes (inspection of the attached files)

Generated from `python scripts/inspect_datasets.py` (details in `outputs/eval/dataset_inspection.json`, shown in the app under
**Dataset Information**). Datasets were used from their existing archives; only the minimum needed is extracted into `data/_cache/`.

| Dataset | Source / licence (as far as known) | Format & structure | Labels / counts | Size | Verdict | Used for |
|---|---|---|---|---|---|---|
| **SDFVD** – Small-scale Deepfake Forgery Video Dataset | public research dataset (Kaggle/Mendeley); licence not stated in the archive | zip → *nested* `SDFVD.zip` → `videos_real/vN.mp4`, `videos_fake/vsN.mp4` | 53 real, 53 fake (balanced) | 182 MB | **Suitable for an experimental baseline only** | Module 1 |
| **DeepfakeTIMIT** | Idiap (Korshunov & Marcel); built on VidTIMIT; citation required (see its README) | a plain **TAR** (named `.tar.gz` but not gzipped): `higher_quality`/`lower_quality` × 32 speakers × 10 `.avi` + `.wav`; plus 3 `.mov`, 2 `.png`, README | 640 face-swapped videos (0 real), 640 WAV (original speech) | 227 MB | **Not suitable** for real-vs-fake training; **partly suitable** for speaker data | Module 4 calibration, tests |
| **UyghurDF preview v0.1** | CC-BY-4.0 (see `LICENSE`, `THIRD_PARTY_NOTICES.md`) | tar.gz: `bonafide|spoof/<generator>/<split>/*.wav`, `metadata.csv`, `protocol.txt`; 16 kHz mono PCM | 3 bona fide, 44 spoof (18 generators) = 47 | 7 MB | **Demonstration / pipeline test only** | Module 2 pipeline tests |
| **SMS Spam Collection v.1** | UCI ML Repository (Almeida & Hidalgo) | zip: `SMSSpamCollection` (tab-separated), `readme` | 4,827 ham, 747 spam = 5,574 (403 exact duplicate texts) | 0.2 MB | **Suitable for SPAM detection on SMS** (not for general scam detection) | Module 5 |
| **Phishing Websites** | UCI ML Repository | zip: `Training Dataset.arff` (+ `.old.arff`, feature description `.docx`) | 11,055 rows, 30 pre-computed -1/0/1 features | 1 MB | **Not suitable** for the live app | not used |
| **1000 QR images (malicious & benign), 2025** | public dataset (2025); licence not stated | zip → *nested* `.rar` → `benign_qr_images_500/`, `malicious_qr_images_500/` PNGs | 500 benign, 500 malicious (987 decode with OpenCV, 981 unique URLs) | 0.6 MB | **Suitable** for decode tests + an experimental URL hint model | Module 6 |

## Details that drive the decisions

* **SDFVD – paired clips.** Each fake `vsN` has the same resolution and (almost always) frame count as real `vN`, i.e. fakes are derived from the paired real.
  A random split would leak the source video across train/test, so all splits and cross-validation are **grouped by pair index**.
  Only 106 clips → the held-out test set holds 11 pairs (22 clips) and every metric has a very wide confidence interval. No unreadable videos were found.
* **DeepfakeTIMIT.** The README states the audio track was *not manipulated*, and the HQ/LQ copies are byte-identical (verified by MD5), so the
  WAVs are original human speech: they contain **no synthetic class** and cannot train the AI-voice detector. They do carry speaker IDs (32 speakers × 10
  sentences), which supports calibrating the experimental voice-comparison method with **speaker-disjoint** splits (16/8/8 speakers). Only one tier is used.
  There is no real-video counterpart in the archive, so it cannot train Module 1 either.
* **UyghurDF preview.** Its README says it "is for listening, not for detector training or evaluation", and 3 bona-fide clips could not support a detector anyway.
  It is used only to verify that WAV loading, resampling and feature extraction work. **No accuracy is claimed from it.** Because no other attached
  dataset has labelled real-vs-synthetic speech, the Module 2 detector is left **untrained** and shows
  *"Experimental — suitable labelled training data required."* (`models/audio_status.json`).
* **SMS Spam Collection.** Exact duplicates (after lower-casing) are removed *before* the stratified 70/15/15 split to prevent leakage. "Spam" here is mostly promotional/premium-rate
  SMS from 2011, not modern scams; there is no UPI/KYC/OTP data. Emails, job adverts and transcripts are therefore a domain shift (handled by chunking + Unknown/Review logic, not solved).
* **Phishing Websites.** All features (SSL state, domain age, Page_Rank, traffic...) require crawling and WHOIS/search look-ups of a live site; the URLs themselves are not provided,
  so the model cannot be applied to a message or screenshot. Documented, not used.
* **QR dataset.** Labels come from folder names. The benign URLs are mostly well-known sites and the malicious ones throw-away domains, so the URL model may learn that sampling artefact.
  It is shown next to URLs as an experimental hint and is **not** fused into the risk score. Reading it needs `unrar-cffi` (or an `unrar`/7-Zip binary).

## Which dataset feeds which module

| Module | Dataset | Status |
|---|---|---|
| 1 Video deepfake | SDFVD | experimental (106 clips) |
| 2 AI/cloned voice | none suitable | **untrained** – pipeline only |
| 3 Face identity | none needed (pretrained OpenCV SFace, optional download) | unavailable until models are downloaded |
| 4 Voice identity | DeepfakeTIMIT speakers | experimental |
| 5 Text scam | SMS Spam Collection | ready (SMS domain) |
| 6 OCR / payment | none (regex + Tesseract); QR dataset for decode tests and URL hint model | ready if Tesseract installed |
| 7 Consistency | none (transparent rules; no paired labelled A/V data) | ready (rules) |
| 8 Fusion | none (no labelled integration data → weighted baseline, no XGBoost) | ready |
| 9 Unknown cases | validation splits of the above | ready |

## Results recorded by `train_models.py` (held-out data, thresholds tuned on validation only)

| Model | Test n | Precision | Recall | F1 | FP | FN | Notes |
|---|---|---|---|---|---|---|---|
| Text – calibrated Linear SVM (TF-IDF word+char) | 774 | 1.000 | 0.938 | 0.968 | 0 | 6 | 5-fold CV F1 0.957 ± 0.010; majority baseline F1 0 |
| Video – RBF-SVM on forensic features (**experimental**) | 22 clips | 0.727 | 0.727 | 0.727 | 3 | 3 | AUC 0.785; accuracy 95 % CI 0.52–0.87; grouped CV over all 106 clips: AUC 0.815, FP 25, FN 5 |
| Voice ID – LDA+cosine (**experimental**) | 720 pairs (8 unseen speakers) | 0.792 | 0.875 | 0.831 | 83 | 45 | EER 0.183, AUC 0.908 on clean studio speech |
| URL lexical (**experimental**, not fused) | 189 | 1.000 | 0.632 | 0.775 | 0 | 39 | AUC 0.949; domain-grouped split |
| Audio deepfake | – | – | – | – | – | – | **no suitable labelled data → no model, no metrics** |

The text numbers describe SMS data of the same distribution as training; they do not predict performance on emails, job adverts or Indian-context scams.

## Spam vs scam (Module 5)
The SMS Spam Collection labels *spam* (mostly 2011 promotions/premium-rate offers) and *ham*. It contains no banking, KYC, OTP, UPI, job-offer or e-mail scams and no
transactional notices (delivery/order/bank alerts). The ML model is therefore a **spam** classifier; scam-specific warning cues come from separate, transparent rules,
and borderline or unfamiliar text is returned as *Unknown / Review*. Out-of-fold reliability of the raw score (SMS domain) is stored in `outputs/eval/text_metrics.json`.
To add suitable labelled scam/legitimate data, see README "Adding labelled scam data later" (`data/extra_text/*.csv`).
