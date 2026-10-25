"""Inspect every attached dataset and write outputs/eval/dataset_inspection.json (read by the app).

    python scripts/inspect_datasets.py

Reads archives in place (listing only, minimal extraction into data/_cache) and records: structure, labels,
counts, size, corrupted/unreadable files and a suitability verdict per dataset.
"""
import json
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import data_access as da  # noqa: E402

out = []


def mb(p):
    return round(Path(p).stat().st_size / 1e6, 1)


def add(**k):
    out.append(k)
    print(f"- {k['name']}: {k['samples']} samples, verdict = {k['verdict']}")


# SDFVD --------------------------------------------------------------------------------------
p = da.find_archive("sdfvd")
if p:
    import cv2
    df = da.ensure_sdfvd()
    bad, res = [], Counter()
    for f in df.path:
        c = cv2.VideoCapture(f)
        ok, fr = c.read()
        if not ok:
            bad.append(Path(f).name)
        else:
            res[f"{int(c.get(3))}x{int(c.get(4))}"] += 1
        c.release()
    pairs = df.groupby("pair_id").label.nunique()
    add(name="SDFVD (Small-scale Deepfake Forgery Video Dataset)", archive=p.name, size_mb=mb(p), samples=len(df),
        classes=df.label.value_counts().to_dict(),
        structure="zip -> SDFVD.zip -> SDFVD/videos_real/vN.mp4 and SDFVD/videos_fake/vsN.mp4 (nested archive)",
        issues=f"{len(bad)} unreadable videos; {int((pairs == 2).sum())}/{len(pairs)} index pairs have both a real (vN) and fake (vsN) clip with "
               f"identical resolution/frame count -> fakes are derived from the paired reals, so splits MUST keep pairs together. Resolutions: {dict(res)}",
        verdict="Suitable for an EXPERIMENTAL video baseline", used_for="Module 1 training/evaluation (pair-grouped splits)",
        reason="Balanced and labelled, but only 106 clips; every metric has a very wide confidence interval.")

# DeepfakeTIMIT ---------------------------------------------------------------------------------
p = da.find_archive("deepfaketimit")
if p:
    with tarfile.open(p, "r:*") as tf:
        names = [m.name for m in tf if m.isfile()]
    ext = Counter(Path(n).suffix.lower() for n in names)
    tiers = Counter(n.split("/")[1] for n in names if n.endswith(".avi"))
    spk = {n.split("/")[2] for n in names if n.endswith(".wav")}
    add(name="DeepfakeTIMIT", archive=p.name, size_mb=mb(p), samples=ext[".avi"],
        classes={"fake videos (HQ)": tiers.get("higher_quality", 0), "fake videos (LQ)": tiers.get("lower_quality", 0),
                 "real videos": 0, "wav (original speech)": ext[".wav"]},
        structure=f"plain TAR (despite the .gz name) -> DeepfakeTIMIT/{{higher,lower}}_quality/<speaker>/<utt>.avi + .wav ; {len(spk)} speakers; extras: {dict(ext)}",
        issues="Contains ONLY face-swapped (fake) videos - no real counterpart (VidTIMIT originals are not included) -> cannot train a real-vs-fake video classifier. "
               "WAVs are the unmodified original speech (README), byte-identical in HQ/LQ folders -> genuine human speech only, no synthetic class.",
        verdict="Partly suitable (speaker data) / not suitable for deepfake classifier training",
        used_for="Module 4 speaker-verification calibration (speaker-disjoint split) and pipeline tests; NOT for Module 1/2 training",
        reason="Speaker-labelled clean speech supports an experimental identity-comparison calibration; one-class media cannot train detectors.")

# UyghurDF -----------------------------------------------------------------------------------------
p = da.find_archive("uyghurdf")
if p:
    md = da.ensure_uyghur()
    add(name="UyghurDF preview v0.1", archive=p.name, size_mb=mb(p), samples=len(md), classes=md.label.value_counts().to_dict(),
        structure="tar.gz -> bonafide|spoof/<generator>/<split>/*.wav + metadata.csv + protocol.txt; 16 kHz mono 16-bit",
        issues=f"README: listening sample, 'not for detector training or evaluation'. {md.generator.nunique() - 1} generators, only "
               f"{int((md.label == 'bonafide').sum())} bona-fide clips. Uyghur language only.",
        verdict="Demonstration / pipeline test only", used_for="Testing audio loading, preprocessing, feature extraction and the audio upload workflow",
        reason="Too small and explicitly not intended for training/evaluation; no accuracy is claimed from it.")

# SMS ----------------------------------------------------------------------------------------------
p = da.find_archive("sms")
if p:
    df = da.load_sms()
    dup = int(df.text.str.lower().str.strip().duplicated().sum())
    add(name="SMS Spam Collection v.1 (UCI)", archive=p.name, size_mb=mb(p), samples=len(df), classes=df.label.value_counts().to_dict(),
        structure="zip -> SMSSpamCollection (tab-separated label<TAB>text) + readme",
        issues=f"{dup} exact duplicate texts (removed before splitting to avoid leakage). English, 2011-era UK/Singapore SMS; 'spam' is mostly promotions, not scams.",
        verdict="Suitable (SMS domain only)", used_for="Module 5 text classifier (TF-IDF); also vocabulary for unknown-case detection",
        reason="Labelled, adequately sized; domain shift to emails/job ads/Indian UPI scams is not covered.")

# Phishing websites ---------------------------------------------------------------------------------
p = da.find_archive("phishing")
if p:
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        raw = z.read(next(n for n in names if n.startswith("Training")))
    txt = raw.decode("latin-1")
    attrs = [l for l in txt.splitlines() if l.lower().startswith("@attribute")]
    data = txt.lower().split("@data")[1].strip().splitlines()
    res = Counter(l.strip().split(",")[-1] for l in data if l.strip() and not l.startswith("%"))
    add(name="Phishing Websites (UCI, ARFF)", archive=p.name, size_mb=mb(p), samples=sum(res.values()),
        classes={"phishing(-1)/legitimate(1) raw counts": dict(res)},
        structure=f"zip -> Training Dataset.arff ({len(attrs)} attributes incl. Result), .old.arff, feature description docx",
        issues="Pre-computed categorical features (-1/0/1) such as SSLfinal_State, age_of_domain, Page_Rank that require crawling/WHOIS lookups; "
               "URLs themselves are not included, so features cannot be computed for a new message/screenshot.",
        verdict="Not suitable for the live app", used_for="Not used (documented only)",
        reason="Cannot be applied to arbitrary uploaded evidence without web crawling/external lookups, which are out of scope.")

# QR ----------------------------------------------------------------------------------------------
p = da.find_archive("qr")
if p:
    try:
        df = da.load_qr_urls()
        add(name="1000 QR Images of Malicious and Benign QR codes (2025)", archive=p.name, size_mb=mb(p), samples=len(df),
            classes=df.label.value_counts().to_dict(),
            structure="zip -> .rar -> benign_qr_images_500/*.png + malicious_qr_images_500/*.png (nested RAR; needs unrar-cffi/unrar/7-Zip)",
            issues=f"{int((df.url == '').sum())} images could not be decoded by OpenCV; labels come from folder names; benign URLs are mostly well-known "
                   "sites, malicious ones throw-away domains (sampling bias).",
            verdict="Suitable for QR-decoding tests and an EXPERIMENTAL URL-lexical hint model",
            used_for="Module 6 QR decoding test; URL lexical model shown beside URLs (not fused)",
            reason="Decoded payloads are URL strings with benign/malicious labels, but the model may learn dataset artefacts.")
    except da.DatasetMissing as e:
        add(name="QR codes dataset", archive=p.name, size_mb=mb(p), samples=0, classes={}, structure="nested rar", issues=str(e),
            verdict="Not readable here", used_for="-", reason=str(e))

config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
(config.EVAL_DIR / "dataset_inspection.json").write_text(json.dumps({"datasets": out}, indent=2, default=str))
print("wrote", config.EVAL_DIR / "dataset_inspection.json")
