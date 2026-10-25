"""Re-evaluate the SAVED models on their persisted held-out splits and (re)draw evaluation figures.

    python scripts/evaluate_models.py

Keeps evaluation separate from live prediction: uses the split recorded by train_models.py (never re-splits),
writes outputs/eval/evaluation_summary.json. A difference from the training-time numbers would signal a
reproducibility problem (e.g. different library versions).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.train_models import binary_report, save_confusion, video_feature_table  # noqa: E402
from src import data_access as da  # noqa: E402
from src.text_ocr import clean_for_model  # noqa: E402

summary = {}
E = config.EVAL_DIR

# text
if config.TEXT_MODEL_PATH.exists() and (E / "text_metrics.json").exists():
    meta = json.loads((E / "text_metrics.json").read_text())
    raw = da.load_sms()
    raw["y"] = (raw.label == "spam").astype(int)
    raw["key"] = raw.text.str.lower().str.strip()
    df = raw.drop_duplicates("key").reset_index(drop=True)
    te = meta["split"]["test_indices"]
    b = joblib.load(config.TEXT_MODEL_PATH)
    p = b["pipeline"].predict_proba(df.text.iloc[te].map(clean_for_model))[:, 1]
    rep = binary_report(df.y.iloc[te].values, p, b["threshold"], ("ham", "spam"))
    save_confusion(rep, "Text model - TEST (re-evaluated)", E / "text_confusion_test.png")
    summary["text"] = {"re_evaluated": rep, "matches_training_run": rep["f1"] == meta["test"]["f1"]}
    print("text  F1", rep["f1"], "FP", rep["false_positives"], "FN", rep["false_negatives"], "| same as training run:", summary["text"]["matches_training_run"])

# video
if config.VIDEO_MODEL_PATH.exists() and (E / "video_metrics.json").exists():
    meta = json.loads((E / "video_metrics.json").read_text())
    df, X, names, _ = video_feature_table()
    te = df.pair_id.isin(meta["split"]["test_pairs"]).values
    b = joblib.load(config.VIDEO_MODEL_PATH)
    y = (df.label == "fake").astype(int).values
    rep = binary_report(y[te], b["model"].predict_proba(X[te])[:, 1], b["threshold"], ("real", "fake"))
    save_confusion(rep, "Video baseline - TEST (re-evaluated)", E / "video_confusion_test.png")
    summary["video"] = {"re_evaluated": rep, "matches_training_run": rep["f1"] == meta["test"]["f1"]}
    print("video F1", rep["f1"], "AUC", rep["roc_auc"], "n", rep["n"], "| same as training run:", summary["video"]["matches_training_run"])

for k in ("voice", "url", "audio"):
    f = E / f"{k}_metrics.json"
    if f.exists():
        summary[k] = {"recorded": json.loads(f.read_text()).get("test") or json.loads(f.read_text()).get("status")}
        print(k, "recorded metrics found (see outputs/eval/%s_metrics.json)" % k)
(E / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
