"""Train the FraudLens models that the attached datasets can honestly support.

    python scripts/train_models.py --only text          # TF-IDF scam/spam classifier (SMS Spam Collection)
    python scripts/train_models.py --only video         # experimental video baseline (SDFVD, 106 clips)
    python scripts/train_models.py --only voice         # experimental speaker-verification calibration (DeepfakeTIMIT)
    python scripts/train_models.py --only url           # experimental URL lexical model (QR dataset URLs)
    python scripts/train_models.py --only audio         # records why no synthetic-voice detector can be trained
    python scripts/train_models.py                      # all of the above

Audio detector with your own labelled data (folders of real / synthetic speech):
    python scripts/train_models.py --only audio --audio-real path/real --audio-fake path/fake

Design rules followed here: separate train/validation/test; thresholds tuned on validation only;
leakage-aware splits (paired SDFVD clips and TIMIT speakers never straddle splits; exact duplicate
SMS removed); every metric written to outputs/eval/*.json; nothing is hard-coded.
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import data_access as da  # noqa: E402
from src import unknown_detection as ud  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, roc_auc_score,  # noqa: E402
                             roc_curve)

SEED = 42
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------ utilities --
def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def binary_report(y, p, thr, names=("negative", "positive")):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    acc = (tp + tn) / len(y)
    return {"threshold": round(float(thr), 4), "n": int(len(y)), "n_positive": int(y.sum()),
            "precision": round(float(pr), 4), "recall": round(float(rc), 4), "f1": round(float(f1), 4),
            "accuracy": round(float(acc), 4), "accuracy_95ci_wilson": wilson(int(tp + tn), len(y)),
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "labels": list(names),
            "false_positives": int(fp), "false_negatives": int(fn)}


def tune_threshold(y, p):
    """Threshold maximising F1 of the positive class on VALIDATION data (ties -> higher threshold)."""
    best_t, best_f = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (p >= t).astype(int)
        f = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)[2]
        if f >= best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def save_confusion(rep, title, path):
    cm = np.array(rep["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], rep["labels"], fontsize=8)
    ax.set_yticks([0, 1], rep["labels"], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_json(name, obj):
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_DIR / name).write_text(json.dumps(obj, indent=2, default=str))


def versions():
    import numpy
    out = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
           "pandas": pd.__version__}
    try:
        import xgboost
        out["xgboost"] = xgboost.__version__
    except Exception:
        pass
    return out


# ------------------------------------------------------------------------------- TEXT ------
def text_candidates():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    c = {"majority_baseline": DummyClassifier(strategy="prior"),
         "naive_bayes": MultinomialNB(alpha=0.1),
         "logreg": LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=SEED),
         "linear_svm_calibrated": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
                                                         cv=3, method="sigmoid")}
    try:
        from xgboost import XGBClassifier
        c["xgboost"] = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.1, subsample=0.9,
                                     colsample_bytree=0.5, eval_metric="logloss", random_state=SEED, n_jobs=2)
    except Exception:
        pass
    return c


def text_features():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=30000))])


def train_text():
    import re
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.pipeline import Pipeline
    from src.text_ocr import clean_for_model
    print("[text] loading SMS Spam Collection ...")
    raw = da.load_sms()
    raw["y"] = (raw["label"] == "spam").astype(int)      # label mapping: spam -> 1 (positive), ham -> 0
    raw["source"] = "sms_spam_collection"
    extra = da.load_extra_text()                          # optional, empty unless you add data/extra_text/*.csv
    if len(extra):
        print(f"[text] adding {len(extra)} extra labelled messages from data/extra_text/ ({int(extra.y.sum())} positive)")
        raw = pd.concat([raw[["text", "y", "source"]], extra[["text", "y", "source"]]], ignore_index=True)
    n_raw = len(raw)
    raw["key"] = raw["text"].str.lower().str.strip()
    conflicting = raw.groupby("key")["y"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    df = raw[~raw["key"].isin(conflicting)].drop_duplicates("key").reset_index(drop=True)   # leakage guard
    df["clean"] = df["text"].map(clean_for_model)
    print(f"[text] raw={n_raw}  after exact-duplicate removal={len(df)}  (conflicting-label texts dropped: {len(conflicting)})")
    idx = np.arange(len(df))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=df.y, random_state=SEED)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=df.y.iloc[tmp_idx], random_state=SEED)
    Xtr, ytr = df.clean.iloc[tr_idx], df.y.iloc[tr_idx].values
    Xva, yva = df.clean.iloc[va_idx], df.y.iloc[va_idx].values
    Xte, yte = df.clean.iloc[te_idx], df.y.iloc[te_idx].values

    cand_report = {}
    fitted = {}
    for name, clf in text_candidates().items():
        pipe = Pipeline([("features", text_features()), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        pv = pipe.predict_proba(Xva)[:, 1]
        thr, f1 = tune_threshold(yva, pv) if name != "majority_baseline" else (0.5, 0.0)
        cand_report[name] = binary_report(yva, pv, thr, ("ham", "spam"))
        fitted[name] = (pipe, thr, pv)
        print(f"[text] {name:24s} val F1={cand_report[name]['f1']:.3f} P={cand_report[name]['precision']:.3f} "
              f"R={cand_report[name]['recall']:.3f} thr={thr:.2f}")
    best = max((n for n in fitted if n != "majority_baseline"), key=lambda n: cand_report[n]["f1"])
    pipe, thr, pv = fitted[best]
    print(f"[text] selected on validation: {best} (threshold {thr:.2f})")
    test_rep = binary_report(yte, pipe.predict_proba(Xte)[:, 1], thr, ("ham", "spam"))
    base_rep = binary_report(yte, fitted["majority_baseline"][0].predict_proba(Xte)[:, 1], 0.5, ("ham", "spam"))
    print(f"[text] TEST  F1={test_rep['f1']:.3f} P={test_rep['precision']:.3f} R={test_rep['recall']:.3f} "
          f"FP={test_rep['false_positives']} FN={test_rep['false_negatives']}")

    # 5-fold stratified CV on train+val (model re-fitted per fold, same hyper-parameters, fixed threshold)
    tv_idx = np.concatenate([tr_idx, va_idx])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    from sklearn.base import clone
    cvp = cross_val_predict(clone(pipe), df.clean.iloc[tv_idx], df.y.iloc[tv_idx].values, cv=skf, method="predict_proba")[:, 1]
    folds = []
    for _, f_idx in skf.split(tv_idx, df.y.iloc[tv_idx].values):
        r = binary_report(df.y.iloc[tv_idx].values[f_idx], cvp[f_idx], thr)
        folds.append(r["f1"])
    cv_summary = {"folds_f1": folds, "mean_f1": round(float(np.mean(folds)), 4), "std_f1": round(float(np.std(folds)), 4)}

    # unknown-case calibration (validation only)
    word_vocab = set(w for t in Xtr for w in re.findall(r"[a-z']+", t))
    covs = [np.mean([w in word_vocab for w in re.findall(r"[a-z']+", t)] or [1.0]) for t in Xva]
    min_cov = float(np.percentile(covs, 1))               # no artificial floor (an earlier version floored it at 0.5)
    # uncertain band: class-overlap region of OUT-OF-FOLD scores on train+val (~4000 held-out predictions, not the
    # 774-message validation split, which has too few errors to locate the overlap)
    ob = ud.overlap_band(df.y.iloc[tv_idx].values, cvp, thr, q=0.02)
    lo, hi = ob["low"], ob["high"]
    unknown = {"min_vocab_coverage": round(min_cov, 3), "uncertain_low": round(lo, 3), "uncertain_high": round(hi, 3),
               "calibration": "uncertain band = [2nd percentile of spam scores, 98th percentile of ham scores] over 5-fold out-of-fold "
                              "predictions on train+val (class-overlap region); min_vocab_coverage = 1st percentile of validation "
                              "word coverage. Scores are NOT calibrated probabilities.",
               "oof_n": ob["n"], "oof_fraction_in_band": round(ob["fraction_inside"], 3),
               "oof_accuracy_outside_band": round(ob["accuracy_outside"], 4), "oof_accuracy_inside_band": None if ob["accuracy_inside"] is None else round(ob["accuracy_inside"], 4),
               "oof_errors_total": ob["errors_total"], "oof_errors_inside_band": ob["errors_inside"]}
    # reliability of the raw scores (out-of-fold): is a score of 0.8 really ~80 % spam?  (it is not -> not calibrated)
    bins = np.linspace(0, 1, 11)
    yb, rel = df.y.iloc[tv_idx].values, []
    for a_, b_ in zip(bins[:-1], bins[1:]):
        m_ = (cvp >= a_) & ((cvp < b_) if b_ < 1 else (cvp <= b_))
        if m_.sum():
            rel.append({"score_bin": f"{a_:.1f}-{b_:.1f}", "n": int(m_.sum()), "mean_score": round(float(cvp[m_].mean()), 3),
                        "observed_spam_rate": round(float(yb[m_].mean()), 3)})
    unknown["reliability_oof"] = rel

    info = {"module": "text_scam_detection", "model": best, "trained_at": now(), "dataset": "SMS Spam Collection v.1 (UCI)",
            "n_raw": n_raw, "n_after_dedup": len(df), "n_extra_labelled": int(len(extra)), "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
            "class_counts_dedup": {"ham": int((df.y == 0).sum()), "spam": int((df.y == 1).sum())},
            "features": "word 1-2gram TF-IDF + char_wb 2-5gram TF-IDF (fit on training split only)",
            "threshold_selected_on": "validation (max F1 of spam class)", "versions": versions(), "experimental": False,
            "task": "SPAM detection on SMS (not general scam/fraud detection)",
            "limitations": ["Trained on 2011-era UK/Singapore English SMS only: 'spam' (mostly promotions/premium-rate) "
                            "is not the same as 'scam'. Banking/KYC/OTP/UPI phishing is outside the training data and is NOT reliably "
                            "scored by the model - the rule-based cues exist to cover that gap.",
                            "Transactional notices (delivery/order/bank alerts) are neither personal chat nor promotional spam in the "
                            "training data and receive middling scores -> often Unknown/Review.",
                            "Emails, job adverts, call transcripts and non-English text are a domain shift (handled by "
                            "the Unknown/Review logic, not solved).",
                            "No Indian-context scam data (UPI/KYC/OTP) in the training set."]}
    bundle = {"pipeline": pipe, "threshold": thr, "unknown": unknown, "word_vocab": word_vocab, "info": info}
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.TEXT_MODEL_PATH)
    save_confusion(test_rep, f"Text model - TEST ({best})", config.EVAL_DIR / "text_confusion_test.png")
    write_json("text_metrics.json", {"info": info, "candidates_validation": cand_report, "selected": best,
                                     "test": test_rep, "test_majority_baseline": base_rep, "cv_5fold_train_val": cv_summary,
                                     "unknown_calibration": unknown, "split": {"test_indices": [int(i) for i in te_idx],
                                                                                "val_indices": [int(i) for i in va_idx],
                                                                                "train_indices": [int(i) for i in tr_idx]}})
    print("[text] saved", config.TEXT_MODEL_PATH)


# ------------------------------------------------------------------------------- VIDEO -----
def _video_feat(path):
    from src import preprocess_video as pv
    try:
        r = pv.extract_video_features(path)
        return path, r["vector"], r["names"], None
    except Exception as e:
        return path, None, None, f"{type(e).__name__}: {e}"


def video_feature_table():
    from joblib import Parallel, delayed
    df = da.ensure_sdfvd()
    cache = config.CACHE_DIR / "sdfvd_features.joblib"
    store = joblib.load(cache) if cache.exists() else {}
    todo = [p for p in df.path if p not in store]
    if todo:
        print(f"[video] extracting features for {len(todo)} clips (~3 s each, parallel) ...", flush=True)
        res = Parallel(n_jobs=min(4, os.cpu_count() or 1))(delayed(_video_feat)(p) for p in todo)
        for p, v, names, err in res:
            store[p] = {"vector": v, "names": names, "error": err}
        joblib.dump(store, cache)
    bad = [p for p in df.path if store[p]["vector"] is None]
    if bad:
        print(f"[video] WARNING: {len(bad)} unreadable clips skipped: {[Path(b).name for b in bad]}")
    df = df[~df.path.isin(bad)].reset_index(drop=True)
    X = np.vstack([store[p]["vector"] for p in df.path])
    names = store[df.path.iloc[0]]["names"]
    return df, X, names, bad


def video_candidates():
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    c = {"majority_baseline": DummyClassifier(strategy="prior"),
         "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000, random_state=SEED)),
         "svm_rbf": make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)),
         "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=2)}
    try:
        from xgboost import XGBClassifier
        c["xgboost"] = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8,
                                     colsample_bytree=0.6, eval_metric="logloss", random_state=SEED, n_jobs=2)
    except Exception:
        pass
    return c


def train_video():
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold
    df, X, names, bad = video_feature_table()
    y = (df.label == "fake").astype(int).values
    groups = df.pair_id.values
    pairs = sorted(set(groups))
    print(f"[video] clips={len(df)} real={int((y == 0).sum())} fake={int(y.sum())} pairs={len(pairs)}")
    rng = np.random.RandomState(SEED)
    test_pairs = set(rng.choice(pairs, size=max(2, round(0.2 * len(pairs))), replace=False))
    te = np.array([g in test_pairs for g in groups])
    tr = ~te
    print(f"[video] train clips={int(tr.sum())} (pairs {len(set(groups[tr]))}); held-out TEST clips={int(te.sum())} (pairs {len(test_pairs)})")
    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    gkf = GroupKFold(n_splits=5)

    cand_report, oof_all = {}, {}
    for name, clf in video_candidates().items():
        oof = np.zeros(len(ytr))
        for a, b in gkf.split(Xtr, ytr, gtr):
            m = clone(clf).fit(Xtr[a], ytr[a])
            oof[b] = m.predict_proba(Xtr[b])[:, 1]
        thr, _ = tune_threshold(ytr, oof) if name != "majority_baseline" else (0.5, 0)
        cand_report[name] = binary_report(ytr, oof, thr, ("real", "fake"))
        oof_all[name] = (oof, thr)
        print(f"[video] {name:18s} grouped-CV(val) AUC={cand_report[name]['roc_auc']:.3f} F1={cand_report[name]['f1']:.3f} thr={thr:.2f}")
    best = max((n for n in oof_all if n != "majority_baseline"), key=lambda n: cand_report[n]["roc_auc"])
    oof, thr = oof_all[best]
    model = clone(video_candidates()[best]).fit(Xtr, ytr)
    pt = model.predict_proba(X[te])[:, 1]
    test_rep = binary_report(y[te], pt, thr, ("real", "fake"))
    base = video_candidates()["majority_baseline"].fit(Xtr, ytr)
    base_rep = binary_report(y[te], base.predict_proba(X[te])[:, 1], 0.5, ("real", "fake"))
    print(f"[video] selected {best}; TEST AUC={test_rep['roc_auc']} F1={test_rep['f1']:.3f} FP={test_rep['false_positives']} FN={test_rep['false_negatives']} (n={test_rep['n']})")

    # grouped 5-fold CV over ALL clips with the selected config (supplementary stability estimate)
    oof_full = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=5).split(X, y, groups):
        oof_full[b] = clone(video_candidates()[best]).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    cv_rep = binary_report(y, oof_full, thr, ("real", "fake"))

    # OOD calibration (validation = out-of-fold samples)
    dists = []
    for a, b in gkf.split(Xtr, ytr, gtr):
        o = ud.FeatureOOD(k=3).fit(Xtr[a])
        dists += [o.distance(x) for x in Xtr[b]]
    ood = ud.FeatureOOD(k=3).fit(Xtr)
    ood.threshold, ood.percentile = float(np.percentile(dists, 95)), 95
    lo, hi, w = ud.uncertain_band(ytr, oof, thr, target_acc=0.85, max_half_width=0.25)
    unknown = {"ood_percentile": 95, "ood_threshold": round(ood.threshold, 3), "uncertain_low": round(lo, 3),
               "uncertain_high": round(hi, 3)}
    info = {"module": "video_deepfake_detection", "model": best, "trained_at": now(),
            "dataset": "SDFVD small-scale deepfake forgery video dataset", "n_clips": len(df), "n_pairs": len(pairs),
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "frames_per_clip": config.VIDEO_N_FRAMES,
            "feature_names": names, "threshold_selected_on": "out-of-fold predictions on training pairs (grouped 5-fold)",
            "versions": versions(), "experimental": True,
            "experimental_reason": "Only 106 clips (53 real/53 fake) from one small dataset; test set has "
                                   f"{int(te.sum())} clips so every metric has a very wide confidence interval.",
            "limitations": ["Classic image statistics, not a deep network; may latch onto dataset-specific artefacts "
                            "(compression, resolution, source) rather than manipulation.",
                            "Clip-level prediction only: no localisation of manipulated frames and no lip-sync analysis.",
                            "Fake clips are derived from the paired real clip; pairs were kept together in every split."]}
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_names": names, "threshold": thr, "ood": ood, "unknown": unknown, "info": info},
                config.VIDEO_MODEL_PATH)
    save_confusion(test_rep, f"Video baseline - held-out TEST ({best})", config.EVAL_DIR / "video_confusion_test.png")
    save_confusion(cv_rep, "Video baseline - grouped 5-fold CV (all clips)", config.EVAL_DIR / "video_confusion_cv.png")
    write_json("video_metrics.json", {"info": {k: v for k, v in info.items() if k != "feature_names"}, "selected": best,
                                      "candidates_validation_oof": cand_report, "test": test_rep,
                                      "test_majority_baseline": base_rep, "grouped_cv_all_clips": cv_rep,
                                      "unknown_calibration": unknown, "skipped_unreadable": [Path(b).name for b in bad],
                                      "split": {"test_pairs": sorted(int(p) for p in test_pairs)}})
    print("[video] saved", config.VIDEO_MODEL_PATH)


# ------------------------------------------------------------------- VOICE (speaker ID) -----
def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def train_voice():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from src import preprocess_audio as pa
    meta = da.ensure_timit_audio()
    print(f"[voice] DeepfakeTIMIT audio: {len(meta)} utterances, {meta.speaker.nunique()} speakers")
    cache = config.CACHE_DIR / "timit_speaker_feats.joblib"
    feats = joblib.load(cache) if cache.exists() else {}
    for p in meta.path:
        if p not in feats:
            y, sr, _ = pa.load_audio(p, check_ext=False)
            feats[p] = pa.speaker_features(y, sr)
    joblib.dump(feats, cache)
    X = np.vstack([feats[p] for p in meta.path])
    spk = meta.speaker.values
    speakers = sorted(set(spk))
    fem = [s for s in speakers if s.startswith("f")]
    mal = [s for s in speakers if s.startswith("m")]
    rng = np.random.RandomState(SEED)
    rng.shuffle(fem), rng.shuffle(mal)

    def split(lst):
        a, b = round(len(lst) * 0.5), round(len(lst) * 0.75)
        return lst[:a], lst[a:b], lst[b:]
    f_tr, f_va, f_te = split(fem)
    m_tr, m_va, m_te = split(mal)
    S_tr, S_va, S_te = set(f_tr + m_tr), set(f_va + m_va), set(f_te + m_te)
    print(f"[voice] speaker-disjoint split: train={len(S_tr)} val={len(S_va)} test={len(S_te)} speakers")

    def pairs(S, rng):
        idx = [i for i, s in enumerate(spk) if s in S]
        same = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:] if spk[a] == spk[b]]
        diff = []
        while len(diff) < len(same):
            a, b = rng.choice(idx, 2, replace=False)
            if spk[a] != spk[b]:
                diff.append((a, b))
        P = same + diff
        return P, np.array([1] * len(same) + [0] * len(diff))

    P_va, y_va = pairs(S_va, np.random.RandomState(1))
    P_te, y_te = pairs(S_te, np.random.RandomState(2))
    tr_idx = [i for i, s in enumerate(spk) if s in S_tr]
    scaler = StandardScaler().fit(X[tr_idx])
    lda = LinearDiscriminantAnalysis(n_components=min(15, len(S_tr) - 1)).fit(scaler.transform(X[tr_idx]), spk[tr_idx])

    def embed(kind, Z):
        if kind == "raw_cosine":
            return Z
        Zs = scaler.transform(Z)
        return Zs if kind == "standardised_cosine" else lda.transform(Zs)

    res = {}
    for kind in ("raw_cosine", "standardised_cosine", "lda_cosine"):
        E = embed(kind, X)
        s_va = np.array([_cosine(E[a], E[b]) for a, b in P_va])
        eer, thr = _eer(y_va, s_va)
        res[kind] = {"val_eer": round(eer, 4), "val_threshold": round(thr, 4)}
        print(f"[voice] {kind:20s} validation EER={eer:.3f}")
    best = min(res, key=lambda k: res[k]["val_eer"])
    thr = res[best]["val_threshold"]
    E = embed(best, X)
    s_te = np.array([_cosine(E[a], E[b]) for a, b in P_te])
    pred = (s_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    test_eer, _ = _eer(y_te, s_te)
    rep = {"scorer": best, "threshold_from_validation": thr, "test_pairs": int(len(y_te)), "same_speaker_pairs": int(y_te.sum()),
           "test_auc": round(float(roc_auc_score(y_te, s_te)), 4), "test_eer": round(test_eer, 4),
           "false_accept_rate": round(float(fp / max(fp + tn, 1)), 4), "false_reject_rate": round(float(fn / max(fn + tp, 1)), 4),
           "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "labels": ["different speaker", "same speaker"],
           "false_positives": int(fp), "false_negatives": int(fn), "precision": round(float(tp / max(tp + fp, 1)), 4),
           "recall": round(float(tp / max(tp + fn, 1)), 4)}
    rep["f1"] = round(2 * rep["precision"] * rep["recall"] / max(rep["precision"] + rep["recall"], 1e-9), 4)
    print(f"[voice] TEST (unseen speakers) EER={test_eer:.3f} AUC={rep['test_auc']} FAR={rep['false_accept_rate']} FRR={rep['false_reject_rate']}")
    save_confusion(rep, f"Voice ID ({best}) - TEST pairs", config.EVAL_DIR / "voice_confusion_test.png")
    info = {"module": "voice_identity_verification", "trained_at": now(), "dataset": "DeepfakeTIMIT (original VidTIMIT audio)",
            "n_utterances": len(meta), "n_speakers": len(speakers), "split_speakers": [len(S_tr), len(S_va), len(S_te)],
            "scorer": best, "versions": versions(), "experimental": True,
            "experimental_reason": "Calibrated on 32 clean studio speakers; real-world recordings (phone codecs, noise, "
                                   "emotion, different sessions) will degrade accuracy substantially.",
            "limitations": ["Handcrafted MFCC statistics, not x-vector/ECAPA embeddings.",
                            "Does NOT detect synthetic speech: cloned voices can be 'Match' and genuine voices 'Mismatch'.",
                            "HQ and LQ copies of the audio are byte-identical; only one copy was used."]}
    joblib.dump({"kind": best, "scaler": scaler, "lda": lda, "threshold": thr, "info": info}, config.VOICE_ID_PATH)
    write_json("voice_metrics.json", {"info": info, "validation_scorers": res, "test": rep})
    print("[voice] saved", config.VOICE_ID_PATH)


# ---------------------------------------------------------------------------- URL (QR) ------
def train_url():
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from src.text_ocr import url_normalise
    df = da.load_qr_urls()
    n_all = len(df)
    df = df[df.url.str.len() > 0].copy()
    df["y"] = (df.label == "malicious").astype(int)
    df["norm"] = df.url.map(url_normalise)
    df = df.drop_duplicates("norm").reset_index(drop=True)

    def dom(u):
        h = re.sub(r"^https?://", "", u).split("/")[0].split(":")[0].split(".")
        return ".".join(h[-2:])
    df["group"] = df.url.str.lower().map(dom)
    print(f"[url] QR images={n_all} decoded+unique URLs={len(df)} (malicious={int(df.y.sum())}) domains={df.group.nunique()}")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, tmp = next(gss.split(df, df.y, df.group))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_rel, te_rel = next(gss2.split(df.iloc[tmp], df.y.iloc[tmp], df.group.iloc[tmp]))
    va, te = tmp[va_rel], tmp[te_rel]
    pipe = Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                     ("clf", LogisticRegression(C=3, class_weight="balanced", max_iter=3000, random_state=SEED))])
    pipe.fit(df.norm.iloc[tr], df.y.iloc[tr])
    pv = pipe.predict_proba(df.norm.iloc[va])[:, 1]
    thr, _ = tune_threshold(df.y.iloc[va].values, pv)
    rep = binary_report(df.y.iloc[te].values, pipe.predict_proba(df.norm.iloc[te])[:, 1], thr, ("benign", "malicious"))
    print(f"[url] TEST F1={rep['f1']:.3f} AUC={rep['roc_auc']} FP={rep['false_positives']} FN={rep['false_negatives']} (n={rep['n']})")
    info = {"module": "url_lexical_model", "trained_at": now(), "dataset": "Dataset of 1000 Images of Malicious and Benign QR codes (2025), URLs decoded from QR images",
            "n_urls": len(df), "n_train": len(tr), "n_val": len(va), "n_test": len(te), "versions": versions(), "experimental": True,
            "experimental_reason": "Benign URLs in this set are mostly well-known sites while malicious ones are throw-away domains; "
                                   "the model may learn that sampling artefact rather than general URL maliciousness.",
            "limitations": ["Shown as an extra hint next to URLs; NOT used in the overall risk fusion."]}
    joblib.dump({"pipeline": pipe, "threshold": thr, "info": info}, config.URL_MODEL_PATH)
    save_confusion(rep, "URL lexical model - TEST", config.EVAL_DIR / "url_confusion_test.png")
    write_json("url_metrics.json", {"info": info, "test": rep, "split": "domain-grouped (no domain in two splits)"})
    print("[url] saved", config.URL_MODEL_PATH)


# ------------------------------------------------------------------------------ AUDIO ------
def train_audio(real_dir=None, fake_dir=None):
    """Synthetic-voice detector. Trains ONLY from user-supplied labelled folders; otherwise records why not."""
    if not (real_dir and fake_dir):
        uy = da.ensure_uyghur()
        n_b, n_s = int((uy.label == "bonafide").sum()), int((uy.label == "spoof").sum())
        status = {"status": "untrained", "checked_at": now(),
                  "message": "Experimental - suitable labelled training data required.",
                  "datasets_checked": {
                      "UyghurDF preview": {"bonafide": n_b, "spoof": n_s,
                                           "verdict": "Not used for training/evaluation: its README states it is a listening sample, "
                                                      "and 3 bona-fide clips cannot support any detector."},
                      "DeepfakeTIMIT audio": {"verdict": "Original, unmanipulated human speech only (README: audio track kept "
                                                         "unmodified) - no synthetic class, so unusable for real-vs-synthetic training."},
                      "SDFVD": {"verdict": "Silent/no labelled audio use."}},
                  "how_to_enable": "Provide a labelled real-vs-synthetic speech set (e.g. ASVspoof 2019 LA, WaveFake, "
                                   "In-the-Wild) and run: python scripts/train_models.py --only audio --audio-real DIR --audio-fake DIR"}
        config.MODELS_DIR.mkdir(exist_ok=True)
        config.AUDIO_STATUS_PATH.write_text(json.dumps(status, indent=2))
        if config.AUDIO_MODEL_PATH.exists():
            config.AUDIO_MODEL_PATH.unlink()
        write_json("audio_metrics.json", {"status": "untrained", "reason": status["message"], "details": status["datasets_checked"]})
        print("[audio] No suitable labelled real-vs-synthetic dataset among the attached files -> detector left UNTRAINED.")
        print("[audio]", status["message"])
        return
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src import preprocess_audio as pa
    files = [(p, 0) for p in sorted(Path(real_dir).rglob("*")) if p.suffix.lower() in config.AUDIO_EXTS] + \
            [(p, 1) for p in sorted(Path(fake_dir).rglob("*")) if p.suffix.lower() in config.AUDIO_EXTS]
    n0, n1 = sum(1 for _, l in files if l == 0), sum(1 for _, l in files if l == 1)
    print(f"[audio] user-supplied data: real={n0} fake={n1}")
    if min(n0, n1) < 20:
        print("[audio] Fewer than 20 clips per class: refusing to train a detector.")
        return
    X, y, names = [], [], []
    for p, l in files:
        try:
            sig, sr, _ = pa.load_audio(p)
            X.append(pa.extract_features(sig, sr))
            y.append(l)
            names.append(str(p))
        except Exception as e:
            print("[audio] skipped", p.name, e)
    X, y = np.array(X), np.array(y)
    print("[audio] NOTE: random stratified split - if clips share speakers/sources, results are optimistic.")
    tr, tmp = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
    cands = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")),
             "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=2)}
    rep_c = {}
    for n, c in cands.items():
        m = clone(c).fit(X[tr], y[tr])
        pv = m.predict_proba(X[va])[:, 1]
        t, _ = tune_threshold(y[va], pv)
        rep_c[n] = (m, t, binary_report(y[va], pv, t, ("real", "synthetic")))
    best = max(rep_c, key=lambda k: rep_c[k][2]["f1"])
    m, t, _ = rep_c[best]
    rep = binary_report(y[te], m.predict_proba(X[te])[:, 1], t, ("real", "synthetic"))
    exp = len(y) < 1000
    info = {"module": "audio_deepfake_detection", "model": best, "trained_at": now(), "n_train": len(tr), "n_val": len(va),
            "n_test": len(te), "versions": versions(), "experimental": exp, "feature_dim": X.shape[1],
            "dataset": "user-supplied folders", "limitations": ["Speaker/source leakage not controlled by this generic script."]}
    joblib.dump({"model": m, "threshold": t, "info": info, "unknown": {}}, config.AUDIO_MODEL_PATH)
    config.AUDIO_STATUS_PATH.write_text(json.dumps({"status": "trained", "message": "Trained on user-supplied data."}, indent=2))
    save_confusion(rep, "Audio detector - TEST", config.EVAL_DIR / "audio_confusion_test.png")
    write_json("audio_metrics.json", {"info": info, "test": rep})
    print("[audio] TEST", {k: rep[k] for k in ("precision", "recall", "f1", "false_positives", "false_negatives")})


# -------------------------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["text", "video", "voice", "url", "audio", "all"], default="all")
    ap.add_argument("--audio-real")
    ap.add_argument("--audio-fake")
    a = ap.parse_args()
    steps = {"text": train_text, "video": train_video, "voice": train_voice, "url": train_url,
             "audio": lambda: train_audio(a.audio_real, a.audio_fake)}
    for name in (steps if a.only == "all" else [a.only]):
        print(f"\n===== {name.upper()} =====")
        try:
            steps[name]()
        except da.DatasetMissing as e:
            print(f"[{name}] SKIPPED - {e}")
    print("\nDone. Metrics are in outputs/eval/, models in models/.")


if __name__ == "__main__":
    main()
