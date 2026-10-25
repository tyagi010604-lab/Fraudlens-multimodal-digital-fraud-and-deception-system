"""FraudLens - Multimodal AI-Based Digital Fraud and Deception Detection System (Streamlit app).

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src import pipeline, report, text_ocr  # noqa: E402

st.set_page_config(page_title="FraudLens", page_icon="🔍", layout="wide")

PAGES = ["Home / Dashboard", "Analyse Evidence", "Individual Results", "Overall Risk Assessment",
         "Model Evaluation", "Dataset Information", "Report"]
ICON = {"ready": "✅ Ready", "experimental": "🧪 Experimental", "unavailable": "⛔ Unavailable",
        "not_assessed": "➖ Not assessed", "error": "⚠️ Error", "unknown": "❓ Unknown / Review",
        "experimental_untrained": "🧪 Experimental (untrained)", "rules_only": "📏 Rules only", "ok": "✅ OK",
        "no_text": "➖ No text found"}


# ----------------------------------------------------------------------------------- state -----
def init_state():
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("upload_round", 0)
    st.session_state.setdefault("page", PAGES[0])
    if "nav_request" in st.session_state:
        st.session_state["page"] = st.session_state.pop("nav_request")


def reset_session():
    """Clear results and all uploader widgets (media is never stored on disk by the app)."""
    st.session_state["analysis"] = None
    st.session_state["upload_round"] += 1
    for k in list(st.session_state.keys()):
        if k.startswith(("txt_", "trn_")):
            del st.session_state[k]


def label_box(label, status=None):
    if not label:
        return
    if label.startswith("Unknown") or status == "unknown":
        st.warning(f"**{label}**")
    elif label in ("Not assessed", "Unavailable"):
        st.info(f"**{label}**")
    elif any(w in label for w in ("Possible", "Spam-like", "Suspicious", "Mismatch", "flagged")):
        st.error(f"**{label}**")
    else:
        st.success(f"**{label}**")


def risk_color(cat):
    return {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(cat, "⚪")


# --------------------------------------------------------------------------------- pages --------
def page_home():
    st.title("🔍 FraudLens")
    st.subheader("Multimodal AI-Based Digital Fraud and Deception Detection")
    st.warning(config.DISCLAIMER)
    st.markdown(
        "FraudLens analyses **video, audio, images/screenshots and text** with nine separate, understandable modules, "
        "combines whatever evidence is available into a **preliminary** risk assessment, and flags unfamiliar cases "
        "for human review. Fake media and fraud are related but different: genuine media can be misused, and a "
        "fake-looking clip is not automatically fraud.")
    st.markdown("**How to use:** 1) *Analyse Evidence* → upload or paste what you have → *Run analysis*  "
                "2) read *Individual Results* and *Overall Risk Assessment*  3) download a *Report*. "
                "Missing evidence is shown as **Not assessed**, never as proof of fraud or of safety.")
    st.subheader("Module availability")
    ms = pipeline.module_status()
    df = pd.DataFrame([{"Module": m["module"], "Status": ICON.get(m["status"], m["status"]), "Details": m["detail"]} for m in ms])
    st.dataframe(df, hide_index=True)
    st.caption("Ready = trained/available and tested. Experimental = runs but trained on very limited data or uncalibrated. "
               "Unavailable = a dependency or model file is missing (see Details). "
               "Train models with `python scripts/train_models.py`.")


def page_analyse():
    st.header("Analyse Evidence")
    st.caption("Files are processed in a temporary folder that is deleted right after analysis. Nothing is stored by default.")
    r = st.session_state["upload_round"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Text evidence")
        text = st.text_area("Paste an SMS, email, job advert, chat message or call transcript", height=180, key=f"txt_text_{r}")
        with st.expander("Optional: transcript of the audio/video (used for consistency checks)"):
            transcript = st.text_area("Transcript", height=100, key=f"txt_trn_{r}")
        images = st.file_uploader("Images / screenshots (OCR + payment evidence)", type=[e[1:] for e in config.IMAGE_EXTS],
                                  accept_multiple_files=True, key=f"img_{r}")
    with c2:
        st.markdown("##### Media evidence")
        video = st.file_uploader("Video", type=[e[1:] for e in config.VIDEO_EXTS], key=f"vid_{r}")
        audio = st.file_uploader("Audio", type=[e[1:] for e in config.AUDIO_EXTS], key=f"aud_{r}")
        st.markdown("##### Reference samples (optional - enable identity checks)")
        st.caption("A *reference* is a sample you already trust of the person (a clear photo / a short clean voice clip). "
                   "It is compared with the **video / image / audio evidence above**. Without a reference these two checks stay 'Not assessed'. "
                   "Face check also needs `python scripts/download_face_models.py` once; voice clips need >= 2 s.")
        ref_face = st.file_uploader("Reference face image", type=[e[1:] for e in config.IMAGE_EXTS], key=f"rf_{r}")
        ref_voice = st.file_uploader("Reference voice recording", type=[e[1:] for e in config.AUDIO_EXTS], key=f"rv_{r}")
    o1, o2 = st.columns(2)
    ocr_frames = o1.checkbox("Also run OCR on 3 frames sampled from the video", value=True)
    extract_audio = o2.checkbox("Use the video's own audio track when no audio file is uploaded (needs ffmpeg)", value=True)

    have_any = bool(text.strip() or video or audio or images or transcript.strip())
    b1, b2, _ = st.columns([1, 1, 4])
    run = b1.button("Run analysis", type="primary", disabled=not have_any)
    if b2.button("Clear session"):
        reset_session()
        st.rerun()
    if not have_any:
        st.info("Provide at least one piece of evidence (text, image, audio or video) to enable analysis.")
    if run:
        inputs = {"text": text, "transcript": transcript,
                  "video": (video.name, video.getvalue()) if video else None,
                  "audio": (audio.name, audio.getvalue()) if audio else None,
                  "images": [(f.name, f.getvalue()) for f in images] if images else [],
                  "ref_face": (ref_face.name, ref_face.getvalue()) if ref_face else None,
                  "ref_voice": (ref_voice.name, ref_voice.getvalue()) if ref_voice else None}
        with st.spinner("Analysing evidence ..."):
            st.session_state["analysis"] = pipeline.run_analysis(inputs, ocr_frames, extract_audio)
        a = st.session_state["analysis"]
        for k, msg in a.get("input_errors", {}).items():
            st.error(f"{k}: {msg}")
        st.success(f"Analysis {a['analysis_id']} complete. Evidence analysed: {', '.join(a['evidence_types']) or 'none'}.")
        fu = a["fusion"]
        if fu["status"] == "assessed":
            st.metric("Preliminary overall risk", f"{risk_color(fu['category'])} {fu['category']}", f"score {fu['score']:.2f}")
        else:
            st.info("No usable score was produced - see Individual Results for what could not be assessed.")
        st.button("Open Individual Results →", on_click=lambda: st.session_state.update(nav_request="Individual Results"))


def render_indicators(ind):
    rows = []
    for u in ind["urls"]:
        rows.append({"Type": "URL", "Value": u["url"], "Notes": "; ".join(u["flags"]) or "no rule flags",
                     "Lexical URL score (experimental)": u.get("lexical_score")})
    for p in ind["phone_numbers"]:
        rows.append({"Type": "Phone number", "Value": p, "Notes": "", "Lexical URL score (experimental)": None})
    for u in ind["upi_ids"]:
        rows.append({"Type": "UPI ID", "Value": u["id"], "Notes": "known UPI handle" if u["known_handle"] else "unrecognised handle",
                     "Lexical URL score (experimental)": None})
    for e in ind["emails"]:
        rows.append({"Type": "E-mail", "Value": e, "Notes": "", "Lexical URL score (experimental)": None})
    for t in ind["payment_terms"]:
        rows.append({"Type": "Payment term", "Value": t, "Notes": "", "Lexical URL score (experimental)": None})
    for t in ind["amounts"]:
        rows.append({"Type": "Amount", "Value": t, "Notes": "", "Lexical URL score (experimental)": None})
    for t in ind["ifsc_codes"] + ind["account_strings"]:
        rows.append({"Type": "Bank detail", "Value": t, "Notes": "", "Lexical URL score (experimental)": None})
    if rows:
        df = pd.DataFrame(rows)
        if df["Lexical URL score (experimental)"].isna().all():
            df = df.drop(columns=["Lexical URL score (experimental)"])
        st.dataframe(df, hide_index=True)
        st.caption("Detecting a URL, phone number or UPI ID does **not** establish fraud - verify with the real organisation.")
    else:
        st.write("No URLs, phone numbers, UPI IDs or payment strings were detected.")


def render_text_analysis(ta, heading_prefix=""):
    m, rules = ta["model"], ta["rules"]
    asm = ta.get("assessment")
    if asm:
        label_box(asm["label"])
        st.caption(f"Basis: {asm['basis']}. {asm['explanation']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🤖 Model prediction** (TF-IDF classifier trained on SMS **spam**, not scams)")
        mi = m.get("model_info")
        if mi:
            st.caption(f"Model: {mi['model']} · trained {mi['trained_at'][:10]} · decision threshold {m.get('threshold')} · "
                       f"vocabulary coverage {m.get('vocab_coverage')}")
        if m.get("risk_score") is not None or m.get("label"):
            label_box(m.get("label"), "unknown" if m.get("unknown") else None)
        if m.get("risk_score") is not None:
            st.metric("Model risk score (uncalibrated)", f"{m['risk_score']:.2f}", help=f"decision threshold {m.get('threshold')}")
        for rsn in m.get("reasons", []) or ([m["reason"]] if m.get("reason") else []):
            st.caption("• " + rsn)
        for n in m.get("notes", []):
            st.caption("ℹ️ " + n)
    with c2:
        st.markdown("**📏 Rule-based warnings** (keyword/regex cues - not a model)")
        st.metric("Rule warning level", f"{rules.get('level', '')} ({rules['score']:.2f})",
                  help="'high' needs >=3 different cue categories at once; score = noisy-OR of small per-category weights, not a probability")
        if rules["categories"]:
            for c in rules["categories"]:
                st.markdown(f"- **{c['category'].replace('_', ' ')}** → {', '.join(f'`{x}`' for x in c['matches'][:5])}  \n  <small>{c['why']}</small>",
                            unsafe_allow_html=True)
        else:
            st.caption("No cue patterns matched.")
    if ta.get("_text") and rules["spans"]:
        st.markdown("**Highlighted cues**")
        st.markdown(text_ocr.highlight_html(ta["_text"][:3000], rules["spans"]), unsafe_allow_html=True)
    st.markdown("**Extracted indicators**")
    render_indicators(ta["indicators"])


def mel_plot(mel):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.imshow(mel.astype(float), origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("frames (10 ms)")
    ax.set_ylabel("Mel band")
    ax.set_title("Log-Mel spectrogram", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def status_tag(prefix):
    """Live module status for a heading (e.g. '1.') - taken from the actual model/dependency state, not hard-coded."""
    for m in pipeline.module_status():
        if m["module"].startswith(prefix):
            return ICON.get(m["status"], m["status"])
    return ""


def page_results():
    st.header("Individual Results")
    a = st.session_state["analysis"]
    if not a:
        st.info("No analysis yet. Go to **Analyse Evidence**.")
        return
    res = a["results"]
    st.caption(f"Analysis {a['analysis_id']} · {a['timestamp']} · evidence: {', '.join(a['evidence_types']) or 'none'}")

    st.subheader(f"5. Text scam detection  ·  {status_tag('5.')}")
    st.caption("The ML model detects **SMS spam**; scam/fraud cues (credential requests, account threats, links, payment demands) are checked by separate transparent rules.")
    if res.get("text"):
        render_text_analysis(res["text"])
    else:
        st.info("Not assessed - " + a["not_assessed"].get("5. Text scam detection", "no text"))

    st.subheader(f"6. OCR & payment evidence  ·  {status_tag('6.')}")
    if res["ocr"]:
        for i, o in enumerate(res["ocr"]):
            with st.expander(f"{o.get('source', 'image')} — {ICON.get(o['status'], o['status'])}", expanded=(i == 0)):
                if o["status"] in ("unavailable", "error", "no_text"):
                    st.warning(o.get("reason", ""))
                    continue
                if o.get("warning"):
                    st.warning(o["warning"])
                if o.get("ocr", {}).get("mean_confidence") is not None:
                    st.caption(f"Tesseract mean word confidence: {o['ocr']['mean_confidence']}% · {o['ocr'].get('n_words', 0)} words")
                st.text_area("Extracted text", o.get("extracted_text", ""), height=120, disabled=True, key=f"ocr_txt_{a['analysis_id']}_{i}")
                if o.get("qr_payload"):
                    st.info(f"QR code payload: `{o['qr_payload']}`")
                if o.get("text_analysis"):
                    render_text_analysis(o["text_analysis"])
    else:
        ok, msg = text_ocr.check_tesseract()
        st.info("Not assessed - " + a["not_assessed"].get("6. OCR & payment evidence", "no image uploaded"))
        if not ok:
            st.warning(msg)

    st.subheader(f"1. Video deepfake detection  ·  {status_tag('1.')}")
    v = res.get("video")
    if v:
        if v["status"] == "error":
            st.error(v["reason"])
        else:
            label_box(v.get("label"), "unknown" if v.get("unknown") else None)
            c = st.columns(4)
            c[0].metric("Manipulation score", "n/a" if v.get("risk_score") is None else f"{v['risk_score']:.2f}",
                        help="Raw classifier score, not a calibrated probability.")
            c[1].metric("Frames analysed", v["analysed_frames"])
            c[2].metric("Frames with a face", v["faces_found_frame"] if "faces_found_frame" in v else v["faces_found_frames"])
            c[3].metric("Duration (s)", v["video_meta"]["duration_s"])
            for r_ in v.get("reasons", []) or ([v["reason"]] if v.get("reason") else []):
                st.caption("• " + r_)
            if v.get("model_info"):
                st.caption("🧪 " + v["model_info"]["experimental_reason"])
            ti = v.get("temporal_instability_frames", [])
            if ti:
                st.markdown("**Frames with abnormal feature jumps**")
                st.dataframe(pd.DataFrame(ti), hide_index=True)
                st.caption(v["timestamp_note"])
            else:
                st.caption("No abnormal frame-to-frame feature jumps found. This tool does not locate manipulated frames or check lip-sync.")
    else:
        st.info("Not assessed - " + a["not_assessed"].get("1. Video deepfake detection", "no video"))

    st.subheader(f"2. AI-generated / cloned voice detection  ·  {status_tag('2.')}")
    au = res.get("audio")
    if au:
        if au["status"] == "error":
            st.error(au["reason"])
        else:
            if au["status"] == "experimental_untrained":
                st.warning("**Experimental — suitable labelled training data required.** No authenticity score is produced.")
            else:
                label_box(au.get("label"))
                st.metric("Synthetic-speech risk score (uncalibrated)", f"{au['risk_score']:.2f}")
            i = au["audio_info"]
            st.dataframe(pd.DataFrame([{"Source": au["source"], "Original sample rate": i["original_sample_rate"],
                                        "Channels": i["original_channels"], "Analysed duration (s)": i["analysed_duration_s"],
                                        "Resampled to (Hz)": i["analysis_sample_rate"], "Silence ratio": i["silence_ratio"],
                                        **au["audio_summary"]}]), hide_index=True)
            if au.get("_mel") is not None:
                mel_plot(au["_mel"])
            for s in au.get("scope_issues", []):
                st.caption("• " + s)
            if au.get("note"):
                st.caption(au["note"])
    else:
        st.info("Not assessed - " + a["not_assessed"].get("2. AI/cloned voice detection", "no audio"))

    st.subheader(f"3. Face identity verification  ·  {status_tag('3.')}")
    f = res.get("face")
    if f:
        label_box(f["label"])
        if f.get("similarity") is not None:
            st.metric("Similarity (cosine)", f["similarity"], help=f"threshold {f['threshold']}")
            st.caption(f"Frames with a detected face: {f['frames_with_face']}/{f['frames_checked']}")
        if f.get("reason"):
            st.caption(f["reason"])
        st.caption("⚠️ " + f["disclaimer"])
    else:
        st.info("Not assessed - " + a["not_assessed"].get("3. Face identity verification", ""))

    st.subheader(f"4. Voice identity verification  ·  {status_tag('4.')}")
    vo = res.get("voice")
    if vo:
        label_box(vo["label"], "unknown" if vo["label"].startswith("Unknown") else None)
        if vo.get("similarity") is not None:
            st.metric("Similarity (cosine)", vo["similarity"], help=f"threshold {vo['threshold']}")
        for r_ in vo.get("reasons", []) or ([vo["reason"]] if vo.get("reason") else []):
            st.caption("• " + r_)
        st.caption("⚠️ " + vo["disclaimer"])
    else:
        st.info("Not assessed - " + a["not_assessed"].get("4. Voice identity verification", ""))

    st.subheader("7. Audio-video consistency")
    c = res.get("consistency")
    if c and c["status"] != "not_assessed":
        label_box(c["label"])
        icon = {"passed": "✅", "flagged": "🚩", "not_assessed": "➖"}
        st.dataframe(pd.DataFrame([{"": icon[k["status"]], "Check": k["check"], "Result": k["status"], "Details": k["detail"]}
                                   for k in c["checks"]]), hide_index=True)
    else:
        st.info("Not assessed - " + (c or {}).get("reason", "insufficient evidence"))
        if c and c.get("checks"):
            for k in c["checks"]:
                st.caption(f"• {k['check']}: {k['detail']}")


def page_overall():
    st.header("Overall Risk Assessment")
    a = st.session_state["analysis"]
    if not a:
        st.info("No analysis yet. Go to **Analyse Evidence**.")
        return
    fu = a["fusion"]
    st.caption("Preliminary risk assessment — not proof of fraud.")
    if fu["status"] != "assessed":
        st.warning("**Not assessed.** " + fu["explanation"][0])
    else:
        c1, c2 = st.columns([1, 2])
        c1.metric("Overall risk", f"{risk_color(fu['category'])} {fu['category']}")
        c1.metric("Fusion score (0-1)", f"{fu['score']:.2f}")
        c2.progress(min(max(fu["score"], 0.0), 1.0))
        c2.caption(f"Low < {config.RISK_LOW_MAX} ≤ Medium < {config.RISK_HIGH_MIN} ≤ High")
        for line in fu["explanation"]:
            st.write("• " + line)
        st.subheader("Modules that contributed")
        df = pd.DataFrame(fu["contributions"])
        st.dataframe(df.rename(columns={"module": "Evidence group", "score": "Score", "weight_used": "Weight used",
                                        "share_of_total_weight": "Share of weight", "points": "Points toward total",
                                        "basis": "Basis"}), hide_index=True)
        st.bar_chart(df.set_index("module")["points"])
    if fu["excluded"]:
        st.subheader("Excluded (not counted as zero risk)")
        st.dataframe(pd.DataFrame(fu["excluded"]).rename(columns={"module": "Module", "reason": "Reason"}),
                     hide_index=True)
    if fu.get("unknown_flagged"):
        st.warning("**Unknown / Review:** " + ", ".join(fu["unknown_flagged"]) +
                   " produced an uncertain or out-of-scope result. This is not treated as fraud; please review manually.")
    with st.expander("How the fusion works / configured weights"):
        st.markdown(
            "Score = Σ(weight × module score) ÷ Σ(weights of modules that actually contributed). Unavailable, missing, "
            "untrained or Unknown-flagged modules are **excluded**, not set to zero. Experimental modules get "
            f"×{config.EXPERIMENTAL_FACTOR} weight. Pasted text and OCR text form **one** evidence group (max over distinct "
            f"texts; rule cues blended at {config.RULE_BLEND:.0%}) so the same evidence is not counted twice. "
            "Edit weights and thresholds in `config.py`.")
        st.json(fu["weights_configured"])
    st.warning(config.DISCLAIMER)


def load_json(name):
    p = config.EVAL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def metrics_table(rep, extra=None):
    row = {"n": rep["n"], "Precision": rep["precision"], "Recall": rep["recall"], "F1": rep["f1"],
           "Accuracy": rep.get("accuracy"), "ROC-AUC": rep.get("roc_auc"),
           "False positives": rep["false_positives"], "False negatives": rep["false_negatives"], "Threshold": rep.get("threshold")}
    row.update(extra or {})
    st.dataframe(pd.DataFrame([row]), hide_index=True)


def page_eval():
    st.header("Model Evaluation")
    st.caption("Results come from held-out data written by `scripts/train_models.py` and are kept separate from live predictions. "
               "Thresholds were tuned on validation data only.")
    tabs = st.tabs(["Text (SMS spam model)", "Video (experimental)", "Voice ID (experimental)", "URL lexical (experimental)", "Audio deepfake"])
    with tabs[0]:
        m = load_json("text_metrics.json")
        if not m:
            st.info("Not trained yet: `python scripts/train_models.py --only text`")
        else:
            i = m["info"]
            st.write(f"Selected model: **{m['selected']}** · train/val/test = {i['n_train']}/{i['n_val']}/{i['n_test']} "
                     f"(after removing {i['n_raw'] - i['n_after_dedup']} exact duplicates of {i['n_raw']})")
            st.markdown("**Held-out test set**")
            metrics_table(m["test"], {"Spam=positive": True})
            c1, c2 = st.columns(2)
            c1.image(str(config.EVAL_DIR / "text_confusion_test.png"), caption="Confusion matrix (test)")
            with c2:
                st.markdown("**Validation comparison of candidate models**")
                st.dataframe(pd.DataFrame({k: {x: v[x] for x in ("precision", "recall", "f1", "roc_auc", "false_positives", "false_negatives", "threshold")}
                                           for k, v in m["candidates_validation"].items()}).T)
                st.caption(f"Majority-class baseline on test: F1 {m['test_majority_baseline']['f1']}, "
                           f"accuracy {m['test_majority_baseline']['accuracy']}.")
            cv = m["cv_5fold_train_val"]
            st.write(f"Stratified 5-fold CV on train+val: F1 = {cv['mean_f1']} ± {cv['std_f1']}")
            uc = dict(m["unknown_calibration"])
            rel = uc.pop("reliability_oof", None)
            st.markdown("**Unknown / uncertain-case calibration**")
            st.write(f"Scores between **{uc['uncertain_low']}** and **{uc['uncertain_high']}** are returned as *Unknown / Review*. "
                     f"On {uc.get('oof_n', '?')} out-of-fold predictions this band holds {uc.get('oof_fraction_in_band', 0):.1%} of messages and "
                     f"{uc.get('oof_errors_inside_band', '?')} of the {uc.get('oof_errors_total', '?')} errors; accuracy is "
                     f"{uc.get('oof_accuracy_outside_band')} outside vs {uc.get('oof_accuracy_inside_band')} inside. "
                     f"Texts with < {uc['min_vocab_coverage']:.0%} known words are also Unknown.")
            st.caption(uc["calibration"])
            if rel:
                st.markdown("**Reliability of the raw score (out-of-fold, SMS domain only)** - the scores are *not* calibrated probabilities")
                st.dataframe(pd.DataFrame(rel), hide_index=True)
            st.warning("This is a spam classifier for SMS-style text. " + "; ".join(i["limitations"]))
    with tabs[1]:
        m = load_json("video_metrics.json")
        if not m:
            st.info("Not trained yet: `python scripts/train_models.py --only video`")
        else:
            st.warning("🧪 " + m["info"]["experimental_reason"])
            st.markdown(f"**Held-out TEST** (selected model: {m['selected']}; paired real/fake clips kept together)")
            t = m["test"]
            metrics_table(t, {"AUC 95% CI": "wide - n is tiny"})
            st.caption(f"Accuracy 95% Wilson interval: {t['accuracy_95ci_wilson']} (n={t['n']} clips)")
            c1, c2 = st.columns(2)
            c1.image(str(config.EVAL_DIR / "video_confusion_test.png"), caption="Test (held-out pairs)")
            c2.image(str(config.EVAL_DIR / "video_confusion_cv.png"), caption="Grouped 5-fold CV over all clips")
            st.markdown("**Candidate comparison (out-of-fold validation on training pairs)**")
            st.dataframe(pd.DataFrame({k: {x: v[x] for x in ("precision", "recall", "f1", "roc_auc", "false_positives", "false_negatives")}
                                       for k, v in m["candidates_validation_oof"].items()}).T)
            st.write("Grouped-CV (all clips):", {k: m["grouped_cv_all_clips"][k] for k in ("precision", "recall", "f1", "roc_auc", "accuracy")})
            st.warning("; ".join(m["info"]["limitations"]))
    with tabs[2]:
        m = load_json("voice_metrics.json")
        if not m:
            st.info("Not calibrated yet: `python scripts/train_models.py --only voice`")
        else:
            st.warning("🧪 " + m["info"]["experimental_reason"])
            t = m["test"]
            st.write(f"Scorer: **{t['scorer']}** · speaker-disjoint test pairs: {t['test_pairs']} ({t['same_speaker_pairs']} same-speaker)")
            st.dataframe(pd.DataFrame([{"AUC": t["test_auc"], "EER": t["test_eer"], "False-accept rate": t["false_accept_rate"],
                                        "False-reject rate": t["false_reject_rate"], "Precision": t["precision"],
                                        "Recall": t["recall"], "F1": t["f1"], "False positives": t["false_positives"],
                                        "False negatives": t["false_negatives"]}]), hide_index=True)
            st.image(str(config.EVAL_DIR / "voice_confusion_test.png"), caption="Pairs (test speakers unseen in training/validation)")
            st.dataframe(pd.DataFrame(m["validation_scorers"]).T)
    with tabs[3]:
        m = load_json("url_metrics.json")
        if not m:
            st.info("Not trained yet: `python scripts/train_models.py --only url`")
        else:
            st.warning("🧪 " + m["info"]["experimental_reason"])
            metrics_table(m["test"])
            st.image(str(config.EVAL_DIR / "url_confusion_test.png"))
    with tabs[4]:
        m = load_json("audio_metrics.json")
        if m and m.get("status") == "untrained":
            st.warning("**Experimental — suitable labelled training data required.** No audio-deepfake metrics exist and none are invented.")
            st.json(m["details"])
        elif m:
            metrics_table(m["test"])
            st.image(str(config.EVAL_DIR / "audio_confusion_test.png"))
        else:
            st.info("Run `python scripts/train_models.py --only audio` to record the audio-detector status.")


def page_datasets():
    st.header("Dataset Information")
    d = load_json("dataset_inspection.json")
    if not d:
        st.info("Run `python scripts/inspect_datasets.py` to generate the inspection.")
    else:
        df = pd.DataFrame([{"Dataset": x["name"], "Archive": x["archive"], "Samples": x["samples"], "Classes": x["classes"],
                            "Size (MB)": x["size_mb"], "Verdict": x["verdict"], "Used for": x["used_for"]} for x in d["datasets"]])
        st.dataframe(df, hide_index=True)
        for x in d["datasets"]:
            with st.expander(f"{x['name']} — {x['verdict']}"):
                st.write(x["structure"])
                st.write("**Issues / notes:**", x["issues"])
                st.write("**Why:**", x["reason"])
    p = config.BASE_DIR / "dataset_notes.md"
    if p.exists():
        with st.expander("Full dataset_notes.md"):
            st.markdown(p.read_text())


def page_report():
    st.header("Report")
    a = st.session_state["analysis"]
    if not a:
        st.info("No analysis yet. Run one first.")
        return
    mask = st.checkbox("Mask phone numbers, UPI IDs and e-mail addresses (recommended)", value=True)
    incl = st.checkbox("Include an excerpt of the pasted text", value=True)
    rep = report.build_report(a, mask_pii=mask, include_text=incl)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ JSON", report.to_json(rep), f"fraudlens_{a['analysis_id']}.json", "application/json")
    c2.download_button("⬇ CSV", report.to_csv(rep), f"fraudlens_{a['analysis_id']}.csv", "text/csv")
    c3.download_button("⬇ Text", report.to_text(rep), f"fraudlens_{a['analysis_id']}.txt", "text/plain")
    if st.button("Save a copy in outputs/reports/ (local disk)"):
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        p = config.REPORTS_DIR / f"fraudlens_{a['analysis_id']}.json"
        p.write_text(report.to_json(rep))
        st.success(f"Saved {p}")
    st.caption("Reports contain derived results only - no uploaded media. File names are not included.")
    st.text_area("Preview", report.to_text(rep), height=420)


# ------------------------------------------------------------------------------------ main -------
def main():
    init_state()
    with st.sidebar:
        st.title("🔍 FraudLens")
        st.radio("Navigate", PAGES, key="page")
        if st.button("🧹 Clear session"):
            reset_session()
            st.rerun()
        st.caption("Student research prototype — decision support only. Human verification required.")
    {"Home / Dashboard": page_home, "Analyse Evidence": page_analyse, "Individual Results": page_results,
     "Overall Risk Assessment": page_overall, "Model Evaluation": page_eval, "Dataset Information": page_datasets,
     "Report": page_report}[st.session_state["page"]]()
    st.divider()
    st.caption(config.DISCLAIMER)


main()
