"""Module 8: multimodal risk fusion - a documented weighted-average baseline.

score = sum(w_i * s_i) / sum(w_i)   over the evidence groups that actually produced a usable score.

Rules
  * Missing / failed / untrained / Unknown-flagged modules are EXCLUDED (never counted as 0 risk).
  * Experimental modules have weight x config.EXPERIMENTAL_FACTOR.
  * No double counting: pasted text and OCR text are one group ("message_content") scored with the MAX over
    distinct texts; rule-based cues are blended into that same group (RULE_BLEND) instead of being a separate vote.
  * The result is a preliminary risk assessment, never proof of fraud.
No XGBoost fusion is trained: there is no labelled integration dataset with module outputs + overall labels.
"""
import re

import config


def categorise(score):
    if score < config.RISK_LOW_MAX:
        return "Low"
    return "High" if score >= config.RISK_HIGH_MIN else "Medium"


def _norm_text(t):
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def _same_text(a, b):
    """Exact (normalised) match, or >=80 % word overlap - OCR of a screenshot of pasted text is rarely byte-identical."""
    if a == b:
        return True
    wa, wb = set(a.split()), set(b.split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.8


def _text_group(text_results):
    """Return the message_content group: the MAX over distinct texts (never a sum), plus notes."""
    seen, cands, notes = [], [], []
    for src, tr in text_results:
        if not tr or tr.get("status") in (None, "not_assessed"):
            continue
        key = _norm_text(tr.get("_text", ""))
        if key and any(_same_text(key, k) for k in seen):
            notes.append(f"{src}: same text as evidence already counted (not double counted).")
            continue
        seen.append(key)
        rule = tr["rules"]["score"]
        level = tr["rules"].get("level", "low")
        m = tr["model"]
        if m.get("risk_score") is not None and not m.get("unknown"):
            s = (1 - config.RULE_BLEND) * m["risk_score"] + config.RULE_BLEND * rule
            basis = "model+rules"
            if level == "high" and rule > s:      # several independent cues: the SMS model is out of domain, rules set a floor
                s, basis = rule, "model+rules (rule floor: several independent warning cues)"
            cands.append((src, s, basis, 1.0))
        elif rule <= 0:                                    # nothing to go on: no evidence, NOT a zero-risk vote
            notes.append(f"{src}: model Unknown/unavailable and no rule cues - not counted.")
        else:                                              # model missing / Unknown -> rules only, down-weighted
            why = "model Unknown/Review" if m.get("unknown") else "model unavailable"
            cands.append((src, rule, f"rules only ({why})", config.EXPERIMENTAL_FACTOR))
    if not cands:
        return None
    best = max(cands, key=lambda c: c[1])
    return {"score": best[1], "factor": min(c[3] for c in cands if c[0] == best[0]), "basis": best[2],
            "source": best[0], "all": [(c[0], round(c[1], 3), c[2]) for c in cands], "notes": notes}


def fuse(results, weights=None):
    """results: dict with keys text, ocr (list), video, audio, face, voice, consistency."""
    W = dict(config.FUSION_WEIGHTS)
    W.update(weights or {})
    items, excluded = [], []

    # --- message content (pasted text + OCR text) ---
    tg = _text_group([("pasted text", results.get("text"))] +
                     [(f"OCR: {o.get('source', 'image')}", (o or {}).get("text_analysis")) for o in results.get("ocr", [])])
    if tg:
        items.append({"group": "message_content", "score": tg["score"], "weight": W["message_content"] * tg["factor"],
                      "basis": f"{tg['basis']} from {tg['source']}", "extra": tg})
    else:
        excluded.append(("message_content", "no text / OCR text was analysed"))

    # --- media authenticity ---
    for key, group, label in (("video", "video_authenticity", "video"), ("audio", "audio_authenticity", "audio")):
        r = results.get(key)
        if not r:
            excluded.append((group, f"no {label} evidence supplied"))
        elif r.get("risk_score") is None:
            excluded.append((group, r.get("reason") or "no usable score"))
        elif r.get("unknown"):
            excluded.append((group, "flagged Unknown / Review: " + "; ".join(r.get("reasons", [])[:2])))
        else:
            f = config.EXPERIMENTAL_FACTOR if r.get("status") == "experimental" else 1.0
            items.append({"group": group, "score": r["risk_score"], "weight": W[group] * f,
                          "basis": f"{label} classifier score" + (" (experimental, weight x%.1f)" % f if f < 1 else "")})

    # --- identity (only vs user-supplied references) ---
    ids = []
    for key, name in (("face", "face"), ("voice", "voice")):
        r = results.get(key)
        if not r:
            excluded.append((f"{name}_identity", "no reference sample supplied"))
        elif r.get("label") == "Match":
            excluded.append((f"{name}_identity", "a match is shown but is not evidence of authenticity, so it adds no risk score"))
        elif r.get("label") == "Mismatch":
            f = config.EXPERIMENTAL_FACTOR if r.get("status") == "experimental" else 1.0
            ids.append((name, 1.0 if r["label"] == "Mismatch" else 0.0, f))
        else:
            excluded.append((f"{name}_identity", r.get("reason") or r.get("label", "not assessed")))
    if ids:
        for name, s, f in ids:
            items.append({"group": f"identity_{name}", "score": s, "weight": W["identity"] / 2 * f * (2 if len(ids) == 1 else 1),
                          "basis": f"{name} {'mismatch' if s else 'match'} vs supplied reference (a match is not proof of authenticity)"})

    # --- consistency ---
    c = results.get("consistency")
    if c and c.get("risk_score") is not None:
        items.append({"group": "consistency", "score": c["risk_score"], "weight": W["consistency"] * config.EXPERIMENTAL_FACTOR,
                      "basis": f"{len(c.get('flagged', []))} of the rule checks flagged"})
    else:
        excluded.append(("consistency", (c or {}).get("reason", "not enough paired evidence")))

    unknown_mods = [k for k in ("text", "video", "audio") if (results.get(k) or {}).get("unknown") or
                    ((results.get(k) or {}).get("model", {}) or {}).get("unknown")]
    if not items:
        return {"status": "not_assessed", "category": "Not assessed", "score": None, "contributions": [],
                "excluded": [{"module": m, "reason": r} for m, r in excluded], "unknown_flagged": unknown_mods,
                "explanation": ["No module produced a usable score, so no risk assessment can be made. "
                                "Missing evidence is NOT treated as low or high risk."],
                "label": "Preliminary risk assessment - not proof of fraud", "weights_configured": W}
    tw = sum(i["weight"] for i in items)
    score = sum(i["weight"] * i["score"] for i in items) / tw
    contribs = [{"module": i["group"], "score": round(i["score"], 3), "weight_used": round(i["weight"], 3),
                 "share_of_total_weight": round(i["weight"] / tw, 3), "points": round(i["weight"] * i["score"] / tw, 3),
                 "basis": i["basis"]} for i in items]
    weak = all(i["group"].startswith(("identity", "consistency")) or i["weight"] < W.get(i["group"], 1) - 1e-9 for i in items)
    capped = False
    if weak and score >= config.RISK_HIGH_MIN:           # only experimental / indirect evidence cannot justify "High"
        score, capped = config.RISK_HIGH_MIN - 0.01, True
    cat = categorise(score)
    exp = [f"Overall score {score:.2f} -> {cat} (Low < {config.RISK_LOW_MAX}, High >= {config.RISK_HIGH_MIN})."]
    top = max(contribs, key=lambda c: c["points"])
    exp.append(f"Largest contribution: {top['module']} ({top['points']:.2f} of {score:.2f}) - {top['basis']}.")
    if capped:
        exp.append("Score capped below High: only experimental or indirect evidence (identity / consistency / "
                   "experimental detector) was available.")
    if len(items) == 1:
        exp.append("Only one kind of evidence contributed; treat the result with extra caution.")
    if unknown_mods:
        exp.append("Some evidence was flagged Unknown / Review and did not count towards the model-based score: "
                   + ", ".join(unknown_mods) + ".")
    if any(i["group"] in ("video_authenticity", "audio_authenticity") and "experimental" in i["basis"] for i in items):
        exp.append("An experimental detector contributed with reduced weight.")
    for n in (tg or {}).get("notes", []):
        exp.append(n)
    return {"status": "assessed", "category": cat, "score": round(score, 3), "contributions": contribs,
            "excluded": [{"module": m, "reason": r} for m, r in excluded], "unknown_flagged": unknown_mods,
            "explanation": exp, "label": "Preliminary risk assessment - not proof of fraud",
            "weights_configured": W, "needs_human_review": True}
