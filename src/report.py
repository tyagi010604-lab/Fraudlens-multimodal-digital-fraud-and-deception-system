"""Report generation (JSON / CSV / plain text) with privacy defaults.

  * No media is ever embedded - only derived results.
  * Evidence files are identified by type, extension, size and a short SHA-256 prefix (not the file name).
  * By default phone numbers, UPI IDs and e-mail addresses are masked in every string of the report.
  * Long text is truncated.
"""
import csv
import io
import json
import re

import numpy as np

import config

DROP_KEYS = {"per_frame", "per_frame_summary", "spans"}
MAX_TEXT = 1200


def _mask_phone(m):
    d = m.group()
    digits = [i for i, c in enumerate(d) if c.isdigit()]
    if len(digits) <= 4:
        return d
    keep = set(digits[:2] + digits[-2:])
    return "".join(c if (not c.isdigit() or i in keep) else "X" for i, c in enumerate(d))


_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s\-()]{7,16}\d(?![\w])")
_EMAIL = re.compile(r"([\w.+-])[\w.+-]*@([\w-]+(?:\.[\w-]+)+)")
_UPI = re.compile(r"(?<![\w.@-])([\w.\-])[\w.\-]{1,63}@([A-Za-z][A-Za-z0-9]{1,20})(?![\w@]|\.[A-Za-z])")


def mask_text(s):
    s = _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", s)
    s = _UPI.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", s)
    return _PHONE.sub(_mask_phone, s)


def sanitize(obj, mask=True):
    """JSON-safe deep copy: numpy -> python, private/bulky keys dropped, optional PII masking."""
    if isinstance(obj, dict):
        return {k: sanitize(v, mask) for k, v in obj.items() if not str(k).startswith("_") and k not in DROP_KEYS}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v, mask) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, str):
        t = mask_text(obj) if mask else obj
        return t if len(t) <= MAX_TEXT else t[:MAX_TEXT] + " …[truncated]"
    return obj


def build_report(analysis, mask_pii=True, include_text=True):
    res = analysis.get("results", {})
    rep = {
        "analysis_id": analysis["analysis_id"], "timestamp": analysis["timestamp"], "app_version": config.APP_VERSION,
        "evidence_types_analysed": analysis.get("evidence_types", []),
        "evidence_files": analysis.get("evidence", []),
        "privacy": {"pii_masked": mask_pii, "media_stored": False,
                    "note": "Uploaded files were processed in a temporary folder that was deleted after analysis."},
        "overall_risk_assessment": analysis.get("fusion"),
        "module_results": {k: v for k, v in res.items() if k != "ocr"},
        "ocr_results": res.get("ocr", []),
        "consistency_findings": res.get("consistency"),
        "not_assessed_or_unavailable": analysis.get("not_assessed", {}),
        "module_status_at_analysis_time": analysis.get("module_status", []),
        "limitations": [
            "Preliminary, automated assessment; it is not proof of fraud, identity or criminal activity.",
            "Video detector is experimental (106 training clips); audio detector is untrained unless suitable data was supplied.",
            "Face / voice matches do not prove authenticity or identity; voice comparison is experimental.",
            "Text model was trained on SMS spam; other text types are a domain shift.",
            "Identifying a URL, phone number or UPI ID does not establish fraud.",
            "No lip-sync analysis and no localisation of manipulated frames are implemented."],
        "disclaimer": config.DISCLAIMER}
    if include_text:
        t = res.get("text")
        if t and t.get("_text"):
            rep["pasted_text_excerpt"] = t["_text"][:MAX_TEXT]
    return sanitize(rep, mask_pii)


def to_json(report):
    return json.dumps(report, indent=2, ensure_ascii=False)


def to_csv(report):
    rows = [("section", "item", "status", "label_or_value", "score", "details")]
    fu = report.get("overall_risk_assessment") or {}
    rows.append(("overall", "fusion", fu.get("status"), fu.get("category"), fu.get("score"), " | ".join(fu.get("explanation", []))))
    for c in fu.get("contributions", []):
        rows.append(("fusion_contribution", c["module"], "used", "", c["points"],
                     f"score={c['score']} weight={c['weight_used']} {c['basis']}"))
    for e in fu.get("excluded", []):
        rows.append(("fusion_excluded", e["module"], "excluded", "", "", e["reason"]))
    for name, r in report.get("module_results", {}).items():
        if not isinstance(r, dict):
            continue
        m = r.get("model", {}) if name == "text" else r
        rows.append(("module", name, r.get("status"), (m or {}).get("label"), (m or {}).get("risk_score", r.get("similarity")),
                     (m or {}).get("reason") or "; ".join((m or {}).get("reasons", []) or [])))
        ind = r.get("indicators")
        if ind:
            for u in ind.get("urls", []):
                rows.append(("indicator", "url", "", u["url"], "", "; ".join(u.get("flags", []))))
            for key in ("phone_numbers", "emails", "payment_terms", "amounts"):
                for v in ind.get(key, []):
                    rows.append(("indicator", key, "", v, "", ""))
            for v in ind.get("upi_ids", []):
                rows.append(("indicator", "upi_id", "", v["id"], "", f"known_handle={v['known_handle']}"))
    for i, o in enumerate(report.get("ocr_results", [])):
        ta = o.get("text_analysis") or {}
        rows.append(("ocr", f"image_{i + 1}", o.get("status"), (ta.get("model") or {}).get("label"),
                     (ta.get("model") or {}).get("risk_score"), o.get("reason") or o.get("warning") or ""))
        for u in (ta.get("indicators") or {}).get("urls", []):
            rows.append(("indicator", f"ocr_{i + 1}_url", "", u["url"], "", "; ".join(u.get("flags", []))))
        for key in ("phone_numbers", "payment_terms"):
            for v in (ta.get("indicators") or {}).get(key, []):
                rows.append(("indicator", f"ocr_{i + 1}_{key}", "", v, "", ""))
        for v in (ta.get("indicators") or {}).get("upi_ids", []):
            rows.append(("indicator", f"ocr_{i + 1}_upi_id", "", v["id"], "", ""))
    for c in (report.get("consistency_findings") or {}).get("checks", []):
        rows.append(("consistency", c["check"], c["status"], "", "", c["detail"]))
    for k, v in report.get("not_assessed_or_unavailable", {}).items():
        rows.append(("not_assessed", k, "not_assessed", "", "", v))
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue()


def to_text(report):
    L = []
    add = L.append
    add("FRAUDLENS ANALYSIS REPORT")
    add("=" * 60)
    add(f"Analysis ID : {report['analysis_id']}")
    add(f"Timestamp   : {report['timestamp']}")
    add(f"Evidence    : {', '.join(report['evidence_types_analysed']) or 'none'}")
    add(f"PII masked  : {report['privacy']['pii_masked']}   Media stored: {report['privacy']['media_stored']}")
    fu = report.get("overall_risk_assessment") or {}
    add("\nOVERALL (preliminary - not proof of fraud)")
    add("-" * 60)
    add(f"Category: {fu.get('category')}    Score: {fu.get('score')}")
    for e in fu.get("explanation", []):
        add(f"  * {e}")
    add("  Contributing modules:")
    for c in fu.get("contributions", []):
        add(f"    - {c['module']}: score {c['score']} x weight {c['weight_used']} = {c['points']} ({c['basis']})")
    add("  Excluded (not counted as zero risk):")
    for e in fu.get("excluded", []):
        add(f"    - {e['module']}: {e['reason']}")
    add("\nMODULE RESULTS")
    add("-" * 60)
    for name, r in report.get("module_results", {}).items():
        if not isinstance(r, dict):
            add(f"[{name}] Not assessed")
            continue
        m = r.get("model", {}) if name == "text" else r
        add(f"[{name}] status={r.get('status')} label={(m or {}).get('label')} score={(m or {}).get('risk_score', r.get('similarity'))}")
        why = (m or {}).get("reason") or "; ".join((m or {}).get("reasons", []) or [])
        if why:
            add(f"    reason: {why}")
        if name == "text":
            for c in r.get("rules", {}).get("categories", []):
                add(f"    rule-based cue [{c['category']}]: {', '.join(c['matches'])}")
        ind = r.get("indicators")
        if ind:
            for u in ind["urls"]:
                add(f"    URL: {u['url']} {('(' + '; '.join(u['flags']) + ')') if u['flags'] else ''}")
            for key in ("phone_numbers", "payment_terms", "emails"):
                if ind.get(key):
                    add(f"    {key}: {', '.join(ind[key])}")
            if ind.get("upi_ids"):
                add("    upi_ids: " + ", ".join(v["id"] for v in ind["upi_ids"]))
    for i, o in enumerate(report.get("ocr_results", [])):
        add(f"[ocr image {i + 1}] status={o.get('status')} {o.get('reason', '')}")
        if o.get("extracted_text"):
            add("    extracted text: " + o["extracted_text"].replace("\n", " ")[:600])
    cf = report.get("consistency_findings")
    if cf:
        add("\nCONSISTENCY")
        add("-" * 60)
        add(f"Status: {cf.get('label')}")
        for c in cf.get("checks", []):
            add(f"  - [{c['status']}] {c['check']}: {c['detail']}")
    add("\nNOT ASSESSED / UNAVAILABLE")
    add("-" * 60)
    for k, v in report.get("not_assessed_or_unavailable", {}).items():
        add(f"  - {k}: {v}")
    add("\nLIMITATIONS")
    add("-" * 60)
    for l in report["limitations"]:
        add(f"  - {l}")
    add("\n" + report["disclaimer"])
    return "\n".join(L)
