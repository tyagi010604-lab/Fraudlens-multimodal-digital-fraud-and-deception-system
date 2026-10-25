"""Modules 5 & 6: text scam detection, OCR and payment-evidence extraction.

Two kinds of output are kept strictly apart:
  * MODEL prediction  - TF-IDF classifier trained on labelled SMS spam/ham data (scripts/train_models.py).
  * RULE-BASED warnings - transparent regex/keyword indicators. They are hints for a human reviewer;
    a keyword hit (e.g. "urgent") is NOT evidence of fraud on its own.
"""
import html
import re
import shutil
from pathlib import Path

import joblib
import numpy as np

import config

# ------------------------------------------------------------------------------ regexes -----
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TLDS = ("com|net|org|in|co|io|xyz|top|info|biz|ly|me|cc|tk|ml|ga|cf|gq|click|link|online|site|shop|live|"
         "app|vip|club|support|work|icu|buzz|cyou|rest|fun|cam|uk|us|de|ru|cn")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"')\]]+|\b(?:[a-z0-9-]+\.)+(?:%s)\b(?:/[^\s<>\"')\]]*)?" % _TLDS)
PHONE_RES = [
    re.compile(r"(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),               # Indian mobile
    re.compile(r"(?<![\w+])\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"),  # +CC numbers
    re.compile(r"(?<![\d.])0\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"),                  # national 0-prefixed
    re.compile(r"(?<![\d.])1[\s-]?8\d\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),                   # toll-free style
]
UPI_RE = re.compile(r"(?<![\w.@-])([\w.\-]{2,64})@([A-Za-z][A-Za-z0-9]{1,20})(?![\w@]|\.[A-Za-z])")
KNOWN_UPI_HANDLES = {"upi", "ybl", "ibl", "axl", "paytm", "oksbi", "okaxis", "okhdfcbank", "okicici", "apl",
                     "sbi", "icici", "hdfcbank", "axisbank", "pnb", "fbl", "barodampay", "ikwik", "postbank",
                     "airtel", "jio", "freecharge", "yesbank", "kotak", "idfcbank", "aubank", "pingpay"}
AMOUNT_RE = re.compile(r"(?i)(?:₹|rs\.?|inr|\$|£|€|usd|eur|gbp)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:rupees|dollars|usd|inr)\b")
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
ACCT_RE = re.compile(r"(?i)\b(?:a/c|acct?|account)(?:\s*(?:no|number|#))?\.?\s*[:\-]?\s*\d{9,18}\b")
PAYMENT_TERMS = re.compile(
    r"(?i)\b(upi|paytm|phonepe|gpay|google\s?pay|bhim|bank\s+transfer|wire\s+transfer|western\s+union|moneygram|"
    r"gift\s?cards?|itunes\s+card|bitcoin|btc|usdt|crypto(?:currency)?|ifsc|processing\s+fee|registration\s+fee|"
    r"advance\s+payment|security\s+deposit|send\s+money|pay\s+now|cash\s+prize|refund|kyc)\b")

# category -> (weight, [patterns], explanation). Weights are deliberately small: single cues are weak and
# only an accumulation of DIFFERENT categories produces a high warning level (noisy-OR, see rule_warnings).
_SENSITIVE = (r"otp|one[- ]time\s+(?:password|pin|code)|cvv|pin|password|passcode|kyc|card\s+(?:number|details|info\w*)|"
              r"account\s+(?:details|number|info\w*|credentials)|bank\s+(?:details|account\s+details|login)|"
              r"aadhaar|aadhar|pan\s+(?:card|number)|login\s+(?:details|credentials)|credentials|net\s*banking")
INDICATOR_RULES = {
    "urgency": (0.15, [r"\burgent(ly)?\b", r"\bimmediately\b", r"\bact\s+now\b", r"within\s+\d+\s*(hours?|hrs?|minutes?|mins?)",
                       r"\blast\s+chance\b", r"\bfinal\s+notice\b", r"\bexpires?\s+(today|soon|tonight)\b",
                       r"\btoday\s+only\b", r"\blimited\s+time\b", r"\bright\s+away\b|\bwithout\s+delay\b"],
                "Pressure to act quickly is common in scams but also in genuine reminders."),
    "account_threat": (0.25, [r"\b(account|card|sim|wallet|number|upi|services?)\b[^.!?\n]{0,30}\b(blocked|suspended|closed|deactivated|"
                              r"frozen|terminated|disabled|locked|restricted)\b",
                              r"\bavoid\s+(suspension|blocking|closure|penalty|legal\s+action|disconnection)\b",
                              r"\b(legal\s+action|arrest\s+warrant|case\s+(will\s+be\s+)?filed)\b"],
                       "Threat of losing an account/service unless you act - a common pressure tactic."),
    "payment_request": (0.25, [r"\b(send|transfer|pay|deposit)\b[^.!?\n]{0,40}\b(money|payment|fee|amount|funds|rs\.?|inr|usd|\$|₹)",
                               r"\bprocessing\s+fee\b", r"\bregistration\s+fee\b", r"\badvance\s+payment\b",
                               r"\bgift\s?cards?\b", r"\bwire\s+transfer\b", r"\bwestern\s+union\b", r"\bbitcoin\b|\busdt\b",
                               r"\bscan\s+(the\s+)?(qr|code)\b"],
                        "A payment request is not fraud by itself; check who is asking and why."),
    "sensitive_info_request": (0.30, [r"\b(enter|provide|share|send|submit|confirm|update|verify|complete|give|reply\s+with|"
                                      r"tell\s+(us|me))\b[^.!?\n]{0,30}\b(?:%s)\b" % _SENSITIVE,
                                      r"\b(?:otp|cvv|pin|password)\b[^.!?\n]{0,15}\b(is|to|for)\b[^.!?\n]{0,15}\b(verify|confirm|share|tell|send)\b",
                                      r"\bupdate\s+(your\s+)?kyc\b", r"\bkyc\b[^.!?\n]{0,20}\b(pending|expired|incomplete|update|verification)\b"],
                               "Genuine organisations rarely ask for OTPs, PINs, passwords or full account details by message."),
    "impersonation_claim": (0.20, [r"\b(reserve\s+bank|rbi|income\s+tax|customs|police|cbi|narcotics|trai|sbi|hdfc|icici|axis\s+bank)\b",
                                   r"\b(this\s+is|i\s+am|we\s+are)\b[^.\n]{0,40}\b(bank|officer|official|manager|support|department|courier|customer\s+care)\b",
                                   r"\b(microsoft|amazon|apple|paypal|netflix)\s+(support|security|team)\b",
                                   r"\byour\s+(ceo|boss|director|manager)\b"],
                            "Claims of being a bank/government/company cannot be verified from the text alone."),
    "too_good_to_be_true": (0.20, [r"\byou\s+(have\s+)?won\b", r"\bwinner\b", r"\bcongratulations\b", r"\blottery\b",
                                   r"\bprize\b", r"\bclaim\s+(your|now)\b", r"\bguaranteed\b",
                                   r"\bearn\s+[\$₹£]?\s?\d[\d,]*\s*(per|/|a)\s*(day|hour|week)\b",
                                   r"\bwork\s+from\s+home\b", r"\bno\s+experience\s+(needed|required)\b"],
                            "Unexpected prizes / unusually easy earnings are classic scam hooks."),
    "suspicious_link": (0.20, [r"\b(click|tap|open|visit|follow|use|using|via|through)\b[^.!?\n]{0,25}\b(this|the|below|following|attached)?\s*(link|url)\b",
                               r"\b(link|url)\s+(below|above|given|provided)\b"],
                        "A prompt to open a link, or a link flagged by the transparent URL rules."),
}
RULE_LEVELS = {"high": 0.50, "moderate": 0.25}      # noisy-OR thresholds for the rule-based warning level
_COMPILED = {k: [re.compile(p, re.I) for p in v[1]] for k, v in INDICATOR_RULES.items()}

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at",
              "tiny.cc", "buff.ly", "rebrand.ly", "t.ly", "s.id", "wa.me"}
SUSPICIOUS_TLDS = {"xyz", "top", "click", "link", "icu", "buzz", "cyou", "rest", "tk", "ml", "ga", "cf", "gq",
                   "vip", "fun", "cam", "online", "site", "live"}


# ------------------------------------------------------------------------------ cleaning ----
def clean_for_model(text):
    """Normalisation shared by training and inference (keep in sync!)."""
    t = str(text).lower()
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = URL_RE.sub(" urltoken ", t)
    t = re.sub(r"\+?\d[\d\s\-]{8,}\d", " phonetoken ", t)
    t = re.sub(r"\d{5,}", " longnum ", t)
    t = re.sub(r"[£$₹€]", " currencytoken ", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------------- URL heuristics --
def url_flags(url):
    """Transparent, explainable URL heuristics. Returns list of human-readable flags."""
    u = url.strip().rstrip(".,;:!?)")
    low = u.lower()
    host = re.sub(r"^https?://", "", low).split("/")[0].split("?")[0].split(":")[0]
    flags = []
    if low.startswith("http://"):
        flags.append("uses plain HTTP (no TLS)")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        flags.append("host is a raw IP address")
    if "@" in re.sub(r"^https?://", "", low).split("/")[0]:
        flags.append("contains '@' in the address")
    if host in SHORTENERS:
        flags.append("link-shortening service (destination hidden)")
    if host.startswith("xn--") or ".xn--" in host:
        flags.append("punycode/internationalised domain (possible look-alike)")
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        flags.append(f"TLD '.{tld}' is frequently used in abuse (many benign sites use it too)")
    if host.count("-") >= 3:
        flags.append("many hyphens in the domain")
    if host.count(".") >= 4:
        flags.append("many sub-domains")
    if len(u) > 100:
        flags.append("very long URL")
    return flags


# --------------------------------------------------------------------------- extraction -----
def _mask_spans(text, spans):
    chars = list(text)
    for s, e in spans:
        for i in range(s, e):
            chars[i] = " "
    return "".join(chars)


def extract_indicators(text):
    """Regex extraction of URLs, phone numbers, UPI IDs, payment strings. Pure evidence listing."""
    emails = [(m.start(), m.end(), m.group()) for m in EMAIL_RE.finditer(text)]
    upi = []
    upi_spans = []
    for m in UPI_RE.finditer(text):
        handle = m.group(2).lower()
        # an e-mail address always has a dotted domain; the regex's lookahead excludes those
        upi.append({"id": m.group(), "known_handle": handle in KNOWN_UPI_HANDLES})
        upi_spans.append((m.start(), m.end()))
    email_clean = [e for e in emails if not any(s <= e[0] < en for s, en in upi_spans)]
    masked = _mask_spans(text, [(s, e) for s, e, _ in emails])
    urls = []
    for m in URL_RE.finditer(masked):
        u = m.group().rstrip(".,;:!?)")
        if u and u not in [x["url"] for x in urls]:
            urls.append({"url": u, "flags": url_flags(u)})
    no_url = _mask_spans(masked, [m.span() for m in URL_RE.finditer(masked)])
    phones = []
    for rx in PHONE_RES:
        for m in rx.finditer(no_url):
            digits = re.sub(r"\D", "", m.group())
            if 8 <= len(digits) <= 15 and m.group().strip() not in phones:
                phones.append(m.group().strip())
    terms = sorted({m.group().lower() for m in PAYMENT_TERMS.finditer(text)})
    return {"urls": urls, "phone_numbers": phones, "upi_ids": upi,
            "emails": [e[2] for e in email_clean], "payment_terms": terms,
            "amounts": sorted({m.group().strip() for m in AMOUNT_RE.finditer(text)})[:10],
            "ifsc_codes": IFSC_RE.findall(text), "account_strings": [m.group() for m in ACCT_RE.finditer(text)]}


def rule_warnings(text, indicators=None):
    """Rule-based warning level. noisy-OR of small category weights -> 0..1 (NOT a probability)."""
    indicators = indicators or extract_indicators(text)
    hits, spans = {}, []
    for cat, regs in _COMPILED.items():
        for rx in regs:
            for m in rx.finditer(text):
                hits.setdefault(cat, []).append(m.group())
                spans.append({"category": cat, "text": m.group(), "start": m.start(), "end": m.end()})
    if any(u["flags"] for u in indicators["urls"]):
        hits.setdefault("suspicious_link", []).extend(u["url"] for u in indicators["urls"] if u["flags"])
        for u in indicators["urls"]:
            if u["flags"]:
                for m in re.finditer(re.escape(u["url"]), text):
                    spans.append({"category": "suspicious_link", "text": u["url"], "start": m.start(), "end": m.end()})
    prod = 1.0
    for cat in hits:
        prod *= 1.0 - INDICATOR_RULES[cat][0]
    score = round(1.0 - prod, 3)
    cats = [{"category": c, "matches": sorted(set(h))[:8], "why": INDICATOR_RULES[c][2],
             "weight": INDICATOR_RULES[c][0]} for c, h in hits.items()]
    # 'high' needs several DIFFERENT kinds of cue at once; a single urgent/payment word never reaches it
    if len(cats) >= 3 and score >= RULE_LEVELS["high"]:
        level = "high"
    elif len(cats) >= 2 and score >= RULE_LEVELS["moderate"]:
        level = "moderate"
    else:
        level = "low"
    return {"score": score, "level": level, "categories": cats, "spans": spans,
            "note": "Rule-based warning from keyword/regex cues. Not a model prediction and not proof of fraud."}


def highlight_html(text, spans):
    """Escape `text` and wrap rule matches in <mark> tags (for st.markdown(unsafe_allow_html=True))."""
    if not spans:
        return html.escape(text)
    merged = []
    for s in sorted(spans, key=lambda d: (d["start"], -d["end"])):
        if merged and s["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            merged[-1]["category"] += ", " + s["category"]
        else:
            merged.append({"start": s["start"], "end": s["end"], "category": s["category"]})
    out, pos = [], 0
    for m in merged:
        out.append(html.escape(text[pos:m["start"]]))
        cats = ", ".join(sorted(set(m["category"].split(", "))))
        out.append(f'<mark title="{html.escape(cats)}">{html.escape(text[m["start"]:m["end"]])}</mark>')
        pos = m["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


# --------------------------------------------------------------------------- text model -----
_TEXT_BUNDLE = {"obj": None, "err": None, "loaded": False}


def load_text_model(force=False):
    if _TEXT_BUNDLE["loaded"] and not force:
        return _TEXT_BUNDLE["obj"]
    _TEXT_BUNDLE["loaded"] = True
    try:
        _TEXT_BUNDLE["obj"] = joblib.load(config.TEXT_MODEL_PATH) if config.TEXT_MODEL_PATH.exists() else None
        if _TEXT_BUNDLE["obj"] is None:
            _TEXT_BUNDLE["err"] = "Text model not trained yet. Run: python scripts/train_models.py --only text"
    except Exception as e:   # version mismatch, corrupted file ...
        _TEXT_BUNDLE["obj"], _TEXT_BUNDLE["err"] = None, (
            f"Text model could not be loaded ({type(e).__name__}); retrain with scripts/train_models.py --only text")
    return _TEXT_BUNDLE["obj"]


def text_model_error():
    load_text_model()
    return _TEXT_BUNDLE["err"]


def _chunks(text, size=config.TEXT_CHUNK_CHARS):
    if len(text) <= int(size * 1.5):
        return [text]
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > size and cur:
            out.append(cur.strip())
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur.strip())
    return [c for c in out if len(c) > 5] or [text]


def model_predict(text):
    """Run the TF-IDF classifier (chunk-wise for long text). Returns dict incl. Unknown/Review logic."""
    b = load_text_model()
    if b is None:
        return {"status": "unavailable", "label": "Not assessed", "risk_score": None,
                "reason": text_model_error()}
    cleaned = clean_for_model(text)
    words = re.findall(r"[a-z']+", cleaned)
    reasons = []
    if len(words) < config.TEXT_MIN_TOKENS:
        return {"status": "unknown", "label": "Unknown / Review", "risk_score": None,
                "reasons": [f"Text has fewer than {config.TEXT_MIN_TOKENS} words - too little to classify."],
                "model_info": b["info"]}
    chunks = _chunks(text)
    probs = b["pipeline"].predict_proba([clean_for_model(c) for c in chunks])[:, 1]
    top = int(np.argmax(probs))
    p = float(probs[top])
    vocab = b["word_vocab"]
    cov = float(np.mean([w in vocab for w in words]))
    unk = b["unknown"]
    if cov < unk["min_vocab_coverage"]:
        reasons.append(f"Only {cov:.0%} of the words appear in the training vocabulary "
                       f"(validation-calibrated minimum {unk['min_vocab_coverage']:.0%}): text looks unlike the "
                       "SMS data the model was trained on.")
    lo, hi = unk["uncertain_low"], unk["uncertain_high"]
    if lo <= p <= hi:
        reasons.append(f"Model score {p:.2f} lies inside the uncertain band [{lo:.2f}, {hi:.2f}] where validation "
                       "predictions were unreliable.")
    thr = b["threshold"]
    if reasons:
        label = "Unknown / Review"
    elif p >= thr:
        label = "Spam-like (SMS-trained model)"
    else:
        label = "No spam pattern detected (SMS-trained model)"
    notes = []
    if len(chunks) > 1:
        notes.append(f"Long text scored in {len(chunks)} chunks; highest chunk score used.")
    if len(text) > 400:
        notes.append("Model was trained on short SMS messages; emails/job ads are a domain shift.")
    notes.append("This model detects SPAM in SMS-style text. It was not trained on banking/KYC/OTP/UPI scams, so a "
                 "'no spam pattern' result does not mean the message is safe; see the rule-based warnings.")
    return {"status": "experimental" if b["info"].get("experimental") else "ready", "label": label,
            "risk_score": round(p, 4), "threshold": round(thr, 4), "vocab_coverage": round(cov, 3),
            "unknown": bool(reasons), "reasons": reasons, "notes": notes, "n_chunks": len(chunks),
            "most_suspicious_chunk": chunks[top][:200], "model_info": b["info"]}


def assess(model, rules):
    """One transparent headline built from the two SEPARATE signals (model prediction, rule-based warnings)."""
    lvl, m_label = rules["level"], model.get("label", "")
    cats = ", ".join(c["category"].replace("_", " ") for c in rules["categories"])
    if m_label.startswith("Spam-like"):
        return {"label": "Suspicious", "basis": "model" + (" + rule cues" if rules["categories"] else ""),
                "explanation": "The SMS-spam model scored this text above its decision threshold."
                               + (f" Rule cues also present: {cats}." if cats else "")}
    if lvl == "high":
        return {"label": "Suspicious", "basis": "rule-based warning",
                "explanation": f"Several different warning cues co-occur ({cats}). The SMS-trained model "
                               f"returned: {m_label or 'no result'}; it was not trained on this kind of scam, so the rules decide here."}
    if m_label.startswith("Unknown"):
        extra = f" Some cues were present ({cats})." if cats else ""
        return {"label": "Unknown / Review", "basis": "model uncertain",
                "explanation": "The model is uncertain or the text looks unlike its training data; no strong rule cues." + extra}
    if lvl == "moderate":
        return {"label": "Review - some warning cues", "basis": "rule-based warning",
                "explanation": f"Two kinds of warning cue were found ({cats}) but the model sees no spam pattern."}
    if model.get("status") == "unavailable":
        return {"label": "Not assessed by model", "basis": "rules only",
                "explanation": "No text model is available; only the rule-based cues were checked."
                               + (f" Cues: {cats}." if cats else " No cues found.")}
    return {"label": "No strong warning signs", "basis": "model + rules",
            "explanation": "No spam pattern from the model and no combination of warning cues." +
                           (f" (Isolated cue only: {cats}.)" if cats else "")}


def analyze_text(text, source="pasted text"):
    """Full Module-5 analysis of one text: model prediction + indicators + rule warnings."""
    text = (text or "").strip()
    if not text:
        return {"status": "not_assessed", "reason": "No text supplied."}
    if len(text) > 50000:
        text = text[:50000]
    ind = extract_indicators(text)
    rules = rule_warnings(text, ind)
    model = model_predict(text)
    for u in ind["urls"]:
        u["lexical_score"] = url_lexical_score(u["url"])
    return {"status": model["status"] if model["status"] != "unavailable" else "rules_only", "source": source,
            "assessment": assess(model, rules), "model": model, "rules": rules, "indicators": ind, "n_chars": len(text), "_text": text,
            "disclaimer": "Identifying a URL, phone number or UPI ID does not establish fraud."}


# ---------------------------------------------------------------------------- URL model -----
_URL_BUNDLE = {"obj": None, "loaded": False}


def load_url_model():
    if not _URL_BUNDLE["loaded"]:
        _URL_BUNDLE["loaded"] = True
        try:
            _URL_BUNDLE["obj"] = joblib.load(config.URL_MODEL_PATH) if config.URL_MODEL_PATH.exists() else None
        except Exception:
            _URL_BUNDLE["obj"] = None
    return _URL_BUNDLE["obj"]


def url_normalise(url):
    u = url.strip().lower()
    return re.sub(r"^https?://(www\.)?", "", u)


def url_lexical_score(url):
    """Experimental lexical URL score from the QR-dataset URL model, or None if unavailable."""
    b = load_url_model()
    if b is None:
        return None
    return round(float(b["pipeline"].predict_proba([url_normalise(url)])[0, 1]), 3)


# ------------------------------------------------------------------------------ OCR ---------
TESSERACT_HELP = (
    "**Tesseract OCR was not found.** OCR (screenshot text extraction) needs the separate Tesseract program.\n\n"
    "**Windows:** 1) download the installer from https://github.com/UB-Mannheim/tesseract/wiki  2) run it and keep the default folder "
    "`C:\\Program Files\\Tesseract-OCR` (tick *English* language data)  3) **restart VS Code and the terminal**, then run "
    "`streamlit run app.py` again. FraudLens looks in the standard install folders automatically - no PATH change is required. "
    "If you installed elsewhere, set `TESSERACT_CMD` in `config.py` (e.g. `r\"D:\\Tools\\Tesseract-OCR\\tesseract.exe\"`) or run "
    "`setx TESSERACT_CMD \"D:\\Tools\\Tesseract-OCR\\tesseract.exe\"` and reopen VS Code.\n\n"
    "**macOS:** `brew install tesseract`   **Ubuntu/Debian:** `sudo apt install tesseract-ocr`\n\n"
    "Check your setup with `python scripts/check_setup.py`.")


def tesseract_candidates():
    """Ordered list of (source, path) places to look for the Tesseract executable."""
    import os
    c = []
    if os.environ.get("TESSERACT_CMD"):
        c.append(("environment variable TESSERACT_CMD", os.environ["TESSERACT_CMD"]))
    if getattr(config, "TESSERACT_CMD", ""):
        c.append(("config.py TESSERACT_CMD", config.TESSERACT_CMD))
    on_path = shutil.which("tesseract")
    if on_path:
        c.append(("PATH", on_path))
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        if os.environ.get(var):
            c.append(("standard Windows folder", str(Path(os.environ[var]) / "Tesseract-OCR" / "tesseract.exe")))
    if os.environ.get("LOCALAPPDATA"):
        c.append(("per-user Windows install", str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Tesseract-OCR" / "tesseract.exe")))
    c += [("standard Windows folder", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
          ("standard Windows folder", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")]
    if os.environ.get("USERPROFILE"):
        up = Path(os.environ["USERPROFILE"])
        c += [("Scoop", str(up / "scoop" / "shims" / "tesseract.exe")),
              ("Scoop", str(up / "scoop" / "apps" / "tesseract" / "current" / "tesseract.exe"))]
    c += [("Chocolatey", r"C:\ProgramData\chocolatey\bin\tesseract.exe"),
          ("Homebrew", "/opt/homebrew/bin/tesseract"), ("Homebrew/Linux", "/usr/local/bin/tesseract"), ("Linux", "/usr/bin/tesseract")]
    return c


_TESS = {"cmd": None, "msg": None}


def check_tesseract(refresh=False):
    """Locate and verify Tesseract. Returns (available: bool, message: str); sets pytesseract's executable path."""
    import pytesseract
    if _TESS["cmd"] and not refresh:
        return True, _TESS["msg"]
    problems = []
    for source, cmd in tesseract_candidates():
        if not Path(cmd).is_file():
            if source.startswith(("environment", "config")):
                problems.append(f"{source} points to a file that does not exist: {cmd}")
            continue
        pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            ver = str(pytesseract.get_tesseract_version())
        except Exception as e:                       # found but not runnable (e.g. blocked, wrong architecture)
            problems.append(f"Found {cmd} ({source}) but it could not be run: {type(e).__name__}")
            continue
        try:
            langs = pytesseract.get_languages(config="")
        except Exception:
            langs = []
        if langs and "eng" not in langs:
            problems.append(f"Tesseract {ver} at {cmd} has no English language data ('eng'); re-run the installer and tick English.")
            continue
        _TESS["cmd"], _TESS["msg"] = cmd, f"Tesseract {ver} found at {cmd} (via {source})"
        return True, _TESS["msg"]
    return False, ("\n\n".join(["⚠️ " + p for p in problems]) + "\n\n" if problems else "") + TESSERACT_HELP


def _prep_for_ocr(img):
    from PIL import Image, ImageOps
    img = img.convert("L")
    if min(img.size) < 700:                          # small screenshots OCR better when upscaled
        f = 700 / min(img.size)
        img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def ocr_image(image):
    """OCR a PIL image or path. Returns dict(status, text, mean_confidence, message)."""
    from PIL import Image
    ok, msg = check_tesseract()
    if not ok:
        return {"status": "unavailable", "text": "", "message": msg}
    import pytesseract
    try:
        img = Image.open(image) if not hasattr(image, "convert") else image
        img.load()
    except Exception as e:
        return {"status": "error", "text": "", "message": f"Image could not be opened ({type(e).__name__})."}
    prepped = _prep_for_ocr(img)
    try:
        data = pytesseract.image_to_data(prepped, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception as e:
        return {"status": "error", "text": "", "message": f"OCR failed ({type(e).__name__}: {e})."}
    words, confs, lines = [], [], {}
    for t, c, b, p, l in zip(data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]):
        if t.strip():
            lines.setdefault((b, p, l), []).append(t)
            try:
                if float(c) >= 0:
                    confs.append(float(c))
            except ValueError:
                pass
    text = "\n".join(" ".join(v) for v in lines.values())
    return {"status": "ok", "text": text, "mean_confidence": round(float(np.mean(confs)), 1) if confs else None,
            "n_words": sum(len(v) for v in lines.values()), "message": msg}


def decode_qr(image):
    """Decode a QR code (if any) with OpenCV. Returns payload string or ''."""
    import cv2
    from PIL import Image
    img = Image.open(image) if not hasattr(image, "convert") else image
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    det = cv2.QRCodeDetector()
    text, _, _ = det.detectAndDecode(arr)
    if not text:
        for scale in (2.0, 0.5):
            r = cv2.resize(arr, None, fx=scale, fy=scale)
            text, _, _ = det.detectAndDecode(r)
            if text:
                break
    return text or ""


def analyze_image(image, label="image"):
    """Module 6: OCR -> indicators -> text classifier. Also decodes QR codes."""
    ocr = ocr_image(image)
    qr = ""
    try:
        qr = decode_qr(image)
    except Exception:
        pass
    result = {"source": label, "ocr": {k: v for k, v in ocr.items() if k != "text"}, "extracted_text": ocr.get("text", ""),
              "qr_payload": qr}
    combined = "\n".join(x for x in (ocr.get("text", ""), qr) if x).strip()
    if ocr["status"] == "unavailable" and not qr:
        result.update({"status": "unavailable", "reason": ocr["message"]})
        return result
    if ocr["status"] == "error" and not qr:
        result.update({"status": "error", "reason": ocr["message"]})
        return result
    if not combined:
        result.update({"status": "no_text", "reason": "No readable text or QR payload was found in the image."})
        return result
    result["text_analysis"] = analyze_text(combined, source=f"OCR of {label}")
    result["status"] = "ok"
    if ocr["status"] == "unavailable":
        result["warning"] = "OCR unavailable (Tesseract missing); only the QR payload was analysed."
    elif ocr.get("mean_confidence") is not None and ocr["mean_confidence"] < 60:
        result["warning"] = f"Low OCR confidence ({ocr['mean_confidence']}%): extracted text may contain errors."
    return result
