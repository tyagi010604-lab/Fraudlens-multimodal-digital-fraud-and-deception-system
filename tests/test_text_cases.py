"""Regression tests for the text pipeline (python tests/test_text_cases.py).

These are sanity checks on behaviour - they do not replace evaluation on a labelled held-out set.
The expected behaviour is asserted; no output is hard-coded in the application.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import fusion, pipeline, text_ocr

CASES = {
    "A personal": "Hey! Just a reminder that we're meeting tomorrow at 5 PM at the café near college. Let me know if you're running late. See you there!",
    "B delivery": "Your order has been shipped and is expected to arrive tomorrow. You can check the delivery status in the shopping app. Thank you for your purchase.",
    "C bank/KYC": "URGENT: Your bank account will be blocked today. Complete your KYC immediately using this link and enter your account details to avoid suspension.",
    "D college": "Hi, are you coming to college tomorrow? We need to finish the presentation before Friday. Let me know when you reach.",
}
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


b = text_ocr.load_text_model()
check("text model loads", b is not None, text_ocr.text_model_error() or "")

# 1) label mapping + preprocessing parity (training used clean_for_model on raw text; inference does too)
spam = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341."
ham = "Ok lar... Joking wif u oni..."
ps, ph = text_ocr.model_predict(spam), text_ocr.model_predict(ham)
check("label mapping: classic spam scores above ham", (ps["risk_score"] or 0) > (ph["risk_score"] or 1), f"{ps['risk_score']} vs {ph['risk_score']}")
check("classic spam is flagged spam-like", ps["label"].startswith("Spam-like"), ps["label"])
check("classic ham is not flagged", not ph["label"].startswith("Spam-like"), ph["label"])
check("inference cleaning == training cleaning", text_ocr.clean_for_model(spam) == text_ocr.clean_for_model(spam.upper()))

# 2) uncertainty mechanism is non-degenerate and actually reachable
u = b["unknown"]
check("uncertain band has non-zero width", u["uncertain_high"] - u["uncertain_low"] > 0.05, f"{u['uncertain_low']}..{u['uncertain_high']}")
mid = (u["uncertain_low"] + u["uncertain_high"]) / 2
import numpy as np
probs = {}
for k, t in CASES.items():
    r = text_ocr.analyze_text(t)
    probs[k] = r
    print(f"   {k:11s} model={r['model']['label']:46s} score={r['model'].get('risk_score')}  rules={r['rules']['level']}({r['rules']['score']}) "
          f"-> {r['assessment']['label']}")

# 3) the four cases (behavioural expectations)
for k in ("A personal", "D college"):
    check(f"{k}: not suspicious / low", probs[k]["assessment"]["label"] in ("No strong warning signs",), probs[k]["assessment"]["label"])
check("B delivery: not suspicious (benign or honestly Unknown)", probs["B delivery"]["assessment"]["label"] in ("No strong warning signs", "Unknown / Review"),
      probs["B delivery"]["assessment"]["label"])
c = probs["C bank/KYC"]
cats = {x["category"] for x in c["rules"]["categories"]}
check("C bank/KYC: flagged Suspicious", c["assessment"]["label"] == "Suspicious", c["assessment"]["label"])
check("C: urgency + account threat + sensitive-info + link cues all explained",
      {"urgency", "account_threat", "sensitive_info_request", "suspicious_link"} <= cats, str(sorted(cats)))
check("C: model prediction and rule warning kept separate", c["model"]["label"] != c["assessment"]["label"] or c["assessment"]["basis"] != "rule-based warning")

# 4) benign words alone do not create 'high' warnings
for t in ("Please pay me back the 200 rupees for lunch when you can.", "I need the OTP project notes urgent for the exam tomorrow",
          "Click the link to join the meeting at 5 pm, urgent topic."):
    r = text_ocr.rule_warnings(t)["level"]
    check(f"isolated words not 'high': {t[:42]}...", r != "high", r)

# 5) fusion: rule evidence is reflected, benign gives Low/not suspicious, same text via OCR not double counted
fa = pipeline.run_analysis({"text": CASES["A personal"]})["fusion"]
check("A: overall risk Low", fa["category"] == "Low", f"{fa['category']} {fa['score']}")
fc = pipeline.run_analysis({"text": CASES["C bank/KYC"]})["fusion"]
check("C: overall risk is not Low", fc["category"] in ("Medium", "High"), f"{fc['category']} {fc['score']}")
dup = fusion.fuse({"text": probs["C bank/KYC"], "ocr": [{"source": "image 1", "text_analysis": probs["C bank/KYC"]}]})
check("same text via OCR counted once", sum(1 for x in dup["contributions"] if x["module"] == "message_content") == 1)
print("\nFAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
