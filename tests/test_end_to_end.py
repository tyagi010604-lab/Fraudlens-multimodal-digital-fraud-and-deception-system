"""End-to-end smoke test using the supplied datasets (run: python tests/test_end_to_end.py)."""
import json, sys, glob, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import data_access as da, pipeline, report, text_ocr
from PIL import Image, ImageDraw, ImageFont

def up(p): return (Path(p).name, Path(p).read_bytes())
def show(title, a):
    f = a["fusion"]; print(f"\n### {title}\n evidence={a['evidence_types']} fusion={f['category']} score={f['score']}")
    for c in f["contributions"]: print("   +", c["module"], c["score"], "w=", c["weight_used"])
    print("   excluded:", [e["module"] for e in f["excluded"]])
    for k, v in a["results"].items():
        if k == "ocr":
            for o in v: print("   ocr:", o["status"], (o.get("extracted_text") or "")[:60].replace("\n", " "), "| qr:", o.get("qr_payload"))
        elif isinstance(v, dict): 
            m = v.get("model", v); print(f"   {k}: {m.get('status', v.get('status'))} | {m.get('label')} | score={m.get('risk_score', v.get('similarity'))}")

sd = da.ensure_sdfvd(); tm = da.ensure_timit_audio(); uy = da.ensure_uyghur()
import json as j
test_pairs = j.loads((config.EVAL_DIR / "video_metrics.json").read_text())["split"]["test_pairs"]
t = sd[sd.pair_id.isin(test_pairs)]
real_v, fake_v = t[t.label == "real"].path.iloc[0], t[t.label == "fake"].path.iloc[0]

# 1 text only (scam-like) and benign
a = pipeline.run_analysis({"text": "URGENT! Your SBI account will be blocked within 24 hours. Pay Rs. 499 processing fee to ravi.k@oksbi or call +91 98765 43210. Verify KYC at http://sbi-kyc-update.xyz/login"}); show("scam-like text", a)
a2 = pipeline.run_analysis({"text": "Hey, are we still meeting for lunch at 1pm tomorrow? Let me know if the place works."}); show("benign text", a2)
a3 = pipeline.run_analysis({"text": "Dear Candidate, we are pleased to offer you the Data Analyst role at our firm. Please review the attached contract, the salary band, the leave policy and the onboarding schedule for next month and confirm acceptance by replying to this email at your convenience."}); show("formal email (domain shift)", a3)
# 2 video real + fake (held-out)
for nm, p in (("held-out REAL video", real_v), ("held-out FAKE video", fake_v)): show(nm, pipeline.run_analysis({"video": up(p)}))
# 3 audio: TIMIT wav, Uyghur wavs; voice-ID same vs different speaker
w = tm[tm.speaker == "fadg0"].path.tolist(); w2 = tm[tm.speaker == "mdab0"].path.tolist()
show("audio TIMIT", pipeline.run_analysis({"audio": up(w[0])}))
show("voice same speaker", pipeline.run_analysis({"audio": up(w[1]), "ref_voice": up(w[0])}))
show("voice diff speaker", pipeline.run_analysis({"audio": up(w2[1]), "ref_voice": up(w[0])}))
ub = uy[uy.label == "spoof"].abs_path.iloc[0]; show("audio Uyghur spoof sample (pipeline test only)", pipeline.run_analysis({"audio": up(ub)}))
# 4 video+audio consistency (TIMIT avi has its own audio)
avi = glob.glob(str(config.CACHE_DIR / "tmit_probe*"))  # placeholder
# 5 screenshot with text + QR
img = Image.new("RGB", (900, 260), "white"); d = ImageDraw.Draw(img)
try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
except Exception: f = None
d.text((20, 30), "Congratulations! You have won a cash prize.", fill="black", font=f)
d.text((20, 90), "Pay Rs. 999 registration fee to claimprize@ybl today", fill="black", font=f)
d.text((20, 150), "Call 9876543210 or visit www.prize-claim.xyz now", fill="black", font=f)
buf = io.BytesIO(); img.save(buf, "PNG"); show("screenshot OCR", pipeline.run_analysis({"images": [("shot.png", buf.getvalue())]}))
import pandas as pd
q = da.load_qr_urls(); print("\nQR decode rate:", (q.url != "").mean())
# 6 invalid inputs
show("corrupt video", pipeline.run_analysis({"video": ("x.mp4", b"not a video at all")}))
show("bad extension", pipeline.run_analysis({"audio": ("x.exe", b"abc"), "text": "hi there friend how are you today"}))
# 7 report
a = pipeline.run_analysis({"text": "Send Rs 500 to a.b@oksbi, call 9876543210", "audio": up(w[0])})
r = report.build_report(a); print("\nREPORT masked sample:", json.dumps(r["module_results"]["text"]["indicators"])[:300])
assert "9876543210" not in report.to_json(r) and "a.b@oksbi" not in report.to_json(r)
report.to_csv(r); report.to_text(r); print("report json/csv/text OK; media stored: no")
import os, tempfile; print("temp dirs left:", [p for p in glob.glob(tempfile.gettempdir() + "/fraudlens_*")])
