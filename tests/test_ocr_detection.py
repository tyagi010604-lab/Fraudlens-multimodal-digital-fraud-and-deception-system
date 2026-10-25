"""OCR discovery / error handling / OCR->text-analysis wiring (python tests/test_ocr_detection.py)."""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from src import pipeline, text_ocr

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:110]}]" if detail else ""))
    if not cond:
        fails.append(name)


real = shutil.which("tesseract")
check("a real tesseract exists in this test machine", real is not None, real)

# 1) Simulated Windows layout: tesseract is NOT on PATH, only in %ProgramFiles%\Tesseract-OCR\tesseract.exe
tmp = Path(tempfile.mkdtemp())
exe_dir = tmp / "Tesseract-OCR"
exe_dir.mkdir()
(exe_dir / "tesseract.exe").symlink_to(real)
code = ("import sys,os;sys.path.insert(0,r'%s');from src import text_ocr as t;"
        "t.tesseract_candidates=lambda c=t.tesseract_candidates:[x for x in c() if x[1]!='/usr/bin/tesseract' and x[0]!='PATH' and 'Linux' not in x[0]];"
        "ok,m=t.check_tesseract();print(ok);print(m)" % ROOT)
env = {k: v for k, v in os.environ.items() if k != "TESSERACT_CMD"}
env.update({"PATH": str(Path(sys.executable).parent), "ProgramFiles": str(tmp)})
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.strip().splitlines()
check("auto-detects Tesseract in a standard Program Files folder when not on PATH", out and out[0] == "True" and "standard Windows folder" in out[-1], out[-1] if out else "")

# 2) explicit TESSERACT_CMD env var
env2 = dict(env, TESSERACT_CMD=str(exe_dir / "tesseract.exe"))
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2).stdout.strip().splitlines()
check("TESSERACT_CMD environment variable honoured", out and out[0] == "True" and "environment variable" in out[-1], out[-1] if out else "")

# 3) bad TESSERACT_CMD and nothing else installed -> clear message, no crash
code_missing = ("import sys;sys.path.insert(0,r'%s');from src import text_ocr as t;"
                "t.tesseract_candidates=lambda:[('environment variable TESSERACT_CMD', r'C:\\\\nope\\\\tesseract.exe')];"
                "ok,m=t.check_tesseract();print(ok);print(m);r=t.ocr_image(Image.new('RGB',(50,50),'white')) if False else None" % ROOT)
out = subprocess.run([sys.executable, "-c", code_missing], capture_output=True, text=True, env=env2).stdout
check("missing/incorrect path -> available=False with setup steps", out.startswith("False") and "does not exist" in out and "UB-Mannheim" in out and "Program Files" in out)

# 4) OCR gracefully unavailable inside the app pipeline (patched in-process)
orig = text_ocr.tesseract_candidates
text_ocr._TESS.update(cmd=None, msg=None)
text_ocr.tesseract_candidates = lambda: []
img = Image.new("RGB", (600, 120), "white"); b = io.BytesIO(); img.save(b, "PNG")
r = pipeline.run_analysis({"images": [("x.png", b.getvalue())]})
o = r["results"]["ocr"][0]
check("no Tesseract: image analysis returns 'unavailable' with instructions (no crash)", o["status"] == "unavailable" and "Tesseract" in o["reason"], o["status"])
check("no Tesseract: dashboard status says unavailable", any(m["module"].startswith("6.") and m["status"] == "unavailable" for m in r["module_status"]))
text_ocr.tesseract_candidates = orig
text_ocr._TESS.update(cmd=None, msg=None)

# 5) OCR works and the extracted text reaches the text-analysis pipeline
try:
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except Exception:
    f = None
MSG = ["URGENT: Your bank account will be blocked today.", "Complete your KYC immediately using this link and",
       "enter your account details to avoid suspension."]
img = Image.new("RGB", (1000, 200), "white"); d = ImageDraw.Draw(img)
for i, line in enumerate(MSG):
    d.text((20, 20 + 50 * i), line, fill="black", font=f)
b = io.BytesIO(); img.save(b, "PNG")
r = pipeline.run_analysis({"images": [("shot.png", b.getvalue())], "text": " ".join(MSG)})
o = r["results"]["ocr"][0]
check("OCR status ok", o["status"] == "ok", o.get("reason"))
check("OCR extracted the screenshot text", "kyc" in o["extracted_text"].lower() and "blocked" in o["extracted_text"].lower(), o["extracted_text"][:80])
ta = o.get("text_analysis")
check("extracted text reached the text-analysis pipeline (model + rules + assessment)", ta and "assessment" in ta and "model" in ta and "rules" in ta)
check("OCR text of a scam screenshot is assessed Suspicious", ta and ta["assessment"]["label"] == "Suspicious", ta["assessment"]["label"] if ta else "")
fu = r["fusion"]
check("typed + OCR copy of the same text -> ONE message_content contribution", sum(c["module"] == "message_content" for c in fu["contributions"]) == 1, [c["module"] for c in fu["contributions"]])
check("fusion explanation notes the duplicate was not double counted", any("not double counted" in e for e in fu["explanation"]))

# 6) corrupt image
r = pipeline.run_analysis({"images": [("bad.png", b"this is not an image")]})
o = r["results"]["ocr"][0]
check("corrupt image handled gracefully", o["status"] in ("error", "no_text") and "reason" in o, o["status"])
r = pipeline.run_analysis({"images": [("bad.exe", b"abc")], "text": "hello there how are you today my friend"})
check("unsupported image type rejected with a message", any("Unsupported" in v for v in r["input_errors"].values()), r["input_errors"])
print("\nFAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
