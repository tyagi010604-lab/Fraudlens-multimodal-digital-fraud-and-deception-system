"""Headless Streamlit test: renders every page, with and without an analysis (python tests/test_app_pages.py)."""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from src import pipeline
from PIL import Image, ImageDraw
img = Image.new("RGB", (800, 200), "white"); ImageDraw.Draw(img).text((20, 40), "You won! Pay Rs 500 fee to x@ybl http://a-b.xyz call 9876543210 urgent", fill="black")
b = io.BytesIO(); img.save(b, "PNG")
d = Path("/tmp/t/DeepfakeTIMIT/higher_quality/fadg0")
analysis = pipeline.run_analysis({"text": "URGENT pay Rs 499 fee to ravi@oksbi or lose account http://x.xyz",
    "images": [("s.png", b.getvalue())], "video": ((d/'sa1-video-fram1.avi').name, (d/'sa1-video-fram1.avi').read_bytes()),
    "audio": ("sa1.wav", (d/'sa1.wav').read_bytes()), "ref_voice": ("sa2.wav", (d/'sa2.wav').read_bytes()), "transcript": "She had your dark suit in greasy wash water all year."})
ok = True
for with_a in (False, True):
    for page in ["Home / Dashboard", "Analyse Evidence", "Individual Results", "Overall Risk Assessment", "Model Evaluation", "Dataset Information", "Report"]:
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)
        at.session_state["page"] = page
        if with_a: at.session_state["analysis"] = analysis
        at.run()
        err = [e.value for e in at.exception]
        print(f"{'with' if with_a else 'no  '} analysis | {page:26s} | exceptions: {err if err else 'none'}")
        ok &= not err
print("ALL PAGES OK" if ok else "FAILURES")
