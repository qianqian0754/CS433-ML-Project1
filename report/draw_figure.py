from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

m = json.load(open(Path(__file__).resolve().parents[1] / "results/metrics.json"))
c = canvas.Canvas(
    str(Path(__file__).resolve().parent / "comparison.pdf"), pagesize=(460, 200)
)
c.setFont("Helvetica", 10)
for value in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    y = 45 + value * 260
    c.setStrokeColor(HexColor("#dddddd"))
    c.line(40, y, 450, y)
    c.setFillColor(HexColor("#222222"))
    c.drawRightString(34, y - 3, f"{value:.1f}")
for i, e in enumerate(m["experiments"]):
    x = 60 + i * 100
    h = e["holdout"]["f1"] * 260
    c.setFillColor(HexColor("#237c87" if i == 2 else "#8a9eb0"))
    c.rect(x, 45, 55, h, fill=1, stroke=0)
    c.setFillColor(HexColor("#222222"))
    c.drawCentredString(x + 27.5, 50 + h, f"{e['holdout']['f1']:.3f}")
for i, label in enumerate(["Ridge", "Numeric LR", "Category LR", "Category LR"]):
    c.drawCentredString(87.5 + i * 100, 29, label)
c.drawCentredString(387.5, 15, "stronger L2")
c.save()
