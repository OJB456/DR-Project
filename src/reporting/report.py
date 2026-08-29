"""Downloadable, honest HTML/JSON screening reports."""
from __future__ import annotations

import html
import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DISCLAIMER = "This research prototype provides AI-assisted screening only. It is not a substitute for professional medical diagnosis, has not been clinically validated here, and must be reviewed by a qualified specialist."


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_report(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write a JSON report and a self-contained HTML report from actual inference values."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    case_id = str(result.get("case_id", "DR-screening"))
    serializable = {key: value for key, value in result.items() if key not in {"heatmap"}}
    json_path = directory / f"{case_id}.json"
    html_path = directory / f"{case_id}.html"
    json_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    probabilities = result.get("probabilities", {})
    probability_rows = "".join(f"<tr><td>{html.escape(str(name))}</td><td>{_percent(float(value))}</td></tr>" for name, value in probabilities.items())
    gradcam = result.get("gradcam_paths", {})
    overlay_path = Path(str(gradcam.get("overlay", ""))) if gradcam.get("overlay") else None
    overlay = ""
    if overlay_path and overlay_path.is_file():
        encoded = base64.b64encode(overlay_path.read_bytes()).decode("ascii")
        overlay = f"data:image/png;base64,{encoded}"
    generated = result.get("timestamp", datetime.now(timezone.utc).isoformat())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>DR Screening Report {html.escape(case_id)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:820px;margin:36px auto;color:#173b3f}} table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d9e4e1;padding:9px;text-align:left}} .flag{{font-size:1.3rem;font-weight:bold;color:#b33838}} .ok{{color:#177e70}} img{{max-width:100%;border-radius:10px}} .disclaimer{{background:#fff8e7;padding:16px;border-left:4px solid #d19a2a}}</style></head>
<body><h1>Diabetic Retinopathy AI Screening Report</h1>
<p><b>Case ID:</b> {html.escape(case_id)}<br><b>Timestamp:</b> {html.escape(str(generated))}</p>
<h2>Screening result</h2><p><b>Image quality:</b> {float(result.get("quality_score", 0)):.1f}/100 — {html.escape(str(result.get("quality_status", "UNKNOWN")))}</p>
<p><b>Predicted grade:</b> {html.escape(str(result.get("predicted_class_name", "Unavailable")))}<br><b>Model confidence:</b> {_percent(float(result.get("confidence", 0)))}<br><b>Referable:</b> <span class="{'flag' if result.get('referable') else 'ok'}">{'YES' if result.get('referable') else 'NO'}</span></p>
<p><b>Recommendation:</b> {html.escape(str(result.get("recommendation", "")))}</p>
<h2>Class probabilities</h2><table><tr><th>Grade</th><th>Probability</th></tr>{probability_rows}</table>
{f'<h2>Grad-CAM explanation</h2><p>Highlighted regions show image areas that most influenced the model prediction; they are not proof that disease is present.</p><img src="{overlay}" alt="Grad-CAM overlay">' if overlay else ''}
<p class="disclaimer"><b>Disclaimer:</b> {html.escape(DISCLAIMER)}</p></body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}
