"""Streamlit dashboard for the local AI-assisted DR screening prototype."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from src.config.settings import load_config
from src.inference.predict import CheckpointNotFoundError, InferenceEngine
from src.reporting.report import generate_report

SEVERITY_LABELS = ("No DR", "Mild", "Moderate", "Severe", "PDR")


@st.cache_resource(show_spinner="Loading the local model...")
def load_engine(config_path: str, checkpoint_path: str) -> InferenceEngine:
    return InferenceEngine(load_config(config_path), checkpoint_path)


def severity_scale(predicted_class: int) -> str:
    stages = "".join(
        f'<div class="severity-stage {"current" if index == predicted_class else ""}">'
        f'<span class="severity-number">{index + 1}</span><span>{label}</span></div>'
        for index, label in enumerate(SEVERITY_LABELS)
    )
    return f'<div class="severity-scale">{stages}</div><div class="severity-arrow">CURRENT MODEL RESULT</div>'


def read_verified_metrics(project_root: Path) -> dict[str, Any] | None:
    path = project_root / "artifacts" / "metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    st.set_page_config(page_title="AI Diabetic Retinopathy Screening", page_icon="+", layout="wide")
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: #f7f9f8; }
    [data-testid="stSidebar"] { background: #eef3f1; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 2rem; }
    .eyebrow { color: #177e89; font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .title { color: #1e3030; font-size: 2rem; font-weight: 700; margin: .25rem 0 .15rem; }
    .subtitle { color: #5b6c69; margin-bottom: 1.3rem; }
    .result-panel { background: #fff; border: 1px solid #d6e1dd; border-radius: 10px; padding: 1.25rem 1.35rem; }
    .result-label { color: #667571; font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .result-name { color: #1e3030; font-size: 2rem; font-weight: 700; margin-top: .25rem; }
    .severity-scale { display: grid; grid-template-columns: repeat(5, 1fr); gap: .45rem; margin: .9rem 0 .35rem; }
    .severity-stage { background: #fff; border: 1px solid #c9d8d3; border-radius: 7px; color: #526461; padding: .65rem .2rem; text-align: center; font-size: .78rem; }
    .severity-stage.current { background: #177e89; border-color: #177e89; color: #fff; font-weight: 700; }
    .severity-number { display: block; font-size: .7rem; margin-bottom: .15rem; opacity: .8; }
    .severity-arrow { color: #177e89; font-size: .68rem; font-weight: 700; letter-spacing: .06em; text-align: center; }
    .notice { background: #fff8e8; border-left: 3px solid #c18c26; color: #5e4e29; padding: .8rem 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Research / demonstration prototype</div><div class="title">AI Diabetic Retinopathy Screening</div><div class="subtitle">Explainable AI screening with quality control and specialist referral prioritization.</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Local screening setup")
        config_path = st.text_input("Configuration", "config.yaml")
        checkpoint_path = st.text_input("Model checkpoint", "artifacts/checkpoints/best_model.pt")
        st.caption("All inference runs locally. No image or result is sent to an external service.")
        with st.expander("Referable rule & safety"):
            st.write("Grades 0-1 are non-referable; grades 2-4 are referable. This is an AI-assisted screening result, not a diagnosis or a substitute for ophthalmologist examination.")

    st.subheader("Upload Fundus Image")
    uploaded = st.file_uploader("PNG / JPG / JPEG", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed")
    if uploaded is None:
        st.info("Choose a retinal fundus image. Uploading only previews the image; analysis starts when you press Run Screening.")
        st.markdown('<div class="notice"><b>Medical safety:</b> This system is an AI-assisted screening prototype for research/demo purposes and is not a substitute for examination or diagnosis by a qualified ophthalmologist.</div>', unsafe_allow_html=True)
        return

    image_bytes = uploaded.getvalue()
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            preview = opened.convert("RGB")
    except (OSError, ValueError) as error:
        st.error(f"The uploaded image could not be opened: {error}")
        return

    image_signature = hashlib.sha256(image_bytes).hexdigest()
    analysis_key = f"{image_signature}:{config_path}:{checkpoint_path}"
    if st.session_state.get("analysis_key") != analysis_key:
        st.session_state["analysis_key"] = analysis_key
        st.session_state.pop("screening_result", None)

    st.image(preview, caption="Uploaded fundus image preview", width="stretch")
    st.success("Image ready for screening")
    run_screening = st.button("Run Screening", type="primary", width="stretch")
    if run_screening:
        try:
            with st.spinner("Running quality gate, model inference, and Grad-CAM..."):
                engine = load_engine(config_path, checkpoint_path)
                result = engine.predict(image_bytes, write_report_file=False)
            st.session_state["screening_result"] = result
        except CheckpointNotFoundError as error:
            st.error(str(error))
            st.info("Train or place the local five-class checkpoint at the configured path before running screening.")
            return
        except Exception as error:
            st.error(f"Screening could not be completed safely: {error}")
            return

    result = st.session_state.get("screening_result")
    if not result:
        return

    st.divider()
    st.subheader("Image Quality")
    quality_columns = st.columns(4)
    quality_columns[0].metric("Quality score", f"{result['quality_score']:.1f}/100")
    quality_columns[1].metric("Status", result["quality_status"])
    quality_columns[2].metric("Blur variance", f"{result['quality_metrics']['blur_variance']:.1f}")
    quality_columns[3].metric("Field of view", f"{result['quality_metrics']['field_of_view_ratio']:.1%}")
    if result["quality_status"] != "ACCEPT":
        st.error("Image quality insufficient")
        st.write("Please upload another fundus image. " + "; ".join(result["quality_reasons"]))
        return
    st.success("Suitable for AI-assisted screening")

    st.subheader("Screening Result")
    st.markdown(f'<div class="result-panel"><div class="result-label">AI-assisted screening result</div><div class="result-name">{result["predicted_class_name"]}</div><p><b>Confidence:</b> {result["confidence"]:.1%}</p>{severity_scale(int(result["predicted_class"]))}</div>', unsafe_allow_html=True)
    result_columns = st.columns(2)
    result_columns[0].metric("Referable DR", "YES" if result["referable"] else "NO")
    result_columns[1].write("**Recommendation**\n\n" + result["recommendation"])
    if result["confidence_is_low"]:
        st.warning("LOW CONFIDENCE: model confidence is not clinical certainty. Specialist review is recommended.")

    st.subheader("Explainable AI")
    st.caption("Highlighted regions indicate areas that contributed to the model's prediction. Grad-CAM is an attention/influence visualization, not proof of a specific lesion.")
    explanation_columns = st.columns(2)
    explanation_columns[0].image(image_bytes, caption="Original Fundus", width="stretch")
    explanation_columns[1].image(result["gradcam_paths"]["overlay"], caption="AI Focus / Grad-CAM overlay", width="stretch")

    st.subheader("Screening Recommendation")
    st.info(result["recommendation"])

    with st.expander("Advanced Model Details"):
        probability_table = {"DR grade": list(result["probabilities"].keys()), "Probability": [f"{value:.1%}" for value in result["probabilities"].values()]}
        st.table(probability_table)
        status = result["model_status"]
        detail_columns = st.columns(4)
        detail_columns[0].metric("Architecture", status["architecture"])
        detail_columns[1].metric("Device", result["device"].upper())
        detail_columns[2].metric("Classes", str(status["classes"]))
        detail_columns[3].metric("Input", f'{status["input_size"]} x {status["input_size"]}')
        metrics = read_verified_metrics(load_config(config_path).project_root)
        if metrics:
            st.markdown("**Verified test-set performance (separate from individual confidence)**")
            performance_columns = st.columns(3)
            performance_columns[0].metric("Test accuracy", f"{metrics['accuracy']:.1%}")
            performance_columns[1].metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
            performance_columns[2].metric("QWK", f"{metrics['quadratic_weighted_kappa']:.4f}")

    st.subheader("Screening Report")
    report_paths = result.get("report_paths") or {}
    if st.button("Generate Screening Report", type="secondary"):
        try:
            config = load_config(config_path)
            report_paths = generate_report(result, config.project_root / "artifacts" / "reports")
            result["report_paths"] = report_paths
            st.session_state["screening_result"] = result
            st.success("Screening report generated from the current result.")
        except Exception as error:
            st.error(f"The report could not be generated: {error}")
    if report_paths:
        st.download_button("Download HTML report", data=Path(report_paths["html"]).read_bytes(), file_name=f"{result['case_id']}.html", mime="text/html")
        st.download_button("Download JSON report", data=Path(report_paths["json"]).read_bytes(), file_name=f"{result['case_id']}.json", mime="application/json")

    st.markdown('<div class="notice"><b>Disclaimer:</b> This system is an AI-assisted screening prototype for research/demo purposes. It is not a substitute for examination or diagnosis by a qualified ophthalmologist, and it has not been clinically validated.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
