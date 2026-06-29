from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
STARTER_DIR = APP_DIR.parent
PROJECT_DIR = STARTER_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predict import (  # noqa: E402
    extract_sequence_for_path,
    load_labels,
    predict_sequence_sklearn,
    predict_sequence_torch,
)
from src.hybrid_recognition import blend_probabilities, predict_sequence_template, rule_based_scores  # noqa: E402


def top_k(labels: List[str], proba: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    idx = np.argsort(-proba)[:k]
    return [(labels[i], float(proba[i])) for i in idx]


def main() -> None:
    st.set_page_config(page_title="NZSL Starter Prototype", layout="centered")
    st.title("NZSL Starter Recognition Prototype")
    st.caption("Transport and daily communication starter vocabulary built from MediaPipe keypoints.")

    st.sidebar.header("Settings")
    model_type = st.sidebar.selectbox("Inference mode", ["sklearn", "lstm", "hybrid", "template", "transformer"])
    hybrid_base_model = st.sidebar.selectbox("Hybrid ML base", ["sklearn", "lstm", "transformer"]) if model_type == "hybrid" else None
    active_model_type = hybrid_base_model if model_type == "hybrid" else model_type
    if active_model_type == "sklearn":
        default_model = STARTER_DIR / "models_cv7_all" / "sklearn_rf_cv7_all.joblib"
    elif active_model_type == "transformer":
        default_model = STARTER_DIR / "models_transformer_cv7_all" / "final_all" / "best_transformer.pt"
    else:
        default_model = STARTER_DIR / "models_test_in_train" / "best_lstm.pt"
    model_path = st.sidebar.text_input("Model path", value=str(default_model))
    processed_dir = st.sidebar.text_input("Processed dir", value=str(STARTER_DIR / "data" / "processed_test_in_train"))
    num_frames = st.sidebar.number_input("Frames", min_value=1, max_value=240, value=60, step=1)
    device_options = ["cpu", "cuda"] if __import__("torch").cuda.is_available() else ["cpu"]
    device = st.sidebar.selectbox("Device", device_options)
    conf_threshold = st.sidebar.slider("Low-confidence threshold", 0.0, 1.0, 0.5, 0.05)

    st.write("Upload a short NZSL video clip. The app extracts MediaPipe Holistic keypoints and predicts a starter label.")
    if model_type == "hybrid":
        st.caption("Hybrid mode blends the trained model, a template-matching scorer, and a small rule-based gesture scorer.")

    uploaded = st.file_uploader("Upload .mp4 / .jpg / .png", type=["mp4", "jpg", "jpeg", "png"])
    if uploaded is None:
        return

    tmp_dir = STARTER_DIR / ".streamlit_tmp"
    tmp_dir.mkdir(exist_ok=True)
    input_path = tmp_dir / uploaded.name
    input_path.write_bytes(uploaded.getbuffer())

    mp = Path(model_path).resolve()
    if not mp.exists():
        st.error(f"Model not found: {mp}")
        return
    processed_path = Path(processed_dir).resolve()
    if model_type in {"template", "hybrid"} and not processed_path.exists():
        st.error(f"Processed dir not found: {processed_path}")
        return

    labels = load_labels(mp.parent)
    with st.spinner("Extracting keypoints..."):
        seq = extract_sequence_for_path(input_path, num_frames=int(num_frames))

    parts = {}
    with st.spinner("Predicting..."):
        if model_type == "sklearn":
            pred, proba = predict_sequence_sklearn(mp, seq, labels=labels)
        elif model_type in {"lstm", "transformer"}:
            pred, proba = predict_sequence_torch(model_type, mp, seq, labels=labels, device=device)
        elif model_type == "template":
            pred, proba = predict_sequence_template(seq, labels=labels, processed_dir=processed_path)
        else:
            if hybrid_base_model == "sklearn":
                _, ml_proba = predict_sequence_sklearn(mp, seq, labels=labels)
            else:
                _, ml_proba = predict_sequence_torch(hybrid_base_model or "lstm", mp, seq, labels=labels, device=device)
            _, template_proba = predict_sequence_template(seq, labels=labels, processed_dir=processed_path)
            rule_proba = rule_based_scores(seq, labels=labels)
            pred, proba, parts = blend_probabilities(
                labels=labels,
                ml_proba=ml_proba,
                template_proba=template_proba,
                rule_proba=rule_proba,
            )

    best_p = float(np.max(proba))
    st.subheader(f"Predicted sign: {pred}")
    st.caption(f"Confidence: {best_p:.3f}")
    if best_p < conf_threshold:
        st.warning("Low confidence. Treat this as a hint, not a trusted translation.")

    st.write("Top-5 probabilities")
    st.table([{"label": lab, "prob": p} for lab, p in top_k(labels, proba, k=5)])
    if parts:
        st.write("Hybrid components")
        for name, component_proba in parts.items():
            st.caption(name)
            st.table([{"label": lab, "prob": p} for lab, p in top_k(labels, component_proba, k=3)])


if __name__ == "__main__":
    main()
