from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
STARTER_DIR = APP_DIR.parent
PROJECT_DIR = STARTER_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.hybrid_recognition import blend_probabilities, predict_sequence_template, rule_based_scores  # noqa: E402
from src.predict import (  # noqa: E402
    extract_sequence_for_path,
    load_labels,
    predict_sequence_sklearn,
    predict_sequence_torch,
)


MODE_LABELS: Dict[str, str] = {
    "transformer": "Transformer",
    "sklearn": "Random Forest",
    "lstm": "LSTM",
    "hybrid": "Hybrid",
    "template": "Template matcher",
}


@dataclass(frozen=True)
class ModelChoice:
    mode: str
    label: str
    path: Optional[Path] = None
    available: bool = False


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        div[data-testid="stFileUploader"] {
            padding: 0.5rem 0;
        }
        div[data-testid="stFileUploader"] section {
            border: 1px dashed #bfd1df;
            border-radius: 8px;
            background: #f8fbfd;
        }
        .hero-copy {
            color: #52616f;
            font-size: 1.02rem;
            margin-top: -0.25rem;
            max-width: 760px;
        }
        .panel {
            border: 1px solid #e3e8ee;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            min-height: 122px;
        }
        .result-card {
            border: 1px solid #dce7ef;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
            margin-bottom: 0.85rem;
        }
        .result-label {
            color: #536574;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .result-value {
            color: #102a43;
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.2;
            margin: 0.18rem 0 0.45rem;
        }
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            background: #e6f4ea;
            color: #146c43;
            font-size: 0.86rem;
            font-weight: 700;
        }
        .confidence-badge.low {
            background: #fff3cd;
            color: #8a5b00;
        }
        .rank-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin: 0.7rem 0 0.25rem;
        }
        .rank-name {
            color: #1f2933;
            font-weight: 650;
        }
        .rank-score {
            color: #52616f;
            font-variant-numeric: tabular-nums;
        }
        .muted {
            color: #697b8c;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def readable_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


def top_k(labels: List[str], proba: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    idx = np.argsort(-proba)[:k]
    return [(labels[i], float(proba[i])) for i in idx]


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def is_inside_ignored_dir(path: Path) -> bool:
    ignored = {".git", ".venv", "__pycache__", ".streamlit_tmp"}
    return any(part in ignored for part in path.parts)


def expected_model_path(model_type: str) -> Path:
    defaults = {
        "transformer": STARTER_DIR / "models_transformer_cv7_all" / "final_all" / "best_transformer.pt",
        "sklearn": STARTER_DIR / "models_cv7_all" / "sklearn_rf_cv7_all.joblib",
        "lstm": STARTER_DIR / "models_test_in_train" / "best_lstm.pt",
    }
    return defaults.get(model_type, Path(""))


def compatible_artifacts(suffixes: Tuple[str, ...], name_terms: Tuple[str, ...] = ()) -> List[Path]:
    artifacts: List[Path] = []
    for path in PROJECT_DIR.rglob("*"):
        if not path.is_file() or is_inside_ignored_dir(path):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        lower_name = path.name.lower()
        if name_terms and not any(term in lower_name for term in name_terms):
            continue
        if find_labels_dir(path) is None:
            continue
        artifacts.append(path)
    return sorted(artifacts, key=lambda p: (len(p.parts), str(p).lower()))


def find_labels_dir(model_path: Path) -> Optional[Path]:
    for candidate in [model_path.parent, *model_path.parents]:
        if is_inside_ignored_dir(candidate):
            continue
        if (candidate / "labels.json").exists():
            return candidate
        if candidate == PROJECT_DIR:
            break
    return None


def discover_model_choices() -> List[ModelChoice]:
    transformer = first_existing(
        [
            expected_model_path("transformer"),
            *compatible_artifacts((".pt", ".pth"), ("transformer",)),
        ]
    )
    sklearn = first_existing(
        [
            expected_model_path("sklearn"),
            *compatible_artifacts((".joblib", ".pkl", ".pickle", ".sav"), ("sklearn", "rf", "random", "forest")),
        ]
    )
    lstm = first_existing(
        [
            expected_model_path("lstm"),
            *compatible_artifacts((".pt", ".pth"), ("lstm",)),
        ]
    )
    processed = STARTER_DIR / "data" / "processed"
    template_available = (processed / "labels.csv").exists() and (processed / "sequences").exists()

    return [
        ModelChoice("transformer", "Transformer model", transformer, transformer is not None),
        ModelChoice("sklearn", "Random Forest model", sklearn, sklearn is not None),
        ModelChoice("lstm", "LSTM model", lstm, lstm is not None),
        ModelChoice("template", "Template matcher", None, template_available),
    ]


def default_choice(choices: List[ModelChoice]) -> ModelChoice:
    for mode in ("transformer", "sklearn", "lstm", "template"):
        for choice in choices:
            if choice.mode == mode and choice.available:
                return choice
    return ModelChoice("template", "Template matcher", None, False)


def choice_for_mode(choices: List[ModelChoice], mode: str) -> Optional[ModelChoice]:
    return next((choice for choice in choices if choice.mode == mode), None)


def mode_option_label(mode: str, choices: List[ModelChoice]) -> str:
    label = MODE_LABELS.get(mode, mode)
    if mode == "hybrid":
        trained_available = any(choice.available for choice in choices if choice.mode in {"sklearn", "lstm", "transformer"})
        template_available = bool(choice_for_mode(choices, "template") and choice_for_mode(choices, "template").available)
        return label if trained_available and template_available else f"{label} (missing files)"

    choice = choice_for_mode(choices, mode)
    if choice is None or choice.available:
        return label
    return f"{label} (missing file)"


def labels_from_processed_dir(processed_dir: Path) -> List[str]:
    labels_path = processed_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")
    labels = set()
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            if label:
                labels.add(str(label))
    if not labels:
        raise RuntimeError(f"No labels found in {labels_path}")
    return sorted(labels)


def torch_device_options() -> List[str]:
    try:
        import torch

        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    except Exception:
        return ["cpu"]


def render_preview(uploaded) -> None:
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.getvalue()
    if suffix == ".mp4":
        st.video(data)
    elif suffix in {".jpg", ".jpeg", ".png"}:
        st.image(data, use_column_width=True)
    else:
        st.info("Preview is not available for this file type.")


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel">
            <div class="result-label">Ready</div>
            <p class="muted">
            Upload a short NZSL clip or image to run the recognition pipeline.
            Results will appear here after keypoint extraction and inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(pred: str, proba: np.ndarray, labels: List[str], conf_threshold: float) -> None:
    best_p = float(np.max(proba)) if len(proba) else 0.0
    badge_class = "confidence-badge low" if best_p < conf_threshold else "confidence-badge"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted sign</div>
            <div class="result-value">{readable_label(pred)}</div>
            <span class="{badge_class}">{best_p * 100:.1f}% confidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if best_p < conf_threshold:
        st.warning("Low confidence. Treat this as a hint, not a trusted translation.")

    st.markdown("#### Top matches")
    for rank, (label, probability) in enumerate(top_k(labels, proba, k=5), start=1):
        st.markdown(
            f"""
            <div class="rank-row">
                <span class="rank-name">{rank}. {readable_label(label)}</span>
                <span class="rank-score">{probability * 100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(np.clip(probability, 0.0, 1.0)))


def render_components(parts: Dict[str, np.ndarray], labels: List[str]) -> None:
    if not parts:
        return
    with st.expander("Hybrid score breakdown"):
        for name, component_proba in parts.items():
            st.markdown(f"**{name.title()} component**")
            for label, probability in top_k(labels, component_proba, k=3):
                st.caption(f"{readable_label(label)} - {probability * 100:.1f}%")
                st.progress(float(np.clip(probability, 0.0, 1.0)))


def run_prediction(
    input_path: Path,
    model_type: str,
    model_path: Optional[Path],
    processed_dir: Path,
    num_frames: int,
    device: str,
    hybrid_base_model: str,
) -> Tuple[str, np.ndarray, List[str], Dict[str, np.ndarray]]:
    if model_type == "template":
        labels = labels_from_processed_dir(processed_dir)
    else:
        if model_path is None:
            raise FileNotFoundError("No model path was provided.")
        labels_dir = find_labels_dir(model_path)
        if labels_dir is None:
            raise FileNotFoundError(f"Missing labels.json for {model_path}")
        labels = load_labels(labels_dir)

    seq = extract_sequence_for_path(input_path, num_frames=num_frames)
    parts: Dict[str, np.ndarray] = {}

    if model_type == "sklearn":
        pred, proba = predict_sequence_sklearn(model_path, seq, labels=labels)  # type: ignore[arg-type]
    elif model_type in {"lstm", "transformer"}:
        pred, proba = predict_sequence_torch(model_type, model_path, seq, labels=labels, device=device)  # type: ignore[arg-type]
    elif model_type == "template":
        pred, proba = predict_sequence_template(seq, labels=labels, processed_dir=processed_dir)
    else:
        if model_path is None:
            raise FileNotFoundError("No model path was provided.")
        if hybrid_base_model == "sklearn":
            _, ml_proba = predict_sequence_sklearn(model_path, seq, labels=labels)
        else:
            _, ml_proba = predict_sequence_torch(hybrid_base_model, model_path, seq, labels=labels, device=device)
        _, template_proba = predict_sequence_template(seq, labels=labels, processed_dir=processed_dir)
        rule_proba = rule_based_scores(seq, labels=labels)
        pred, proba, parts = blend_probabilities(
            labels=labels,
            ml_proba=ml_proba,
            template_proba=template_proba,
            rule_proba=rule_proba,
        )

    return pred, proba, labels, parts


def main() -> None:
    st.set_page_config(page_title="NZSL Recognition Demo", layout="wide")
    inject_css()

    st.title("NZSL Recognition Demo")
    st.markdown(
        '<p class="hero-copy">Upload a short New Zealand Sign Language clip or image. '
        "The demo extracts MediaPipe keypoints and returns the most likely starter-vocabulary sign.</p>",
        unsafe_allow_html=True,
    )

    choices = discover_model_choices()
    selected_default = default_choice(choices)
    processed_default = STARTER_DIR / "data" / "processed"

    with st.expander("Advanced settings", expanded=False):
        available_modes = [choice.mode for choice in choices if choice.available]
        if not available_modes:
            available_modes = ["template"]
        has_trained_model = any(choice.available for choice in choices if choice.mode in {"sklearn", "lstm", "transformer"})
        if has_trained_model and "template" in available_modes:
            available_modes = ["hybrid", *available_modes]
        show_unavailable = st.checkbox("Show unavailable model modes", value=False)
        mode_options = available_modes
        if show_unavailable:
            mode_options = mode_options + [
                mode for mode in ["hybrid", "sklearn", "lstm", "transformer"] if mode not in mode_options
            ]
        model_type = st.selectbox(
            "Inference mode",
            mode_options,
            index=mode_options.index(selected_default.mode) if selected_default.mode in mode_options else 0,
            format_func=lambda mode: mode_option_label(mode, choices),
        )

        hybrid_base_model = "sklearn"
        if model_type == "hybrid":
            hybrid_options = [choice.mode for choice in choices if choice.mode in {"sklearn", "transformer", "lstm"} and choice.available]
            if show_unavailable:
                hybrid_options = hybrid_options + [
                    mode for mode in ["sklearn", "transformer", "lstm"] if mode not in hybrid_options
                ]
            hybrid_base_model = st.selectbox(
                "Hybrid ML base",
                hybrid_options,
                format_func=lambda mode: mode_option_label(mode, choices),
            )

        active_model_type = hybrid_base_model if model_type == "hybrid" else model_type
        active_choice = next((choice for choice in choices if choice.mode == active_model_type), None)
        default_model_path = active_choice.path if active_choice and active_choice.path else expected_model_path(active_model_type)
        model_path_text = ""
        if model_type != "template":
            model_path_text = st.text_input(
                "Model path",
                value=str(default_model_path),
                key=f"model_path_{active_model_type}",
            )
            if not Path(model_path_text).exists():
                st.warning(
                    "This model file is not present yet. Train/export the model or place the artifact at this path "
                    "before running this inference mode."
                )

        processed_dir_text = st.text_input("Processed data directory", value=str(processed_default))
        num_frames = int(st.number_input("Frames to sample", min_value=1, max_value=240, value=60, step=1))
        device = st.selectbox("Device", torch_device_options())
        conf_threshold = float(st.slider("Low-confidence threshold", 0.0, 1.0, 0.5, 0.05))

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Input")
        uploaded = st.file_uploader("Upload an NZSL clip or image", type=["mp4", "jpg", "jpeg", "png"])
        if uploaded is None:
            st.markdown(
                '<p class="muted">Supported formats: MP4 video, JPG image, or PNG image.</p>',
                unsafe_allow_html=True,
            )
        else:
            render_preview(uploaded)

    with right:
        st.subheader("Recognition")
        if uploaded is None:
            render_empty_state()
            if not selected_default.available:
                st.info(
                    "No trained model files were found. Add a model artifact or keep template mode available with "
                    "`starter_nzsl/data/processed/labels.csv` and sequence files."
                )
            return

        tmp_dir = STARTER_DIR / ".streamlit_tmp"
        tmp_dir.mkdir(exist_ok=True)
        input_path = tmp_dir / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())

        processed_path = Path(processed_dir_text).resolve()
        model_path = Path(model_path_text).resolve() if model_path_text else None

        if model_type != "template" and (model_path is None or not model_path.exists()):
            st.warning(
                "Model file missing. The path is auto-filled to the expected output location, but the artifact "
                "is not present in this project yet."
            )
            if model_path is not None:
                st.caption(str(model_path))
            st.info("Use Template matcher for the current checked-in data, or train/copy the selected model file first.")
            return

        if model_type in {"template", "hybrid"} and not processed_path.exists():
            st.error("The processed data directory was not found.")
            st.caption(str(processed_path))
            return

        try:
            with st.spinner("Extracting keypoints and running recognition..."):
                pred, proba, labels, parts = run_prediction(
                    input_path=input_path,
                    model_type=model_type,
                    model_path=model_path,
                    processed_dir=processed_path,
                    num_frames=num_frames,
                    device=device,
                    hybrid_base_model=hybrid_base_model,
                )
        except Exception as exc:
            st.error("Recognition could not be completed.")
            st.caption(str(exc))
            return

        render_result(pred, proba, labels, conf_threshold)
        render_components(parts, labels)


if __name__ == "__main__":
    main()
