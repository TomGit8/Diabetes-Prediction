"""
Streamlit application to test the trained diabetes prediction model.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_DATA_PATH = Path("diabetes.csv")


def load_config() -> Dict:
    """Load training configuration if it exists, otherwise fall back to defaults."""
    config_path = DEFAULT_ARTIFACT_DIR / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        config["artifact_dir"] = Path(config.get("artifact_dir", DEFAULT_ARTIFACT_DIR))
    else:
        config = {
            "artifact_dir": DEFAULT_ARTIFACT_DIR,
            "data_path": str(DEFAULT_DATA_PATH),
            "target_column": "Outcome",
        }
    return config


@st.cache_resource
def load_model(model_path: Path):
    """Load the serialized estimator from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Aucun modèle trouvé. Entraînez d'abord via `python model.py`."
        )
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_feature_stats(data_path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset statistics to drive the UI defaults."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable à {data_path}. Assurez-vous que diabetes.csv est disponible."
        )
    df = pd.read_csv(data_path)
    feature_df = df.drop(columns=[target_column])
    stats = feature_df.describe().T
    return feature_df, stats


def build_input_form(stats: pd.DataFrame) -> pd.DataFrame | None:
    """Render the feature input form and return a DataFrame or None if not submitted."""
    st.subheader("Paramètres patient")
    columns = list(stats.index)
    col1, col2 = st.columns(2)
    user_values = {}
    with st.form("prediction_form"):
        for idx, feature in enumerate(columns):
            container = col1 if idx % 2 == 0 else col2
            with container:
                min_val = float(stats.loc[feature, "min"])
                max_val = float(stats.loc[feature, "max"])
                median_val = float(stats.loc[feature, "50%"])
                step = max((max_val - min_val) / 100, 0.01)
                user_values[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=float(np.clip(median_val, min_val, max_val)),
                    step=step,
                    format="%.1f",
                )
        submitted = st.form_submit_button("Prédire")
    if not submitted:
        return None
    data = pd.DataFrame([user_values])
    return data


def display_prediction(probability: float, threshold: float = 0.5) -> None:
    """Display a styled prediction card."""
    label = "Diabétique" if probability >= threshold else "Non diabétique"
    st.markdown("### Résultat")
    st.metric("Probabilité estimée", f"{probability * 100:.1f} %")
    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:1rem;
                    background:linear-gradient(135deg,#1d976c,#93f9b9);
                    color:white;font-size:1.1rem;">
            <strong>Conclusion :</strong> {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
    st.title("🩺 Prédicteur de diabète")
    st.caption("Modèle entraîné hors-ligne via le pipeline `model.py`.")

    config = load_config()
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(exist_ok=True)
    joblibs = sorted(artifact_dir.glob("*.joblib"))
    if not joblibs:
        st.error(
            "Aucun fichier modèle détecté. Veuillez exécuter `python model.py` pour générer un artefact."
        )
        return
    model_path = joblibs[0]
    model = load_model(model_path)

    feature_df, stats = load_feature_stats(
        Path(config.get("data_path", DEFAULT_DATA_PATH)),
        config.get("target_column", "Outcome"),
    )
    st.sidebar.header("Infos modèle")
    st.sidebar.write(f"Artefact: `{model_path.name}`")
    st.sidebar.write(f"Features: {', '.join(feature_df.columns)}")

    input_df = build_input_form(stats)
    if input_df is None:
        st.info("Complétez les champs ci-dessus puis cliquez sur *Prédire*.")
        return

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[:, 1][0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(input_df)
        probabilities = 1 / (1 + np.exp(-scores))[0]
    else:
        probabilities = float(model.predict(input_df)[0])

    display_prediction(float(probabilities))


if __name__ == "__main__":
    main()
