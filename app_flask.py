import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_DATA_PATH = Path("diabetes.csv")

def load_config():
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

def load_model(artifact_dir):
    joblibs = sorted(artifact_dir.glob("*.joblib"))
    if not joblibs:
        return None
    model_path = joblibs[0]
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model

def load_stats(data_path, target_column):
    if not data_path.exists():
        return {}
    df = pd.read_csv(data_path)
    feature_df = df.drop(columns=[target_column])
    stats = feature_df.describe().T
    
    # Convert stats to a dictionary for easier usage in Jinja2
    features_config = {}
    for col in feature_df.columns:
        features_config[col] = {
            "min": float(stats.loc[col, "min"]),
            "max": float(stats.loc[col, "max"]),
            "mean": float(stats.loc[col, "50%"]), # Use median as default
            "step": max((float(stats.loc[col, "max"]) - float(stats.loc[col, "min"])) / 100, 0.01)
        }
    return features_config

# Metadata for UI
FIELD_INFO = {
    "Pregnancies": {"label": "Grossesses", "unit": "", "desc": "Nombre de fois enceinte"},
    "Glucose": {"label": "Glucose", "unit": "mg/dL", "desc": "Concentration plasmatique (test 2h)"},
    "BloodPressure": {"label": "Tension Artérielle", "unit": "mm Hg", "desc": "Pression artérielle diastolique"},
    "SkinThickness": {"label": "Épaisseur Cutanée", "unit": "mm", "desc": "Épaisseur du pli triceps"},
    "Insulin": {"label": "Insuline", "unit": "mu U/ml", "desc": "Insuline sérique (2h)"},
    "BMI": {"label": "IMC", "unit": "kg/m²", "desc": "Indice de Masse Corporelle"},
    "DiabetesPedigreeFunction": {"label": "Prédisposition Génétique", "unit": "score", "desc": "Fonction pedigree diabète"},
    "Age": {"label": "Âge", "unit": "ans", "desc": "Âge du patient"}
}

# Initialize app state
config = load_config()
model = load_model(Path(config["artifact_dir"]))
features_config = load_stats(Path(config.get("data_path", DEFAULT_DATA_PATH)), config.get("target_column", "Outcome"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    
    if request.method == "POST":
        if not model:
            return render_template("index.html", features=features_config, field_info=FIELD_INFO, error="Modèle introuvable.")
        
        # Collect form data
        input_data = {}
        try:
            for feature in features_config:
                input_data[feature] = float(request.form.get(feature))
            
            input_df = pd.DataFrame([input_data])
            
            # Predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)
                prob_value = probs[0][1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(input_df)
                prob_value = 1 / (1 + np.exp(-scores))[0]
            else:
                prob_value = float(model.predict(input_df)[0])
            
            probability = round(prob_value * 100, 1)
            prediction = "Diabétique" if prob_value >= 0.5 else "Non diabétique"
            
        except Exception as e:
            return render_template("index.html", features=features_config, field_info=FIELD_INFO, error=f"Erreur de prédiction: {str(e)}")

    return render_template("index.html", features=features_config, field_info=FIELD_INFO, prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
