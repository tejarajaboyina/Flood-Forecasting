from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH = "./model/global_flood_model.h5"
SCALER_PATH = "./model/scalers_joblib.pkl"
CSV_PATH = "india_28_states_hydrology_2010_2020.csv"

app = Flask(__name__)

global_model = load_model(MODEL_PATH)
scalers = joblib.load(SCALER_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=['date'])

TIME_STEPS = 30

# Pick features automatically
exclude = ["state", "date"]
numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
features = [c for c in numeric_cols if c not in exclude]
target_candidates = ["streamflow_m3s", "water_level_m", "streamflow", "water_level"]
target = None
for t in target_candidates:
    if t in df.columns:
        target = t
        break


def prepare_last_sequence(state):
    subset = df[df["state"] == state].sort_values("date")
    req_cols = features + [target]
    subset = subset[req_cols].dropna()
    if len(subset) < TIME_STEPS:
        return None, "Not enough rows"

    scaler = scalers.get(state)
    if scaler is None:
        return None, "Scaler missing"

    scaled = scaler.transform(subset.values)
    seq = scaled[-TIME_STEPS:]
    return seq, None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    state = data["state"]
    days = int(data["days"])

    seq, err = prepare_last_sequence(state)
    if err:
        return jsonify({"error": err}), 400

    n_features = len(features)
    seq_f = seq[:, :n_features]

    preds = []
    scaler = scalers[state]

    for i in range(days):
        x = seq_f.reshape(1, TIME_STEPS, n_features)
        pred_scaled = global_model.predict(x, verbose=0)[0][0]

        last_full = seq[-1].copy()
        last_full[-1] = pred_scaled

        pred_real = scaler.inverse_transform([last_full])[0][-1]

        risk = float((pred_real / 5.0) * 100)
        risk = max(0, min(risk, 100))

        preds.append(pred_real)

        seq_f = np.vstack([seq_f[1:], seq_f[-1]])

    return jsonify({
        "predictions": preds,
        "risk": risk
    })


if __name__ == "__main__":
    app.run(debug=True)
