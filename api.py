import random
import logging
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR  = Path(__file__).parent
CLF_PATH  = BASE_DIR / "burnout_extratree_classifier.pkl"
REG_PATH  = BASE_DIR / "burnout_extratree_regressor.pkl"


def load_models():
    if not CLF_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {CLF_PATH}")
    if not REG_PATH.exists():
        raise FileNotFoundError(f"Regressor not found: {REG_PATH}")

    clf_bundle = joblib.load(CLF_PATH)
    reg_bundle = joblib.load(REG_PATH)

    clf = clf_bundle["model"]
    le  = clf_bundle["label_encoder"]
    reg = reg_bundle["model"]

    log.info("Models loaded successfully")
    return clf, le, reg


try:
    clf_model, label_encoder, reg_model = load_models()
    MODELS_READY = True
except FileNotFoundError as e:
    log.warning(f"{e}")
    clf_model = label_encoder = reg_model = None
    MODELS_READY = False


@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_READY:
        return jsonify({"error": "Models not loaded on server"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required = {"motivation", "happiness", "stress", "caffeine"}
    missing  = required - data.keys()
    if missing:
        return jsonify({"error": f"Missing fields: {sorted(missing)}"}), 400

    try:
        motivation = float(data["motivation"])
        happiness  = float(data["happiness"])
        stress     = float(data["stress"])
        caffeine   = float(data["caffeine"])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"All fields must be numeric: {e}"}), 422

    sleep_hours      = round(random.uniform(1, 12), 2)
    work_hours       = round(random.uniform(1, 7),  2)
    screen_time      = round(random.uniform(1, 10), 2)
    activity_minutes = int(random.uniform(10, 120))
    mood_score       = round(((happiness + motivation) - stress) / 2, 4)

    feature_vector = [[
        sleep_hours, work_hours, screen_time,
        activity_minutes, mood_score, caffeine,
    ]]

    burnout_score   = round(float(reg_model.predict(feature_vector)[0]), 4)
    label_enc       = clf_model.predict(feature_vector)[0]
    predicted_label = label_encoder.inverse_transform([label_enc])[0]

    log.info(f"Prediction: score={burnout_score}, label={predicted_label}")

    return jsonify({
        "burnout_score":   burnout_score,
        "predicted_label": predicted_label,
        "predicted_at":    datetime.now().isoformat(),
    }), 200


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "models_loaded": MODELS_READY,
        "server_time":   datetime.now().isoformat(),
    }), 200


if __name__ == "__main__":
    log.info("Starting Burnout Prediction API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
