import random
import logging
import threading
import joblib
import numpy as np
from datetime import datetime, time as dtime
from pathlib import Path
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR  # put .pkl files next to app.py
CLF_PATH   = MODEL_DIR / "burnout_extratree_classifier.pkl"
REG_PATH   = MODEL_DIR / "burnout_extratree_regressor.pkl"

state = {
    "user_input": None,         
    "user_input_at": None,       

    "prediction": None,          
    "predicted_at": None,        

    "dashboard_ready": False,
}
state_lock = threading.Lock()

def load_models():
    if not CLF_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {CLF_PATH}")
    if not REG_PATH.exists():
        raise FileNotFoundError(f"Regressor not found:  {REG_PATH}")

    clf_bundle = joblib.load(CLF_PATH)
    reg_bundle = joblib.load(REG_PATH)

    clf = clf_bundle["model"]
    le  = clf_bundle["label_encoder"]
    reg = reg_bundle["model"]

    log.info("✔  Models loaded successfully")
    return clf, le, reg

try:
    clf_model, label_encoder, reg_model = load_models()
    MODELS_READY = True
except FileNotFoundError as e:
    log.warning(f"⚠  {e}  —  /api/trigger will fail until models are present")
    clf_model = label_encoder = reg_model = None
    MODELS_READY = False

FEATURES = ["sleep_hours", "work_hours", "screen_time",
            "activity_minutes", "mood_score", "caffeine"]


def run_prediction():
    """
    1. Read latest user_input from state.
    2. Generate random vitals.
    3. Compute mood_score.
    4. Run classifier + regressor.
    5. Store full result in state["prediction"].
    """
    if not MODELS_READY:
        log.error("Models not loaded — skipping prediction job")
        return

    with state_lock:
        user_input = state["user_input"]

    if user_input is None:
        log.warning("No user input available — skipping prediction job")
        return

    # --- Random vitals (generated at job time) ---
    sleep_hours      = round(random.uniform(1, 12), 2)
    work_hours       = round(random.uniform(1, 7),  2)
    screen_time      = round(random.uniform(1, 10), 2)
    activity_minutes = int(random.uniform(10, 120))

    # --- User-supplied values ---
    motivation = float(user_input["motivation"])
    happiness  = float(user_input["happiness"])
    stress     = float(user_input["stress"])
    caffeine   = float(user_input["caffeine"])

    # --- Derived feature ---
    mood_score = round(((happiness + motivation) - stress) / 2, 4)

    feature_vector = [[
        sleep_hours, work_hours, screen_time,
        activity_minutes, mood_score, caffeine,
    ]]

    # --- Inference ---
    burnout_score = round(float(reg_model.predict(feature_vector)[0]), 4)
    label_enc     = clf_model.predict(feature_vector)[0]
    predicted_label = label_encoder.inverse_transform([label_enc])[0]

    result = {
        # Input features
        "features": {
            "sleep_hours":      sleep_hours,
            "work_hours":       work_hours,
            "screen_time":      screen_time,
            "activity_minutes": activity_minutes,
            "mood_score":       mood_score,
            "caffeine":         caffeine,
        },
        # User-provided inputs (for transparency)
        "user_input": {
            "motivation": motivation,
            "happiness":  happiness,
            "stress":     stress,
            "caffeine":   caffeine,
        },
        # Model outputs
        "burnout_score":   burnout_score,
        "predicted_label": predicted_label,
        # Metadata
        "predicted_at": datetime.now().isoformat(),
    }

    with state_lock:
        state["prediction"]   = result
        state["predicted_at"] = datetime.now()
        state["dashboard_ready"] = False   # reset until 20:30

    log.info(f"✔  Prediction complete — score={burnout_score}, label={predicted_label}")


def release_to_dashboard():
    """Called at 20:30 — marks results ready for dashboard.html."""
    with state_lock:
        if state["prediction"] is None:
            log.warning("No prediction available at 20:30 — dashboard not updated")
            return
        state["dashboard_ready"] = True
    log.info("✔  Results released to dashboard")


# ── Scheduler (pure stdlib — no extra packages) ───────────────────────────────
JOBS = [
    (dtime(20, 0),  run_prediction),
    (dtime(20, 30), release_to_dashboard),
]

def _scheduler_loop():
    """Runs in a background daemon thread. Fires jobs at their scheduled time."""
    fired_today: set[dtime] = set()
    log.info("⏰  Scheduler started  (jobs: 20:00 predict  |  20:30 release)")

    while True:
        now = datetime.now()
        today_time = dtime(now.hour, now.minute)

        # Reset fired set at midnight
        if today_time == dtime(0, 0):
            fired_today.clear()

        for job_time, job_fn in JOBS:
            if today_time >= job_time and job_time not in fired_today:
                log.info(f"⏰  Firing job: {job_fn.__name__} at {now.strftime('%H:%M')}")
                threading.Thread(target=job_fn, daemon=True).start()
                fired_today.add(job_time)

        # Sleep until the next whole minute
        seconds_left = 60 - now.second
        threading.Event().wait(seconds_left)


scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
scheduler_thread.start()


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/user-input", methods=["POST"])
def receive_user_input():
    """
    Called by user_input.html.

    Expected JSON body:
    {
        "motivation": <float  0–10>,
        "happiness":  <float  0–10>,
        "stress":     <float  0–10>,
        "caffeine":   <float  0–6>
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required = {"motivation", "happiness", "stress", "caffeine"}
    missing  = required - data.keys()
    if missing:
        return jsonify({"error": f"Missing fields: {sorted(missing)}"}), 400

    try:
        validated = {k: float(data[k]) for k in required}
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"All fields must be numeric — {e}"}), 422

    # Compute mood_score so the client can preview it
    mood_score = round(
        ((validated["happiness"] + validated["motivation"]) - validated["stress"]) / 2, 4
    )

    with state_lock:
        state["user_input"]    = validated
        state["user_input_at"] = datetime.now().isoformat()

    log.info(f"📥  User input received — mood_score preview: {mood_score}")

    return jsonify({
        "status":     "received",
        "mood_score": mood_score,
        "message":    "Input saved. Prediction will run at 20:00 and results released at 20:30.",
    }), 200


@app.route("/api/results", methods=["GET"])
def get_results():
    """
    Polled by dashboard.html.
    Returns 200 + results when dashboard_ready is True,
    otherwise 202 Accepted with a 'pending' status.
    """
    with state_lock:
        ready      = state["dashboard_ready"]
        prediction = state["prediction"]

    if not ready or prediction is None:
        return jsonify({
            "status":  "pending",
            "message": "Results not yet available. Check back after 20:30.",
        }), 202

    return jsonify({
        "status": "ready",
        **prediction,
    }), 200


@app.route("/api/status", methods=["GET"])
def get_status():
    """Health-check — shows scheduler state, last input time, last prediction time."""
    with state_lock:
        return jsonify({
            "models_loaded":    MODELS_READY,
            "user_input_at":    state["user_input_at"],
            "has_user_input":   state["user_input"] is not None,
            "predicted_at":     state["predicted_at"].isoformat() if state["predicted_at"] else None,
            "dashboard_ready":  state["dashboard_ready"],
            "schedule": {
                "predict":  "20:00 daily",
                "release":  "20:30 daily",
            },
            "server_time": datetime.now().isoformat(),
        }), 200


@app.route("/api/trigger", methods=["POST"])
def manual_trigger():
    """
    DEV ONLY — immediately runs both jobs so you can test without waiting.
    Remove or auth-guard this endpoint before going to production.
    """
    log.info("🔧  Manual trigger received")
    run_prediction()
    release_to_dashboard()

    with state_lock:
        prediction = state["prediction"]

    if prediction is None:
        return jsonify({"error": "Prediction failed — check logs"}), 500

    return jsonify({
        "status": "triggered",
        **prediction,
    }), 200


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀  Starting Burnout Prediction API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)