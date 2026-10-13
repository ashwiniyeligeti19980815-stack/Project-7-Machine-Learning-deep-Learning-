import json
import os
import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).with_name("model.joblib")))
ANALYSIS_PATH = Path(__file__).with_name("static") / "analysis_summary.json"


def load_artifact():
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run "
            "'python train_model.py <dataset-path>' before starting the app."
        )
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "vectorizer", "class_names"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise RuntimeError(f"Invalid model artifact: expected keys {', '.join(sorted(required))}.")
    return artifact


artifact = load_artifact()


def load_analysis():
    if not ANALYSIS_PATH.is_file():
        return None
    return json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()


def predict_tweet(tweet):
    cleaned = clean_text(tweet)
    features = artifact["vectorizer"].transform([cleaned])
    prediction = int(artifact["model"].predict(features)[0])
    probabilities = artifact["model"].predict_proba(features)[0]
    confidence = float(probabilities[prediction])
    return {
        "prediction": artifact["class_names"].get(str(prediction), str(prediction)),
        "label": prediction,
        "confidence": confidence,
    }


@app.get("/")
def home():
    return render_template("index.html", tweet="", result=None, error=None, analysis=load_analysis())


@app.post("/")
def classify():
    tweet = request.form.get("tweet", "").strip()
    if not tweet:
        return render_template(
            "index.html", tweet=tweet, result=None, error="Enter a tweet to analyze.", analysis=load_analysis()
        ), 400

    result = predict_tweet(tweet)
    return render_template("index.html", tweet=tweet, result=result, error=None, analysis=load_analysis())


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("tweet"), str) or not payload["tweet"].strip():
        return jsonify(error="Send a non-empty 'tweet' string in the JSON body."), 400
    return jsonify(predict_tweet(payload["tweet"].strip()))


@app.get("/healthz")
def healthcheck():
    return jsonify(status="ok", model_loaded=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)