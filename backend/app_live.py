import os
import uuid
import numpy as np
import joblib
import librosa
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model.mfcc_features import SAMPLE_RATE, SEGMENT_SAMPLES, extract_mfcc
app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models_live",
    "best_model.joblib"
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")
print("Model classes:", model.classes_)
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )
def load_audio(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )
    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value
    return audio
def segment_audio(audio):
    segments = []
    for start in range(0, len(audio), SEGMENT_SAMPLES):
        end = start + SEGMENT_SAMPLES
        segment = audio[start:end]
        if len(segment) < SEGMENT_SAMPLES:
            continue
        segments.append(segment)
    return segments
def predict_audio(file_path):
    audio = load_audio(file_path)
    duration = len(audio) / SAMPLE_RATE
    segments = segment_audio(audio)
    if not segments:
        raise ValueError(
            "Audio is too short. Please upload at least 1 second of audio."
        )
    features = np.array([
        extract_mfcc(segment)
        for segment in segments
    ])
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    real_index = classes.index(0)
    fake_index = classes.index(1)
    real_probability = float(
        np.mean(probabilities[:, real_index])
    )
    fake_probability = float(
        np.mean(probabilities[:, fake_index])
    )
    if fake_probability >= real_probability:
        prediction = "FAKE"
        confidence = fake_probability
    else:
        prediction = "REAL"
        confidence = real_probability
    segment_predictions = model.predict(features)
    real_segments = int(
        np.sum(segment_predictions == 0)
    )
    fake_segments = int(
        np.sum(segment_predictions == 1)
    )
    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),

        "real_probability": round(
            real_probability * 100, 2
        ),
        "fake_probability": round(
            fake_probability * 100, 2
        ),
        "duration": round(duration, 2),
        "segments_analyzed": len(segments),
        "real_segments": real_segments,
        "fake_segments": fake_segments
    }
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/api/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file uploaded."
        }), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an audio file."
        }), 400
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported audio format."
        }), 400
    filename = secure_filename(file.filename)
    unique_name = (
        str(uuid.uuid4())
        + "_"
        + filename
    )
    file_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )
    try:
        file.save(file_path)
        result = predict_audio(file_path)
        result["filename"] = filename
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("      DEEPFAKE VOICE DETECTION SYSTEM")
    print("=" * 60)
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )