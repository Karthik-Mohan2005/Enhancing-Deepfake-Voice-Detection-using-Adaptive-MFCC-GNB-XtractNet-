import os
import sys
import json
import joblib
import librosa
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_final"
)


# ------------------------------------------------------------
# LOAD CONFIG
# ------------------------------------------------------------

with open(
    os.path.join(
        MODEL_DIR,
        "feature_config.json"
    ),
    "r"
) as f:

    config = json.load(f)


MODEL = joblib.load(
    os.path.join(
        MODEL_DIR,
        "best_model.joblib"
    )
)


SAMPLE_RATE = config["sample_rate"]
N_MFCC = config["n_mfcc"]
N_MELS = config["n_mels"]
N_FFT = config["n_fft"]


# ------------------------------------------------------------
# FEATURE EXTRACTION
# MUST BE IDENTICAL TO TRAINING
# ------------------------------------------------------------

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(
            "Audio file is empty."
        )

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:
        audio = audio / max_value

    # MFCC

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_features = np.mean(
        mfcc.T,
        axis=0
    )

    # Mel

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_features = np.mean(
        mel_db.T,
        axis=0
    )

    # STFT

    stft = librosa.stft(
        audio,
        n_fft=N_FFT
    )

    magnitude = np.abs(stft)

    spec_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    spec_features = np.mean(
        spec_db.T,
        axis=0
    )

    features = np.concatenate(
        [
            mfcc_features,
            mel_features,
            spec_features
        ]
    )

    return features.reshape(
        1,
        -1
    )


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

def predict_audio(file_path):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    features = extract_features(
        file_path
    )

    prediction = MODEL.predict(
        features
    )[0]

    probabilities = MODEL.predict_proba(
        features
    )[0]

    real_probability = probabilities[0]
    fake_probability = probabilities[1]

    if prediction == 0:

        label = "REAL"
        confidence = real_probability

    else:

        label = "FAKE"
        confidence = fake_probability

    print("\n===================================")
    print("DEEPFAKE VOICE DETECTION")
    print("===================================")

    print(
        f"File       : {os.path.basename(file_path)}"
    )

    print(
        f"Prediction : {label}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    print(
        f"Real       : {real_probability * 100:.2f}%"
    )

    print(
        f"Fake       : {fake_probability * 100:.2f}%"
    )

    print("===================================\n")

    return {
        "prediction": label,
        "confidence": float(confidence),
        "real_probability": float(real_probability),
        "fake_probability": float(fake_probability)
    }


# ------------------------------------------------------------
# COMMAND LINE
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python predict.py path_to_audio.wav"
        )

        sys.exit(1)

    predict_audio(
        sys.argv[1]
    )