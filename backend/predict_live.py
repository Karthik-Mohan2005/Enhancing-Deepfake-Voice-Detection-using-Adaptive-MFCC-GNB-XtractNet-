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
    "models_live"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)


MODEL = joblib.load(
    MODEL_PATH
)


with open(
    CONFIG_PATH,
    "r"
) as f:

    CONFIG = json.load(f)


SAMPLE_RATE = CONFIG[
    "sample_rate"
]

SEGMENT_DURATION = CONFIG[
    "segment_duration"
]

SEGMENT_SAMPLES = int(
    SAMPLE_RATE *
    SEGMENT_DURATION
)

N_MFCC = CONFIG[
    "n_mfcc"
]


def extract_features(
    segment
):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    return np.concatenate(
        [
            mfcc_mean,
            mfcc_std
        ]
    )


def predict(
    audio_path
):

    print("\n========================================")
    print("DEEPFAKE VOICE DETECTOR")
    print("========================================")

    print(
        "File:",
        os.path.basename(
            audio_path
        )
    )


    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )


    if len(audio) == 0:

        raise ValueError(
            "Empty audio."
        )


    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio / max_value
        )


    segments = []


    for start in range(
        0,
        len(audio),
        SEGMENT_SAMPLES
    ):

        end = (
            start +
            SEGMENT_SAMPLES
        )

        segment = audio[
            start:end
        ]


        if len(segment) < SEGMENT_SAMPLES:

            continue


        segments.append(
            segment
        )


    print(
        "Segments:",
        len(segments)
    )


    X = np.asarray(
        [
            extract_features(
                segment
            )

            for segment in segments
        ]
    )


    print(
        "Feature matrix:",
        X.shape
    )


    predictions = MODEL.predict(
        X
    )


    probabilities = (
        MODEL.predict_proba(
            X
        )
    )


    real_probability = float(
        np.mean(
            probabilities[:, 0]
        )
    )


    fake_probability = float(
        np.mean(
            probabilities[:, 1]
        )
    )


    real_segments = int(
        np.sum(
            predictions == 0
        )
    )


    fake_segments = int(
        np.sum(
            predictions == 1
        )
    )


    if fake_probability > real_probability:

        result = "FAKE"

        confidence = fake_probability

    else:

        result = "REAL"

        confidence = real_probability


    print("\n========================================")
    print("FINAL RESULT")
    print("========================================")

    print(
        "\nPrediction:",
        result
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"\nREAL probability: "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"FAKE probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"\nREAL segments: "
        f"{real_segments}"
    )

    print(
        f"FAKE segments: "
        f"{fake_segments}"
    )

    print(
        "\n========================================"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            'Usage: python predict_live.py "audio.wav"'
        )

        sys.exit(1)


    predict(
        sys.argv[1]
    )