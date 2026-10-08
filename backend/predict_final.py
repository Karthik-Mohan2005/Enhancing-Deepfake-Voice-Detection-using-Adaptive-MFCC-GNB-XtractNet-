import os
import sys
import json
import joblib
import librosa
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_final"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "feature_config.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

SEGMENT_DURATION = 1.0

SEGMENT_SAMPLES = int(
    SAMPLE_RATE * SEGMENT_DURATION
)

N_MFCC = 20


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Trained model not found:\n"
        + MODEL_PATH
    )


if not os.path.exists(CONFIG_PATH):

    raise FileNotFoundError(
        "Feature configuration not found:\n"
        + CONFIG_PATH
    )


MODEL = joblib.load(
    MODEL_PATH
)


with open(
    CONFIG_PATH,
    "r"
) as f:

    CONFIG = json.load(f)


FEATURE_COLUMNS = CONFIG[
    "feature_columns"
]


# ============================================================
# FEATURE EXTRACTION FOR ONE SEGMENT
# ============================================================

def extract_features_from_segment(
    audio,
    sr
):

    # --------------------------------------------------------
    # 1. Chroma STFT
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    chroma_stft = np.mean(
        chroma
    )

    # --------------------------------------------------------
    # 2. RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    rms_value = np.mean(
        rms
    )

    # --------------------------------------------------------
    # 3. Spectral Centroid
    # --------------------------------------------------------

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    spectral_centroid_value = np.mean(
        spectral_centroid
    )

    # --------------------------------------------------------
    # 4. Spectral Bandwidth
    # --------------------------------------------------------

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    spectral_bandwidth_value = np.mean(
        spectral_bandwidth
    )

    # --------------------------------------------------------
    # 5. Spectral Rolloff
    # --------------------------------------------------------

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )
    )

    rolloff_value = np.mean(
        rolloff
    )

    # --------------------------------------------------------
    # 6. Zero Crossing Rate
    # --------------------------------------------------------

    zero_crossing = (
        librosa.feature.zero_crossing_rate(
            y=audio
        )
    )

    zero_crossing_value = np.mean(
        zero_crossing
    )

    # --------------------------------------------------------
    # 7. MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_values = np.mean(
        mfcc,
        axis=1
    )

    # --------------------------------------------------------
    # Build dictionary
    # --------------------------------------------------------

    feature_dict = {

        "chroma_stft":
            chroma_stft,

        "rms":
            rms_value,

        "spectral_centroid":
            spectral_centroid_value,

        "spectral_bandwidth":
            spectral_bandwidth_value,

        "rolloff":
            rolloff_value,

        "zero_crossing_rate":
            zero_crossing_value
    }


    for i in range(N_MFCC):

        feature_dict[
            f"mfcc{i + 1}"
        ] = mfcc_values[i]


    # --------------------------------------------------------
    # Exact training feature order
    # --------------------------------------------------------

    features = [

        feature_dict[column]

        for column in FEATURE_COLUMNS

    ]


    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# LOAD AND SEGMENT AUDIO
# ============================================================

def load_and_segment_audio(
    audio_path
):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )


    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )


    duration = (
        len(audio) / sr
    )


    print(
        f"Sample rate : {sr}"
    )

    print(
        f"Duration    : {duration:.2f} seconds"
    )


    # --------------------------------------------------------
    # Segment audio into 1-second chunks
    # --------------------------------------------------------

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


        # Ignore incomplete final segment

        if len(segment) < SEGMENT_SAMPLES:

            continue


        segments.append(
            segment
        )


    return segments


# ============================================================
# PREDICT AUDIO
# ============================================================

def predict_audio(
    audio_path
):

    print("\n========================================")
    print("DEEPFAKE VOICE DETECTION")
    print("========================================")

    print(
        "File:",
        os.path.basename(
            audio_path
        )
    )


    # --------------------------------------------------------
    # Segment audio
    # --------------------------------------------------------

    segments = load_and_segment_audio(
        audio_path
    )


    print(
        "Segments    :",
        len(segments)
    )


    if len(segments) == 0:

        raise ValueError(
            "No complete 1-second segments found."
        )


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    print(
        "\nExtracting features..."
    )


    all_features = []


    for index, segment in enumerate(
        segments
    ):

        features = (
            extract_features_from_segment(
                segment,
                SAMPLE_RATE
            )
        )

        all_features.append(
            features
        )


    X = np.asarray(
        all_features,
        dtype=np.float64
    )


    print(
        "Feature matrix:",
        X.shape
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Preserve feature names used during training
    # --------------------------------------------------------

    X_df = pd.DataFrame(
        X,
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # Segment predictions
    # --------------------------------------------------------

    predictions = MODEL.predict(
        X_df
    )


    probabilities = MODEL.predict_proba(
        X_df
    )


    fake_probabilities = (
        probabilities[:, 1]
    )


    real_probabilities = (
        probabilities[:, 0]
    )


    # --------------------------------------------------------
    # Aggregate segment probabilities
    # --------------------------------------------------------

    mean_real_probability = float(
        np.mean(
            real_probabilities
        )
    )


    mean_fake_probability = float(
        np.mean(
            fake_probabilities
        )
    )


    # --------------------------------------------------------
    # Majority vote
    # --------------------------------------------------------

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


    total_segments = len(
        predictions
    )


    real_percentage = (
        real_segments /
        total_segments *
        100
    )


    fake_percentage = (
        fake_segments /
        total_segments *
        100
    )


    # --------------------------------------------------------
    # Final decision
    #
    # Probability aggregation is used as primary decision.
    # --------------------------------------------------------

    if (
        mean_fake_probability
        >=
        mean_real_probability
    ):

        final_label = "FAKE"

        confidence = (
            mean_fake_probability
        )

    else:

        final_label = "REAL"

        confidence = (
            mean_real_probability
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n========================================")
    print("          FINAL RESULT")
    print("========================================")


    print(
        "\nPrediction:",
        final_label
    )


    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


    print("\nAverage Probability:")


    print(
        f"REAL : "
        f"{mean_real_probability * 100:.2f}%"
    )


    print(
        f"FAKE : "
        f"{mean_fake_probability * 100:.2f}%"
    )


    print("\nSegment Voting:")


    print(
        f"REAL segments : "
        f"{real_segments} "
        f"({real_percentage:.2f}%)"
    )


    print(
        f"FAKE segments : "
        f"{fake_segments} "
        f"({fake_percentage:.2f}%)"
    )


    print(
        "\nTotal segments:",
        total_segments
    )


    print(
        "\n========================================"
    )


    return {

        "file":
            os.path.basename(
                audio_path
            ),

        "prediction":
            final_label,

        "confidence":
            confidence,

        "real_probability":
            mean_real_probability,

        "fake_probability":
            mean_fake_probability,

        "total_segments":
            total_segments,

        "real_segments":
            real_segments,

        "fake_segments":
            fake_segments
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python predict_final.py '
            '"path_to_audio.wav"'
        )

        sys.exit(1)


    audio_path = sys.argv[1]


    if not os.path.exists(
        audio_path
    ):

        print(
            "\nERROR: File not found:"
        )

        print(
            audio_path
        )

        sys.exit(1)


    try:

        predict_audio(
            audio_path
        )

    except Exception as e:

        print(
            "\nPrediction failed:"
        )

        print(
            str(e)
        )

        sys.exit(1)