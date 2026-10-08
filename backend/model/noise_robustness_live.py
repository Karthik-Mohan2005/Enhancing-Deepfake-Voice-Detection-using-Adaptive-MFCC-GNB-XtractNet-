import os
import json
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models_live",
    "best_model.joblib"
)

REAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "REAL"
)

FAKE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "FAKE"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results_noise_live"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SAMPLE_RATE = 16000

SEGMENT_DURATION = 1.0

SEGMENT_SAMPLES = int(
    SAMPLE_RATE * SEGMENT_DURATION
)

N_MFCC = 40

NOISE_LEVELS = [
    "clean",
    "20dB",
    "10dB",
    "5dB",
    "0dB"
]

RANDOM_STATE = 42


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("LIVE MODEL NOISE ROBUSTNESS EXPERIMENT")
print("=" * 60)

print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")
print(
    "Model:",
    type(model).__name__
)

print(
    "Model classes:",
    model.classes_
)


# ============================================================
# MODEL CLASS MAPPING
# ============================================================
#
# The live training script uses:
#
# REAL = 0
# FAKE = 1
#
# Therefore:
#
# predict_proba() column 0 = REAL
# predict_proba() column 1 = FAKE
#
# ============================================================

if len(model.classes_) != 2:

    raise ValueError(
        "Expected a binary classification model."
    )

REAL_INDEX = 0
FAKE_INDEX = 1

print(
    "\nDetected model classes:",
    model.classes_
)

print(
    "REAL class index:",
    REAL_INDEX
)

print(
    "FAKE class index:",
    FAKE_INDEX
)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(
        np.float32
    )

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio / max_value
        )

    return audio


# ============================================================
# SEGMENT AUDIO
# ============================================================

def segment_audio(audio):

    total_samples = len(audio)

    number_of_segments = (
        total_samples //
        SEGMENT_SAMPLES
    )

    if number_of_segments == 0:

        return np.empty(
            (
                0,
                SEGMENT_SAMPLES
            ),
            dtype=np.float32
        )

    usable_length = (
        number_of_segments *
        SEGMENT_SAMPLES
    )

    audio = audio[
        :usable_length
    ]

    segments = audio.reshape(
        number_of_segments,
        SEGMENT_SAMPLES
    )

    return segments


# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

def extract_features_from_segments(
    segments
):

    feature_list = []

    for segment in segments:

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC
        )

        # 40 MFCC mean values
        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        # 40 MFCC standard deviation values
        mfcc_std = np.std(
            mfcc,
            axis=1
        )

        # Total = 80 features
        features = np.concatenate(
            [
                mfcc_mean,
                mfcc_std
            ]
        )

        feature_list.append(
            features
        )

    return np.asarray(
        feature_list,
        dtype=np.float32
    )


# ============================================================
# ADD WHITE GAUSSIAN NOISE
# ============================================================

def add_white_noise(
    audio,
    snr_db
):

    signal_power = np.mean(
        audio ** 2
    )

    if signal_power <= 0:

        return audio.copy()

    noise_power = (
        signal_power /
        (10 ** (snr_db / 10))
    )

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=audio.shape
    ).astype(
        np.float32
    )

    noisy_audio = (
        audio + noise
    )

    # Prevent clipping
    max_value = np.max(
        np.abs(noisy_audio)
    )

    if max_value > 1:

        noisy_audio = (
            noisy_audio /
            max_value
        )

    return noisy_audio.astype(
        np.float32
    )


# ============================================================
# APPLY NOISE LEVEL
# ============================================================

def apply_noise(
    audio,
    noise_level
):

    if noise_level == "clean":

        return audio.copy()

    elif noise_level == "20dB":

        return add_white_noise(
            audio,
            20
        )

    elif noise_level == "10dB":

        return add_white_noise(
            audio,
            10
        )

    elif noise_level == "5dB":

        return add_white_noise(
            audio,
            5
        )

    elif noise_level == "0dB":

        return add_white_noise(
            audio,
            0
        )

    else:

        raise ValueError(
            f"Unknown noise level: "
            f"{noise_level}"
        )


# ============================================================
# PREDICT ONE AUDIO FILE
# ============================================================

def predict_file(
    file_path,
    noise_level
):

    # ----------------------------------------
    # Load audio
    # ----------------------------------------

    audio = load_audio(
        file_path
    )

    # ----------------------------------------
    # Apply noise
    # ----------------------------------------

    noisy_audio = apply_noise(
        audio,
        noise_level
    )

    # ----------------------------------------
    # Split into 1-second segments
    # ----------------------------------------

    segments = segment_audio(
        noisy_audio
    )

    if len(segments) == 0:

        return None

    # ----------------------------------------
    # Extract 80 features
    # ----------------------------------------

    features = (
        extract_features_from_segments(
            segments
        )
    )

    # ----------------------------------------
    # Predict
    # ----------------------------------------

    predictions = model.predict(
        features
    )

    probabilities = (
        model.predict_proba(
            features
        )
    )

    # ----------------------------------------
    # Average probabilities across segments
    # ----------------------------------------

    real_probability = float(
        np.mean(
            probabilities[
                :,
                REAL_INDEX
            ]
        )
    )

    fake_probability = float(
        np.mean(
            probabilities[
                :,
                FAKE_INDEX
            ]
        )
    )

    # ----------------------------------------
    # Final file prediction
    # ----------------------------------------

    if (
        real_probability >=
        fake_probability
    ):

        final_prediction = "REAL"

    else:

        final_prediction = "FAKE"

    # ----------------------------------------
    # Segment predictions
    # ----------------------------------------

    # Numeric labels:
    # 0 = REAL
    # 1 = FAKE

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

    return {

        "prediction":
            final_prediction,

        "real_probability":
            real_probability,

        "fake_probability":
            fake_probability,

        "real_segments":
            real_segments,

        "fake_segments":
            fake_segments,

        "total_segments":
            len(segments)
    }


# ============================================================
# COLLECT DATASET FILES
# ============================================================

files = []

# ----------------------------
# REAL files
# ----------------------------

if not os.path.exists(
    REAL_DIR
):

    raise FileNotFoundError(
        f"REAL directory not found:\n"
        f"{REAL_DIR}"
    )

for filename in sorted(
    os.listdir(REAL_DIR)
):

    if filename.lower().endswith(
        ".wav"
    ):

        files.append({

            "path":
                os.path.join(
                    REAL_DIR,
                    filename
                ),

            "filename":
                filename,

            "true_label":
                "REAL"
        })


# ----------------------------
# FAKE files
# ----------------------------

if not os.path.exists(
    FAKE_DIR
):

    raise FileNotFoundError(
        f"FAKE directory not found:\n"
        f"{FAKE_DIR}"
    )

for filename in sorted(
    os.listdir(FAKE_DIR)
):

    if filename.lower().endswith(
        ".wav"
    ):

        files.append({

            "path":
                os.path.join(
                    FAKE_DIR,
                    filename
                ),

            "filename":
                filename,

            "true_label":
                "FAKE"
        })


# ============================================================
# DATASET SUMMARY
# ============================================================

real_file_count = sum(
    x["true_label"] == "REAL"
    for x in files
)

fake_file_count = sum(
    x["true_label"] == "FAKE"
    for x in files
)

print("\nDataset:")
print(
    "REAL files:",
    real_file_count
)

print(
    "FAKE files:",
    fake_file_count
)

print(
    "Total files:",
    len(files)
)


# ============================================================
# RUN NOISE EXPERIMENT
# ============================================================

all_results = []


for noise_level in NOISE_LEVELS:

    print("\n")

    print("=" * 60)

    print(
        f"TESTING NOISE LEVEL: "
        f"{noise_level}"
    )

    print("=" * 60)

    # Reproducible noise
    np.random.seed(
        RANDOM_STATE
    )

    y_true = []

    y_pred = []

    y_probability = []

    file_results = []


    # ========================================================
    # PROCESS EVERY FILE
    # ========================================================

    for index, item in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index:02d}/{len(files)}] "
            f"{item['filename']}"
        )

        try:

            result = predict_file(
                item["path"],
                noise_level
            )

            if result is None:

                print(
                    "   Skipped: "
                    "no complete segments"
                )

                continue


            # --------------------------------------------
            # True label
            # --------------------------------------------

            true_label = (
                item["true_label"]
            )


            # --------------------------------------------
            # Numeric true label
            # --------------------------------------------

            true_numeric = (
                1
                if true_label == "FAKE"
                else 0
            )


            # --------------------------------------------
            # Numeric predicted label
            # --------------------------------------------

            predicted_numeric = (
                1
                if result["prediction"]
                == "FAKE"
                else 0
            )


            # --------------------------------------------
            # Store metrics data
            # --------------------------------------------

            y_true.append(
                true_numeric
            )

            y_pred.append(
                predicted_numeric
            )

            y_probability.append(
                result[
                    "fake_probability"
                ]
            )


            # --------------------------------------------
            # Store file result
            # --------------------------------------------

            file_results.append({

                "filename":
                    item["filename"],

                "true_label":
                    true_label,

                "prediction":
                    result[
                        "prediction"
                    ],

                "real_probability":
                    result[
                        "real_probability"
                    ],

                "fake_probability":
                    result[
                        "fake_probability"
                    ],

                "real_segments":
                    result[
                        "real_segments"
                    ],

                "fake_segments":
                    result[
                        "fake_segments"
                    ],

                "total_segments":
                    result[
                        "total_segments"
                    ]
            })


        except Exception as e:

            print(
                "   ERROR:",
                str(e)
            )


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if len(y_true) == 0:

        print(
            "\nNo valid predictions "
            "were produced."
        )

        continue


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    # --------------------------------------------
    # AUC
    # --------------------------------------------

    try:

        auc = roc_auc_score(
            y_true,
            y_probability
        )

    except Exception:

        auc = 0.0


    # --------------------------------------------
    # Confusion matrix
    # --------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\nRESULT")

    print("-" * 40)

    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1        : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"AUC       : "
        f"{auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        "              Predicted"
    )

    print(
        "              REAL  FAKE"
    )

    print(
        f"Actual REAL   "
        f"{cm[0, 0]:4d}  "
        f"{cm[0, 1]:4d}"
    )

    print(
        f"Actual FAKE   "
        f"{cm[1, 0]:4d}  "
        f"{cm[1, 1]:4d}"
    )


    # ========================================================
    # SAVE FILE-LEVEL RESULTS
    # ========================================================

    file_df = pd.DataFrame(
        file_results
    )

    file_csv_path = os.path.join(
        OUTPUT_DIR,
        f"file_results_{noise_level}.csv"
    )

    file_df.to_csv(
        file_csv_path,
        index=False
    )


    # ========================================================
    # SAVE CONFUSION MATRIX IMAGE
    # ========================================================

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm
    )

    plt.title(
        f"Confusion Matrix - "
        f"{noise_level}"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

    plt.yticks(
        [0, 1],
        ["REAL", "FAKE"]
    )


    # Add numbers to matrix

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )


    plt.tight_layout()

    confusion_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_{noise_level}.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()


    # ========================================================
    # STORE SUMMARY
    # ========================================================

    all_results.append({

        "Noise Level":
            noise_level,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "AUC":
            auc
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")

print("=" * 60)

print(
    "FINAL NOISE ROBUSTNESS COMPARISON"
)

print("=" * 60)


if len(results_df) > 0:

    display_df = (
        results_df.copy()
    )

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"


    display_df["AUC"] = (
        display_df["AUC"]
        .round(4)
    )


    print(
        display_df.to_string(
            index=False
        )
    )

else:

    print(
        "No experiment results."
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

results_csv_path = os.path.join(
    OUTPUT_DIR,
    "noise_robustness_results.csv"
)

results_df.to_csv(
    results_csv_path,
    index=False
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

results_json_path = os.path.join(
    OUTPUT_DIR,
    "noise_robustness_results.json"
)

with open(
    results_json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=4
    )


# ============================================================
# ACCURACY GRAPH
# ============================================================

if len(results_df) > 0:

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results_df["Noise Level"],
        results_df["Accuracy"] * 100,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Noise Level"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Deepfake Voice Detection "
        "Under Different Noise Levels"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "noise_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# PRECISION / RECALL / F1 GRAPH
# ============================================================

if len(results_df) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results_df["Noise Level"],
        results_df["Precision"] * 100,
        marker="o",
        label="Precision"
    )

    plt.plot(
        results_df["Noise Level"],
        results_df["Recall"] * 100,
        marker="o",
        label="Recall"
    )

    plt.plot(
        results_df["Noise Level"],
        results_df["F1"] * 100,
        marker="o",
        label="F1"
    )

    plt.xlabel(
        "Noise Level"
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "Detection Metrics Under Noise"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "noise_metrics.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")

print("=" * 60)

print(
    "NOISE ROBUSTNESS EXPERIMENT COMPLETE"
)

print("=" * 60)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated summary files:"
)

print(
    " - noise_robustness_results.csv"
)

print(
    " - noise_robustness_results.json"
)

print(
    " - noise_accuracy.png"
)

print(
    " - noise_metrics.png"
)

print(
    "\nIndividual file results:"
)

print(
    " - file_results_clean.csv"
)

print(
    " - file_results_20dB.csv"
)

print(
    " - file_results_10dB.csv"
)

print(
    " - file_results_5dB.csv"
)

print(
    " - file_results_0dB.csv"
)

print(
    "\nConfusion matrices:"
)

print(
    " - confusion_clean.png"
)

print(
    " - confusion_20dB.png"
)

print(
    " - confusion_10dB.png"
)

print(
    " - confusion_5dB.png"
)

print(
    " - confusion_0dB.png"
)