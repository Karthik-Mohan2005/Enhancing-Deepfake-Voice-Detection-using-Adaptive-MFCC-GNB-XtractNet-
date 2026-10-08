import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

REAL_DIR = os.path.join(
    DATASET_DIR,
    "REAL"
)

FAKE_DIR = os.path.join(
    DATASET_DIR,
    "FAKE"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_live"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results_live"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# AUDIO CONFIG
# ============================================================

SAMPLE_RATE = 16000

SEGMENT_DURATION = 1.0

SEGMENT_SAMPLES = int(
    SAMPLE_RATE * SEGMENT_DURATION
)

N_MFCC = 40

RANDOM_STATE = 42


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path):

    import librosa

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:

        raise ValueError(
            "Empty audio"
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
# SEGMENTATION
# ============================================================

def segment_audio(audio):

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

    return segments


# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

def extract_mfcc_features(
    segment
):

    import librosa

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

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std
        ]
    )

    return features


# ============================================================
# PROCESS DIRECTORY
# ============================================================

def process_directory(
    directory,
    label
):

    X = []

    y = []

    file_ids = []

    filenames = sorted(
        os.listdir(directory)
    )

    for filename in filenames:

        if not filename.lower().endswith(
            ".wav"
        ):

            continue

        path = os.path.join(
            directory,
            filename
        )

        try:

            print(
                f"Processing: {filename}"
            )

            audio = load_audio(
                path
            )

            segments = segment_audio(
                audio
            )

            print(
                f"  Segments: {len(segments)}"
            )

            for segment_index, segment in enumerate(
                segments
            ):

                features = (
                    extract_mfcc_features(
                        segment
                    )
                )

                X.append(
                    features
                )

                y.append(
                    label
                )

                file_ids.append(
                    filename
                )

        except Exception as e:

            print(
                f"ERROR: {filename}: {e}"
            )

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(file_ids)
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("FINAL LIVE AUDIO MODEL TRAINING")
print("========================================")


print("\nProcessing REAL audio...")

X_real, y_real, files_real = (
    process_directory(
        REAL_DIR,
        0
    )
)


print("\nProcessing FAKE audio...")

X_fake, y_fake, files_fake = (
    process_directory(
        FAKE_DIR,
        1
    )
)


print("\n========================================")
print("RAW DATASET")
print("========================================")

print(
    "REAL segments:",
    len(X_real)
)

print(
    "FAKE segments:",
    len(X_fake)
)


# ============================================================
# BALANCE DATASET
# ============================================================

minimum_count = min(
    len(X_real),
    len(X_fake)
)

print(
    "\nBalancing to:",
    minimum_count,
    "samples per class"
)


rng = np.random.RandomState(
    RANDOM_STATE
)


real_indices = rng.choice(
    len(X_real),
    size=minimum_count,
    replace=False
)


fake_indices = rng.choice(
    len(X_fake),
    size=minimum_count,
    replace=False
)


X_real_balanced = X_real[
    real_indices
]

X_fake_balanced = X_fake[
    fake_indices
]


y_real_balanced = y_real[
    real_indices
]

y_fake_balanced = y_fake[
    fake_indices
]


files_real_balanced = files_real[
    real_indices
]

files_fake_balanced = files_fake[
    fake_indices
]


X = np.vstack(
    [
        X_real_balanced,
        X_fake_balanced
    ]
)


y = np.concatenate(
    [
        y_real_balanced,
        y_fake_balanced
    ]
)


file_ids = np.concatenate(
    [
        files_real_balanced,
        files_fake_balanced
    ]
)


# Shuffle

indices = rng.permutation(
    len(X)
)


X = X[
    indices
]

y = y[
    indices
]

file_ids = file_ids[
    indices
]


print("\nBalanced dataset:")

print(
    "Features:",
    X.shape
)

print(
    "REAL:",
    np.sum(y == 0)
)

print(
    "FAKE:",
    np.sum(y == 1)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print("\n========================================")
print("TRAIN / TEST")
print("========================================")

print(
    "Training:",
    len(X_train)
)

print(
    "Testing:",
    len(X_test)
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE
        ),

    "Gaussian Naive Bayes":
        GaussianNB()
}


results = []

trained_models = {}


# ============================================================
# TRAIN MODELS
# ============================================================

for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1        : {f1 * 100:.2f}%"
    )

    print(
        f"AUC       : {auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "AUC": auc

    })

    trained_models[
        name
    ] = model


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)


print("\n========================================")
print("FINAL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_row = results_df.iloc[0]

best_name = best_row[
    "Model"
]

best_accuracy = best_row[
    "Accuracy"
]

best_model = trained_models[
    best_name
]


print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Model:",
    best_name
)

print(
    f"Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "best_model.joblib"
    )
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "sample_rate":
        SAMPLE_RATE,

    "segment_duration":
        SEGMENT_DURATION,

    "n_mfcc":
        N_MFCC,

    "feature_count":
        int(X.shape[1]),

    "feature_description":
        "40 MFCC mean + 40 MFCC standard deviation",

    "label_mapping": {

        "0": "REAL",

        "1": "FAKE"

    },

    "best_model":
        best_name,

    "accuracy":
        float(best_accuracy),

    "accuracy_percent":
        float(
            best_accuracy * 100
        )
}


with open(
    os.path.join(
        MODEL_DIR,
        "config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "real_original_segments":
        int(len(X_real)),

    "fake_original_segments":
        int(len(X_fake)),

    "balanced_segments_per_class":
        int(minimum_count),

    "total_balanced_segments":
        int(len(X)),

    "feature_count":
        int(X.shape[1]),

    "best_model":
        best_name,

    "accuracy":
        float(best_accuracy),

    "accuracy_percent":
        float(
            best_accuracy * 100
        )

}


with open(
    os.path.join(
        RESULTS_DIR,
        "training_summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Best Model : {best_name}"
)

print(
    f"Accuracy   : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    "Saved:"
)

print(
    os.path.join(
        MODEL_DIR,
        "best_model.joblib"
    )
)