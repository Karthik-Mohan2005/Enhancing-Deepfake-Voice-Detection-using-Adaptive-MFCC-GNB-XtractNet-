import os
import warnings

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

# IMPORTANT:
# V3 uses a NEW results directory.
# Existing results/ is NOT touched.
RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results_v3"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

SAMPLE_RATE = 16000

N_MFCC = 20

SEGMENT_DURATION = 1.0

SEGMENT_SAMPLES = int(
    SAMPLE_RATE * SEGMENT_DURATION
)

SNR_LEVELS = [
    None,
    20,
    10,
    5,
    0
]


# ============================================================
# LOAD AUDIO
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
        audio = audio / max_value

    return audio, sr


# ============================================================
# ADD WHITE GAUSSIAN NOISE
# ============================================================

def add_noise(
    audio,
    snr_db,
    seed
):

    if snr_db is None:
        return audio.copy()

    rng = np.random.default_rng(
        seed
    )

    signal_power = np.mean(
        audio ** 2
    )

    if signal_power <= 1e-12:
        return audio.copy()

    noise_power = (
        signal_power /
        (10 ** (snr_db / 10))
    )

    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        size=len(audio)
    )

    noisy_audio = (
        audio + noise
    )

    peak = np.max(
        np.abs(noisy_audio)
    )

    if peak > 0:
        noisy_audio = (
            noisy_audio / peak
        )

    return noisy_audio.astype(
        np.float32
    )


# ============================================================
# EXTRACT 26 FEATURES
# ============================================================

def extract_features(
    segment,
    sr
):

    features = []

    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=sr
    )

    features.append(
        np.mean(chroma)
    )

    # 2. RMS
    rms = librosa.feature.rms(
        y=segment
    )

    features.append(
        np.mean(rms)
    )

    # 3. Spectral Centroid
    centroid = (
        librosa.feature.spectral_centroid(
            y=segment,
            sr=sr
        )
    )

    features.append(
        np.mean(centroid)
    )

    # 4. Spectral Bandwidth
    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=segment,
            sr=sr
        )
    )

    features.append(
        np.mean(bandwidth)
    )

    # 5. Spectral Rolloff
    rolloff = (
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=sr
        )
    )

    features.append(
        np.mean(rolloff)
    )

    # 6. Zero Crossing Rate
    zcr = (
        librosa.feature.zero_crossing_rate(
            segment
        )
    )

    features.append(
        np.mean(zcr)
    )

    # 7-26. MFCC 1-20
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    features.extend(
        mfcc_mean.tolist()
    )

    return np.asarray(
        features,
        dtype=np.float64
    )


# ============================================================
# EXTRACT FEATURES FROM ONE FILE
# ============================================================

def extract_file_features(
    file_path,
    snr_db=None,
    seed=42
):

    audio, sr = load_audio(
        file_path
    )

    audio = add_noise(
        audio,
        snr_db,
        seed
    )

    features = []

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

        # Only complete 1-second segments
        if len(segment) < SEGMENT_SAMPLES:
            continue

        vector = extract_features(
            segment,
            sr
        )

        features.append(
            vector
        )

    if not features:

        return np.empty(
            (0, 26)
        )

    return np.asarray(
        features
    )


# ============================================================
# GET ALL AUDIO FILES
# ============================================================

def get_audio_files():

    files = []

    for filename in sorted(
        os.listdir(REAL_DIR)
    ):

        if filename.lower().endswith(".wav"):

            files.append(
                (
                    os.path.join(
                        REAL_DIR,
                        filename
                    ),
                    "REAL"
                )
            )

    for filename in sorted(
        os.listdir(FAKE_DIR)
    ):

        if filename.lower().endswith(".wav"):

            files.append(
                (
                    os.path.join(
                        FAKE_DIR,
                        filename
                    ),
                    "FAKE"
                )
            )

    return files


# ============================================================
# FILE-LEVEL SPLIT
# ============================================================

def create_file_split(
    files
):

    file_paths = np.array(
        [
            item[0]
            for item in files
        ]
    )

    labels = np.array(
        [
            item[1]
            for item in files
        ]
    )

    train_paths, test_paths = (
        train_test_split(
            file_paths,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    return (
        list(train_paths),
        list(test_paths)
    )


# ============================================================
# DETERMINE LABEL FROM PATH
# ============================================================

def get_label(
    file_path
):

    directory = os.path.dirname(
        file_path
    )

    if os.path.normcase(directory) == os.path.normcase(REAL_DIR):
        return "REAL"

    return "FAKE"


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

def build_dataset(
    file_paths,
    snr_db=None
):

    X = []

    y = []

    file_ids = []

    print("\n" + "-" * 60)

    if snr_db is None:

        print(
            "BUILDING CLEAN DATASET"
        )

    else:

        print(
            f"BUILDING DATASET - SNR {snr_db} dB"
        )

    print("-" * 60)

    for file_index, file_path in enumerate(
        file_paths
    ):

        filename = os.path.basename(
            file_path
        )

        label = get_label(
            file_path
        )

        try:

            features = extract_file_features(
                file_path,
                snr_db=snr_db,
                seed=RANDOM_STATE + file_index
            )

            for vector in features:

                X.append(vector)

                y.append(label)

                file_ids.append(
                    filename
                )

            print(
                f"{label:4} | "
                f"{filename:30} | "
                f"{len(features)} segments"
            )

        except Exception as error:

            print(
                f"ERROR | "
                f"{filename}: {error}"
            )

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(file_ids)
    )


# ============================================================
# TRAIN BALANCED RANDOM FOREST
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n" + "=" * 60)
    print("TRAINING BALANCED RANDOM FOREST")
    print("=" * 60)

    print(
        "\nTraining class distribution:"
    )

    print(
        pd.Series(
            y_train
        ).value_counts()
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# FILE-LEVEL PREDICTION
# ============================================================

def aggregate_file_predictions(
    model,
    X,
    y,
    file_ids
):

    probabilities = model.predict_proba(
        X
    )

    fake_index = list(
        model.classes_
    ).index("FAKE")

    fake_probabilities = (
        probabilities[:, fake_index]
    )

    unique_files = np.unique(
        file_ids
    )

    file_true = []
    file_probability = []
    file_prediction = []

    for filename in unique_files:

        indices = np.where(
            file_ids == filename
        )[0]

        true_label = y[
            indices[0]
        ]

        mean_fake_probability = np.mean(
            fake_probabilities[
                indices
            ]
        )

        prediction = (
            "FAKE"
            if mean_fake_probability >= 0.5
            else "REAL"
        )

        file_true.append(
            true_label
        )

        file_probability.append(
            mean_fake_probability
        )

        file_prediction.append(
            prediction
        )

    return (
        np.asarray(file_true),
        np.asarray(file_prediction),
        np.asarray(file_probability),
        unique_files
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate_file_level(
    model,
    X,
    y,
    file_ids,
    condition,
    snr_db
):

    (
        file_true,
        file_prediction,
        file_probability,
        unique_files
    ) = aggregate_file_predictions(
        model,
        X,
        y,
        file_ids
    )

    metrics = {
        "condition":
            condition,

        "snr_db":
            snr_db,

        "files":
            len(unique_files),

        "accuracy":
            accuracy_score(
                file_true,
                file_prediction
            ),

        "precision":
            precision_score(
                file_true,
                file_prediction,
                pos_label="FAKE",
                zero_division=0
            ),

        "recall":
            recall_score(
                file_true,
                file_prediction,
                pos_label="FAKE",
                zero_division=0
            ),

        "f1":
            f1_score(
                file_true,
                file_prediction,
                pos_label="FAKE",
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                (file_true == "FAKE").astype(int),
                file_probability
            )
    }

    print("\n" + "=" * 60)
    print(
        f"FILE-LEVEL RESULT: {condition}"
    )
    print("=" * 60)

    print(
        f"Files      : {len(unique_files)}"
    )

    print(
        f"Accuracy   : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-Score   : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"AUC        : "
        f"{metrics['auc']:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            file_true,
            file_prediction,
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            file_true,
            file_prediction,
            labels=["REAL", "FAKE"]
        )
    )

    return (
        metrics,
        file_true,
        file_prediction,
        file_probability,
        unique_files
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    condition,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "REAL",
            "FAKE"
        ]
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        f"File-Level Confusion Matrix\n{condition}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

    plt.yticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

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

    output_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SAVE ACCURACY GRAPH
# ============================================================

def save_accuracy_graph(
    results_df
):

    noisy = results_df[
        results_df["snr_db"].notna()
    ].copy()

    if noisy.empty:
        return

    noisy = noisy.sort_values(
        "snr_db",
        ascending=False
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        noisy["snr_db"],
        noisy["accuracy"] * 100,
        marker="o"
    )

    for x, y in zip(
        noisy["snr_db"],
        noisy["accuracy"] * 100
    ):

        plt.text(
            x,
            y + 2,
            f"{y:.1f}%",
            ha="center"
        )

    plt.xlabel(
        "Signal-to-Noise Ratio (dB)"
    )

    plt.ylabel(
        "File-Level Accuracy (%)"
    )

    plt.title(
        "File-Level Deepfake Detection under Noise"
    )

    plt.ylim(
        0,
        105
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SAVE METRICS GRAPH
# ============================================================

def save_metrics_graph(
    results_df
):

    noisy = results_df[
        results_df["snr_db"].notna()
    ].copy()

    if noisy.empty:
        return

    noisy = noisy.sort_values(
        "snr_db",
        ascending=False
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        noisy["snr_db"],
        noisy["accuracy"] * 100,
        marker="o",
        label="Accuracy"
    )

    plt.plot(
        noisy["snr_db"],
        noisy["precision"] * 100,
        marker="o",
        label="Precision"
    )

    plt.plot(
        noisy["snr_db"],
        noisy["recall"] * 100,
        marker="o",
        label="Recall"
    )

    plt.plot(
        noisy["snr_db"],
        noisy["f1"] * 100,
        marker="o",
        label="F1"
    )

    plt.xlabel(
        "Signal-to-Noise Ratio (dB)"
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "File-Level Model Performance under Noise"
    )

    plt.ylim(
        0,
        105
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("V3 BALANCED FILE-LEVEL NOISE ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    print(
        "\nOutput directory:"
    )

    print(
        RESULTS_DIR
    )

    # --------------------------------------------------------
    # 1. Get files
    # --------------------------------------------------------

    files = get_audio_files()

    print(
        "\nTotal audio files:",
        len(files)
    )

    print(
        "REAL files:",
        sum(
            1
            for _, label in files
            if label == "REAL"
        )
    )

    print(
        "FAKE files:",
        sum(
            1
            for _, label in files
            if label == "FAKE"
        )
    )

    # --------------------------------------------------------
    # 2. File-level split
    # --------------------------------------------------------

    (
        train_files,
        test_files
    ) = create_file_split(
        files
    )

    print("\n" + "=" * 60)
    print("FILE-LEVEL SPLIT")
    print("=" * 60)

    train_real = sum(
        1
        for path in train_files
        if get_label(path) == "REAL"
    )

    train_fake = sum(
        1
        for path in train_files
        if get_label(path) == "FAKE"
    )

    test_real = sum(
        1
        for path in test_files
        if get_label(path) == "REAL"
    )

    test_fake = sum(
        1
        for path in test_files
        if get_label(path) == "FAKE"
    )

    print(
        f"Training files : {len(train_files)}"
    )

    print(
        f"  REAL         : {train_real}"
    )

    print(
        f"  FAKE         : {train_fake}"
    )

    print(
        f"\nTesting files  : {len(test_files)}"
    )

    print(
        f"  REAL         : {test_real}"
    )

    print(
        f"  FAKE         : {test_fake}"
    )

    # --------------------------------------------------------
    # 3. Build clean training data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        train_ids
    ) = build_dataset(
        train_files,
        snr_db=None
    )

    print(
        "\nTraining matrix:",
        X_train.shape
    )

    # --------------------------------------------------------
    # 4. Train balanced model
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # 5. Clean test
    # --------------------------------------------------------

    (
        X_test,
        y_test,
        test_ids
    ) = build_dataset(
        test_files,
        snr_db=None
    )

    (
        clean_metrics,
        clean_true,
        clean_pred,
        clean_probability,
        clean_files
    ) = evaluate_file_level(
        model,
        X_test,
        y_test,
        test_ids,
        "Clean",
        None
    )

    results = [
        clean_metrics
    ]

    save_confusion_matrix(
        clean_true,
        clean_pred,
        "Clean",
        "confusion_clean.png"
    )

    # --------------------------------------------------------
    # 6. Noise conditions
    # --------------------------------------------------------

    for snr_db in [
        20,
        10,
        5,
        0
    ]:

        (
            X_noisy,
            y_noisy,
            noisy_ids
        ) = build_dataset(
            test_files,
            snr_db=snr_db
        )

        (
            metrics,
            true_labels,
            predictions,
            probabilities,
            unique_files
        ) = evaluate_file_level(
            model,
            X_noisy,
            y_noisy,
            noisy_ids,
            f"SNR {snr_db} dB",
            snr_db
        )

        results.append(
            metrics
        )

        save_confusion_matrix(
            true_labels,
            predictions,
            f"SNR {snr_db} dB",
            f"confusion_{snr_db}db.png"
        )

    # --------------------------------------------------------
    # 7. Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    csv_path = os.path.join(
        RESULTS_DIR,
        "noise_robustness_v3_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # 8. Print final results
    # --------------------------------------------------------

    display_df = results_df.copy()

    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    display_df["auc"] = (
        display_df["auc"]
        .round(4)
    )

    print("\n")
    print("=" * 70)
    print("V3 FINAL RESULTS")
    print("=" * 70)

    print(
        display_df.to_string(
            index=False
        )
    )

    print(
        "\nCSV saved:",
        csv_path
    )

    # --------------------------------------------------------
    # 9. Graphs
    # --------------------------------------------------------

    save_accuracy_graph(
        results_df
    )

    save_metrics_graph(
        results_df
    )

    print("\n")
    print("=" * 70)
    print("V3 EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        "\nAll V3 outputs are inside:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()