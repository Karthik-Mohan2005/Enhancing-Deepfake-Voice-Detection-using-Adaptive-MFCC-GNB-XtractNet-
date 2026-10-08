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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
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

# Noise conditions
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

    # Amplitude normalization
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

    if signal_power < 1e-12:
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

    # Normalize after adding noise
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

    # --------------------------------------------------------
    # 1. Chroma STFT
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=sr
    )

    features.append(
        np.mean(chroma)
    )

    # --------------------------------------------------------
    # 2. RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=segment
    )

    features.append(
        np.mean(rms)
    )

    # --------------------------------------------------------
    # 3. Spectral Centroid
    # --------------------------------------------------------

    centroid = (
        librosa.feature.spectral_centroid(
            y=segment,
            sr=sr
        )
    )

    features.append(
        np.mean(centroid)
    )

    # --------------------------------------------------------
    # 4. Spectral Bandwidth
    # --------------------------------------------------------

    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=segment,
            sr=sr
        )
    )

    features.append(
        np.mean(bandwidth)
    )

    # --------------------------------------------------------
    # 5. Spectral Rolloff
    # --------------------------------------------------------

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=sr
        )
    )

    features.append(
        np.mean(rolloff)
    )

    # --------------------------------------------------------
    # 6. Zero Crossing Rate
    # --------------------------------------------------------

    zcr = (
        librosa.feature.zero_crossing_rate(
            segment
        )
    )

    features.append(
        np.mean(zcr)
    )

    # --------------------------------------------------------
    # 7-26. MFCC 1-20
    # --------------------------------------------------------

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
# GET FILES
# ============================================================

def get_audio_files():

    real_files = sorted(
        [
            os.path.join(
                REAL_DIR,
                f
            )
            for f in os.listdir(REAL_DIR)
            if f.lower().endswith(".wav")
        ]
    )

    fake_files = sorted(
        [
            os.path.join(
                FAKE_DIR,
                f
            )
            for f in os.listdir(FAKE_DIR)
            if f.lower().endswith(".wav")
        ]
    )

    files = (
        [(f, "REAL") for f in real_files]
        +
        [(f, "FAKE") for f in fake_files]
    )

    return files


# ============================================================
# FILE-LEVEL TRAIN/TEST SPLIT
# ============================================================

def create_file_split(files):

    file_paths = np.array(
        [x[0] for x in files]
    )

    labels = np.array(
        [x[1] for x in files]
    )

    train_files, test_files = (
        train_test_split(
            file_paths,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    return (
        list(train_files),
        list(test_files)
    )


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

def build_dataset(
    file_paths,
    snr_db=None
):

    X = []

    y = []

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

        if os.path.dirname(file_path) == REAL_DIR:

            label = "REAL"

        else:

            label = "FAKE"

        try:

            features = extract_file_features(
                file_path,
                snr_db=snr_db,
                seed=RANDOM_STATE + file_index
            )

            for vector in features:

                X.append(vector)
                y.append(label)

            print(
                f"{label:4} | "
                f"{filename:30} | "
                f"{len(features)} segments"
            )

        except Exception as e:

            print(
                f"ERROR | "
                f"{filename}: {e}"
            )

    return (
        np.asarray(X),
        np.asarray(y)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_baseline(
    X_train,
    y_train
):

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X_test,
    y_test,
    condition,
    snr_db
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    fake_index = list(
        model.classes_
    ).index("FAKE")

    fake_probability = (
        probabilities[:, fake_index]
    )

    metrics = {
        "condition": condition,
        "snr_db": snr_db,

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                pos_label="FAKE",
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                pos_label="FAKE",
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                predictions,
                pos_label="FAKE",
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                (y_test == "FAKE").astype(int),
                fake_probability
            )
    }

    print("\nCondition:", condition)

    print(
        f"Samples    : {len(y_test)}"
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

    return metrics, predictions


# ============================================================
# CONFUSION MATRIX
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
        labels=["REAL", "FAKE"]
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        f"Random Forest - {condition}"
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

    output = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


# ============================================================
# ACCURACY GRAPH
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
            y + 1,
            f"{y:.2f}%",
            ha="center"
        )

    plt.xlabel(
        "Signal-to-Noise Ratio (dB)"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Random Forest Noise Robustness"
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

    output = os.path.join(
        RESULTS_DIR,
        "noise_robustness_v2_accuracy.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


# ============================================================
# METRIC GRAPH
# ============================================================

def save_metric_graph(
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
        "Random Forest Performance under Noise"
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

    output = os.path.join(
        RESULTS_DIR,
        "noise_robustness_v2_metrics.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CONTROLLED NOISE ROBUSTNESS EXPERIMENT - V2")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Get raw audio files
    # --------------------------------------------------------

    files = get_audio_files()

    print(
        "\nTotal audio files:",
        len(files)
    )

    real_count = sum(
        1 for _, label in files
        if label == "REAL"
    )

    fake_count = sum(
        1 for _, label in files
        if label == "FAKE"
    )

    print(
        "REAL files:",
        real_count
    )

    print(
        "FAKE files:",
        fake_count
    )

    # --------------------------------------------------------
    # 2. File-level split
    # --------------------------------------------------------

    train_files, test_files = (
        create_file_split(files)
    )

    print("\n" + "=" * 60)
    print("FILE-LEVEL SPLIT")
    print("=" * 60)

    print(
        "Training files:",
        len(train_files)
    )

    print(
        "Testing files :",
        len(test_files)
    )

    # --------------------------------------------------------
    # 3. TRAIN CLEAN MODEL
    # --------------------------------------------------------

    X_train, y_train = build_dataset(
        train_files,
        snr_db=None
    )

    print(
        "\nTraining feature matrix:",
        X_train.shape
    )

    print(
        "Training labels:",
        pd.Series(y_train).value_counts().to_dict()
    )

    model = train_baseline(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # 4. CLEAN TEST
    # --------------------------------------------------------

    X_test_clean, y_test_clean = (
        build_dataset(
            test_files,
            snr_db=None
        )
    )

    clean_metrics, clean_predictions = evaluate(
        model,
        X_test_clean,
        y_test_clean,
        "Clean",
        None
    )

    save_confusion_matrix(
        y_test_clean,
        clean_predictions,
        "Clean",
        "noise_v2_confusion_clean.png"
    )

    results = [
        clean_metrics
    ]

    # --------------------------------------------------------
    # 5. NOISE TESTS
    # --------------------------------------------------------

    for snr_db in [
        20,
        10,
        5,
        0
    ]:

        X_test_noisy, y_test_noisy = (
            build_dataset(
                test_files,
                snr_db=snr_db
            )
        )

        condition = (
            f"SNR {snr_db} dB"
        )

        metrics, predictions = evaluate(
            model,
            X_test_noisy,
            y_test_noisy,
            condition,
            snr_db
        )

        results.append(
            metrics
        )

        safe_name = (
            f"noise_v2_confusion_"
            f"{snr_db}db.png"
        )

        save_confusion_matrix(
            y_test_noisy,
            predictions,
            condition,
            safe_name
        )

    # --------------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_csv = os.path.join(
        RESULTS_DIR,
        "noise_robustness_v2_results.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL NOISE ROBUSTNESS RESULTS")
    print("=" * 60)

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

    print(
        display_df.to_string(
            index=False
        )
    )

    print(
        "\nResults saved:",
        output_csv
    )

    # --------------------------------------------------------
    # 7. Graphs
    # --------------------------------------------------------

    save_accuracy_graph(
        results_df
    )

    save_metric_graph(
        results_df
    )

    print("\n")
    print("=" * 70)
    print("CONTROLLED NOISE EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()