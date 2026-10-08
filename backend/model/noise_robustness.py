import os
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
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

CSV_PATH = os.path.join(
    DATASET_DIR,
    "DATASET-balanced.csv"
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

SNR_LEVELS = [
    None, 
    20,
    10,
    5,
    0
]

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


def add_gaussian_noise(
    audio,
    snr_db,
    rng
):


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

    if peak > 1:
        noisy_audio = (
            noisy_audio / peak
        )

    return noisy_audio.astype(
        np.float32
    )

def extract_features(
    audio,
    sr
):

    features = []

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.append(
        np.mean(chroma)
    )


    rms = librosa.feature.rms(
        y=audio
    )

    features.append(
        np.mean(rms)
    )


    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    features.append(
        np.mean(spectral_centroid)
    )


    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    features.append(
        np.mean(spectral_bandwidth)
    )


    rolloff = (
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )
    )

    features.append(
        np.mean(rolloff)
    )


    zcr = (
        librosa.feature.zero_crossing_rate(
            audio
        )
    )

    features.append(
        np.mean(zcr)
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
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


def extract_file_features(
    file_path,
    snr_db=None
):

    audio, sr = load_audio(
        file_path
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if snr_db is not None:

        audio = add_gaussian_noise(
            audio,
            snr_db,
            rng
        )

    segment_length = SAMPLE_RATE

    feature_vectors = []

    for start in range(
        0,
        len(audio),
        segment_length
    ):

        end = (
            start +
            segment_length
        )

        segment = audio[
            start:end
        ]

        if len(segment) < segment_length:
            continue

        feature_vector = extract_features(
            segment,
            sr
        )

        feature_vectors.append(
            feature_vector
        )

    if not feature_vectors:
        return np.empty(
            (0, 26)
        )

    return np.asarray(
        feature_vectors
    )


def load_raw_dataset():

    X = []

    y = []

    file_names = []

    print("\n" + "=" * 60)
    print("LOADING RAW AUDIO DATASET")
    print("=" * 60)

    real_files = sorted(
        [
            f
            for f in os.listdir(REAL_DIR)
            if f.lower().endswith(".wav")
        ]
    )

    print(
        "\nREAL files:",
        len(real_files)
    )

    for filename in real_files:

        path = os.path.join(
            REAL_DIR,
            filename
        )

        try:

            features = extract_file_features(
                path
            )

            for vector in features:

                X.append(vector)
                y.append("REAL")
                file_names.append(filename)

            print(
                f"REAL : {filename} "
                f"-> {len(features)} segments"
            )

        except Exception as e:

            print(
                f"ERROR REAL {filename}: {e}"
            )

    fake_files = sorted(
        [
            f
            for f in os.listdir(FAKE_DIR)
            if f.lower().endswith(".wav")
        ]
    )

    print(
        "\nFAKE files:",
        len(fake_files)
    )

    for filename in fake_files:

        path = os.path.join(
            FAKE_DIR,
            filename
        )

        try:

            features = extract_file_features(
                path
            )

            for vector in features:

                X.append(vector)
                y.append("FAKE")
                file_names.append(filename)

            print(
                f"FAKE : {filename} "
                f"-> {len(features)} segments"
            )

        except Exception as e:

            print(
                f"ERROR FAKE {filename}: {e}"
            )

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(file_names)
    )


def create_file_split(
    file_names,
    labels
):

    unique_files = np.unique(
        file_names
    )

    file_labels = []

    for filename in unique_files:

        indices = np.where(
            file_names == filename
        )[0]

        file_labels.append(
            labels[indices[0]]
        )

    file_labels = np.asarray(
        file_labels
    )

    train_files, test_files = (
        train_test_split(
            unique_files,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=file_labels
        )
    )

    train_files = set(
        train_files
    )

    test_files = set(
        test_files
    )

    train_indices = []
    test_indices = []

    for index, filename in enumerate(
        file_names
    ):

        if filename in train_files:

            train_indices.append(
                index
            )

        elif filename in test_files:

            test_indices.append(
                index
            )

    return (
        np.asarray(train_indices),
        np.asarray(test_indices)
    )


def load_csv_dataset():

    print("\n" + "=" * 60)
    print("LOADING CLEAN CSV DATASET")
    print("=" * 60)

    df = pd.read_csv(
        CSV_PATH
    )

    X = df.drop(
        columns=["LABEL"]
    )

    y = df["LABEL"]

    X = X.astype(
        np.float64
    )

    y = y.astype(
        str
    ).str.upper()

    print(
        "CSV shape:",
        df.shape
    )

    print(
        "\nLabels:"
    )

    print(
        y.value_counts()
    )

    return (
        X.values,
        y.values
    )

def train_model():

    X, y = load_csv_dataset()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print("\n" + "=" * 60)
    print("TRAINING CLEAN BASELINE")
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

    prediction = model.predict(
        X_test
    )

    probability = model.predict_proba(
        X_test
    )

    fake_index = list(
        model.classes_
    ).index("FAKE")

    metrics = {
        "condition": "Clean CSV Test",
        "snr_db": None,
        "accuracy": accuracy_score(
            y_test,
            prediction
        ),
        "precision": precision_score(
            y_test,
            prediction,
            pos_label="FAKE",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            prediction,
            pos_label="FAKE",
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            prediction,
            pos_label="FAKE",
            zero_division=0
        ),
        "auc": roc_auc_score(
            (y_test == "FAKE").astype(int),
            probability[:, fake_index]
        )
    }

    print("\nClean CSV baseline:")

    print(
        f"Accuracy  : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1        : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"AUC       : "
        f"{metrics['auc']:.4f}"
    )

    return model, metrics

def evaluate_noise(
    model,
    snr_db,
    X_raw,
    y_raw,
    file_names
):

    predictions = []
    probabilities = []
    true_labels = []

    print("\n" + "-" * 60)

    if snr_db is None:

        print(
            "Evaluating RAW AUDIO - CLEAN"
        )

    else:

        print(
            f"Evaluating RAW AUDIO - "
            f"SNR {snr_db} dB"
        )

    print("-" * 60)

    processed_files = set()

    for filename in file_names:

        if filename in processed_files:
            continue

        processed_files.add(
            filename
        )

        if filename in [
            os.path.basename(p)
            for p in []
        ]:
            continue

        indices = np.where(
            file_names == filename
        )[0]

        if len(indices) == 0:
            continue

        label = y_raw[
            indices[0]
        ]

        if label == "REAL":

            file_path = os.path.join(
                REAL_DIR,
                filename
            )

        else:

            file_path = os.path.join(
                FAKE_DIR,
                filename
            )

        try:

            features = extract_file_features(
                file_path,
                snr_db=snr_db
            )

            if len(features) == 0:
                continue

            preds = model.predict(
                features
            )

            probs = model.predict_proba(
                features
            )

            fake_index = list(
                model.classes_
            ).index("FAKE")

            for pred, prob in zip(
                preds,
                probs[:, fake_index]
            ):

                predictions.append(
                    pred
                )

                probabilities.append(
                    prob
                )

                true_labels.append(
                    label
                )

        except Exception as e:

            print(
                f"Error processing "
                f"{filename}: {e}"
            )

    if len(true_labels) == 0:

        print(
            "No valid raw-audio samples."
        )

        return None

    true_labels = np.asarray(
        true_labels
    )

    predictions = np.asarray(
        predictions
    )

    probabilities = np.asarray(
        probabilities
    )

    metrics = {
        "condition":
            "Clean" if snr_db is None
            else f"SNR {snr_db} dB",

        "snr_db":
            snr_db,

        "accuracy":
            accuracy_score(
                true_labels,
                predictions
            ),

        "precision":
            precision_score(
                true_labels,
                predictions,
                pos_label="FAKE",
                zero_division=0
            ),

        "recall":
            recall_score(
                true_labels,
                predictions,
                pos_label="FAKE",
                zero_division=0
            ),

        "f1":
            f1_score(
                true_labels,
                predictions,
                pos_label="FAKE",
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                (true_labels == "FAKE").astype(int),
                probabilities
            )
    }

    print(
        f"Samples    : {len(true_labels)}"
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
        f"F1         : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"AUC        : "
        f"{metrics['auc']:.4f}"
    )

    return metrics


def plot_results(
    results
):

    df = pd.DataFrame(
        results
    )

    plot_df = df[
        df["snr_db"].notna()
    ].copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(
        "snr_db",
        ascending=False
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        plot_df["snr_db"],
        plot_df["accuracy"] * 100,
        marker="o"
    )

    plt.xlabel(
        "Signal-to-Noise Ratio (dB)"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Deepfake Detection Accuracy under Noise"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "noise_robustness_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved:",
        output_path
    )


def plot_all_metrics(
    results
):

    df = pd.DataFrame(
        results
    )

    plot_df = df[
        df["snr_db"].notna()
    ].copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(
        "snr_db",
        ascending=False
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        plot_df["snr_db"],
        plot_df["precision"] * 100,
        marker="o",
        label="Precision"
    )

    plt.plot(
        plot_df["snr_db"],
        plot_df["recall"] * 100,
        marker="o",
        label="Recall"
    )

    plt.plot(
        plot_df["snr_db"],
        plot_df["f1"] * 100,
        marker="o",
        label="F1"
    )

    plt.plot(
        plot_df["snr_db"],
        plot_df["accuracy"] * 100,
        marker="o",
        label="Accuracy"
    )

    plt.xlabel(
        "Signal-to-Noise Ratio (dB)"
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "Deepfake Detection Performance under Noise"
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "noise_robustness_metrics.png"
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


def main():

    print("\n")
    print("=" * 70)
    print("NOISE ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    model, clean_metric = train_model()


    X_raw, y_raw, file_names = (
        load_raw_dataset()
    )

    print("\nRaw feature matrix:")
    print(
        X_raw.shape
    )

    print(
        "Raw labels:",
        len(y_raw)
    )
    results = []

    for snr_db in SNR_LEVELS:

        metric = evaluate_noise(
            model,
            snr_db,
            X_raw,
            y_raw,
            file_names
        )

        if metric is not None:

            results.append(
                metric
            )

    results_df = pd.DataFrame(
        results
    )

    csv_output = os.path.join(
        RESULTS_DIR,
        "noise_robustness_results.csv"
    )

    results_df.to_csv(
        csv_output,
        index=False
    )

    print(
        "\nResults saved:",
        csv_output
    )

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
    print("NOISE ROBUSTNESS RESULTS")
    print("=" * 70)

    print(
        display_df.to_string(
            index=False
        )
    )

    plot_results(
        results
    )

    plot_all_metrics(
        results
    )

    print("\n")
    print("=" * 70)
    print("NOISE ROBUSTNESS EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()