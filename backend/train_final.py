import os
import json
import warnings
import joblib
import librosa
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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REAL_DIR = os.path.join(BASE_DIR, "dataset", "REAL")
FAKE_DIR = os.path.join(BASE_DIR, "dataset", "FAKE")

RESULTS_DIR = os.path.join(BASE_DIR, "results_final")
MODEL_DIR = os.path.join(BASE_DIR, "models_final")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
N_MFCC = 20
N_MELS = 40
N_FFT = 512


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("Empty audio")

        # Normalize
        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        # ----------------------------------------------------
        # MFCC - 20
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc_features = np.mean(
            mfcc.T,
            axis=0
        )

        # ----------------------------------------------------
        # MEL SPECTROGRAM - 40
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # STFT / SPECTRAL FEATURES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        features = np.concatenate(
            [
                mfcc_features,
                mel_features,
                spec_features
            ]
        )

        return features

    except Exception as e:

        print(
            f"[WARNING] Could not process {file_path}: {e}"
        )

        return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    X = []
    y = []
    files = []

    # REAL = 0
    # FAKE = 1

    for folder, label in [
        (REAL_DIR, 0),
        (FAKE_DIR, 1)
    ]:

        if not os.path.isdir(folder):

            print(
                f"[ERROR] Directory not found: {folder}"
            )

            continue

        for filename in sorted(os.listdir(folder)):

            if not filename.lower().endswith(
                (".wav", ".mp3", ".flac", ".ogg")
            ):
                continue

            path = os.path.join(
                folder,
                filename
            )

            print(
                f"Extracting: {filename}"
            )

            features = extract_features(path)

            if features is not None:

                X.append(features)
                y.append(label)
                files.append(path)

    X = np.asarray(X)
    y = np.asarray(y)

    print("\n==========================================")
    print("DATASET")
    print("==========================================")

    print("Samples :", len(X))
    print("Features:", X.shape[1])

    print(
        "REAL:",
        np.sum(y == 0)
    )

    print(
        "FAKE:",
        np.sum(y == 1)
    )

    return X, y, files


# ============================================================
# MAIN
# ============================================================

def main():

    X, y, files = load_dataset()

    if len(X) < 10:

        raise RuntimeError(
            "Dataset is too small for training."
        )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n==========================================")
    print("TRAIN / TEST")
    print("==========================================")

    print("Training:", len(X_train))
    print("Testing :", len(X_test))

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance"
            ),

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42
            ),

        "Gaussian Naive Bayes":
            GaussianNB()
    }

    results = []

    best_model = None
    best_name = None
    best_accuracy = -1

    # --------------------------------------------------------
    # TRAIN ALL MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        print("\n==========================================")
        print(name)
        print("==========================================")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            auc = roc_auc_score(
                y_test,
                probabilities
            )

        except Exception:

            auc = 0.0

        print(
            f"Accuracy : {accuracy * 100:.2f}%"
        )

        print(
            f"Precision: {precision * 100:.2f}%"
        )

        print(
            f"Recall   : {recall * 100:.2f}%"
        )

        print(
            f"F1       : {f1 * 100:.2f}%"
        )

        print(
            f"AUC      : {auc:.4f}"
        )

        print(
            classification_report(
                y_test,
                predictions,
                target_names=["REAL", "FAKE"],
                zero_division=0
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

        # Save every model

        filename = name.lower().replace(
            " ",
            "_"
        ) + ".joblib"

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                filename
            )
        )

        # Confusion matrix

        cm = confusion_matrix(
            y_test,
            predictions
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["REAL", "FAKE"]
        )

        disp.plot()

        plt.title(
            f"{name} - Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                name.lower().replace(
                    " ",
                    "_"
                ) + "_confusion_matrix.png"
            ),
            dpi=200
        )

        plt.close()

        # Best model

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_name = name

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Accuracy",
        ascending=False
    )

    print("\n==========================================")
    print("FINAL MODEL COMPARISON")
    print("==========================================")

    print(results_df.to_string(index=False))

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"] * 100
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.title(
        "Deepfake Voice Detection - Model Comparison"
    )

    plt.xticks(
        rotation=20
    )

    plt.ylim(
        0,
        100
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        os.path.join(
            MODEL_DIR,
            "best_model.joblib"
        )
    )

    # --------------------------------------------------------
    # FEATURE CONFIG
    # --------------------------------------------------------

    config = {

        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "feature_count": int(X.shape[1]),
        "label_mapping": {
            "0": "REAL",
            "1": "FAKE"
        },
        "best_model": best_name,
        "best_accuracy": float(best_accuracy)

    }

    with open(
        os.path.join(
            MODEL_DIR,
            "feature_config.json"
        ),
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "dataset_samples": int(len(X)),
        "feature_count": int(X.shape[1]),
        "real_samples": int(np.sum(y == 0)),
        "fake_samples": int(np.sum(y == 1)),
        "test_samples": int(len(X_test)),
        "best_algorithm": best_name,
        "best_accuracy": float(best_accuracy),
        "best_accuracy_percent":
            float(best_accuracy * 100)

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

    print("\n==========================================")
    print("TRAINING COMPLETE")
    print("==========================================")

    print(
        f"BEST MODEL : {best_name}"
    )

    print(
        f"ACCURACY   : {best_accuracy * 100:.2f}%"
    )

    print(
        f"MODEL FILE : {os.path.join(MODEL_DIR, 'best_model.joblib')}"
    )


if __name__ == "__main__":
    main()