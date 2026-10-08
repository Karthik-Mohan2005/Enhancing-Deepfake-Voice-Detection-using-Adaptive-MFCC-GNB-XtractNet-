import os
import time
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

from mfcc_features import (
    extract_features_from_file
)

DATASET_PATH = "dataset"

REAL_DIR = os.path.join(
    DATASET_PATH,
    "REAL"
)

FAKE_DIR = os.path.join(
    DATASET_PATH,
    "FAKE"
)

MODEL_DIR = "."

MODEL_OUTPUT = os.path.join(
    MODEL_DIR,
    "deepfake_model.pkl"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42

NMF_COMPONENTS = 2

def load_dataset():

    X = []
    y = []
    file_names = []

    print("\n========================================")
    print("LOADING DATASET")
    print("========================================")
    print("\nLoading REAL audio...")

    real_files = sorted(
        [
            f for f in os.listdir(REAL_DIR)
            if f.lower().endswith(".wav")
        ]
    )

    for filename in real_files:

        file_path = os.path.join(
            REAL_DIR,
            filename
        )

        try:

            features = extract_features_from_file(
                file_path
            )

            for feature_vector in features:

                X.append(feature_vector)
                y.append(0)
                file_names.append(filename)

            print(
                f"REAL  : {filename} "
                f"-> {len(features)} segments"
            )

        except Exception as e:

            print(
                f"ERROR processing {filename}: {e}"
            )

    print("\nLoading FAKE audio...")

    fake_files = sorted(
        [
            f for f in os.listdir(FAKE_DIR)
            if f.lower().endswith(".wav")
        ]
    )

    for filename in fake_files:

        file_path = os.path.join(
            FAKE_DIR,
            filename
        )

        try:

            features = extract_features_from_file(
                file_path
            )

            for feature_vector in features:

                X.append(feature_vector)
                y.append(1)
                file_names.append(filename)

            print(
                f"FAKE  : {filename} "
                f"-> {len(features)} segments"
            )

        except Exception as e:

            print(
                f"ERROR processing {filename}: {e}"
            )

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, file_names

def create_file_level_split(file_names, labels):

    unique_files = np.unique(file_names)

    file_labels = []

    for filename in unique_files:

        indices = np.where(
            np.asarray(file_names) == filename
        )[0]

        file_labels.append(
            labels[indices[0]]
        )

    file_labels = np.asarray(file_labels)

    train_files, test_files = train_test_split(
        unique_files,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=file_labels
    )

    train_files = set(train_files)
    test_files = set(test_files)

    train_indices = []
    test_indices = []

    for index, filename in enumerate(file_names):

        if filename in train_files:
            train_indices.append(index)

        elif filename in test_files:
            test_indices.append(index)

    return (
        np.asarray(train_indices),
        np.asarray(test_indices)
    )


def train_model():

    start_time = time.time()

    X, y, file_names = load_dataset()

    print("\n========================================")
    print("DATASET SUMMARY")
    print("========================================")

    print("Total samples :", len(X))
    print("REAL samples  :", np.sum(y == 0))
    print("FAKE samples  :", np.sum(y == 1))
    print("Feature count :", X.shape[1])

    print("\n========================================")
    print("TRAIN / TEST SPLIT")
    print("========================================")

    train_indices, test_indices = create_file_level_split(
        file_names,
        y
    )

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    print("Training samples :", len(X_train))
    print("Testing samples  :", len(X_test))

    print(
        "Training REAL    :",
        np.sum(y_train == 0)
    )

    print(
        "Training FAKE    :",
        np.sum(y_train == 1)
    )

    print(
        "Testing REAL     :",
        np.sum(y_test == 0)
    )

    print(
        "Testing FAKE     :",
        np.sum(y_test == 1)
    )

    print("\n========================================")
    print("APPLYING SMOTE")
    print("========================================")

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    print(
        "Before SMOTE:",
        X_train.shape
    )

    print(
        "After SMOTE :",
        X_train_balanced.shape
    )

    print(
        "REAL after SMOTE:",
        np.sum(y_train_balanced == 0)
    )

    print(
        "FAKE after SMOTE:",
        np.sum(y_train_balanced == 1)
    )

    print("\n========================================")
    print("GNB FEATURE ENHANCEMENT")
    print("========================================")

    gnb_feature_model = GaussianNB()

    gnb_feature_model.fit(
        X_train_balanced,
        y_train_balanced
    )

    X_train_gnb = gnb_feature_model.predict_proba(
        X_train_balanced
    )

    X_test_gnb = gnb_feature_model.predict_proba(
        X_test
    )

    print(
        "GNB transformed training shape:",
        X_train_gnb.shape
    )

    print(
        "GNB transformed testing shape:",
        X_test_gnb.shape
    )

    print("\n========================================")
    print("NMF FEATURE TRANSFORMATION")
    print("========================================")

    X_train_gnb = np.maximum(
        X_train_gnb,
        0
    )

    X_test_gnb = np.maximum(
        X_test_gnb,
        0
    )

    nmf = NMF(
        n_components=NMF_COMPONENTS,
        init="random",
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    X_train_nmf = nmf.fit_transform(
        X_train_gnb
    )

    X_test_nmf = nmf.transform(
        X_test_gnb
    )

    print(
        "NMF training shape:",
        X_train_nmf.shape
    )

    print(
        "NMF testing shape:",
        X_test_nmf.shape
    )


    print("\n========================================")
    print("FINAL GNB CLASSIFIER")
    print("========================================")

    final_gnb = GaussianNB()

    final_gnb.fit(
        X_train_nmf,
        y_train_balanced
    )

    y_pred = final_gnb.predict(
        X_test_nmf
    )

    y_probability = final_gnb.predict_proba(
        X_test_nmf
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

    try:

        auc = roc_auc_score(
            y_test,
            y_probability
        )

    except ValueError:

        auc = 0.0

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n========================================")
    print("MODEL RESULTS")
    print("========================================")

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
        f"F1-Score  : {f1 * 100:.2f}%"
    )

    print(
        f"AUC       : {auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "REAL",
                "DEEPFAKE"
            ],
            zero_division=0
        )
    )
    model_package = {

        "feature_model": gnb_feature_model,

        "nmf": nmf,

        "classifier": final_gnb,

        "sample_rate": 16000,

        "n_mfcc": 40,

        "nmf_components": NMF_COMPONENTS,

        "metrics": {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "auc": auc
        }
    }

    joblib.dump(
        model_package,
        MODEL_OUTPUT
    )

    elapsed_time = time.time() - start_time

    print("\n========================================")
    print("MODEL SAVED")
    print("========================================")

    print(
        f"Location: {MODEL_OUTPUT}"
    )

    print(
        f"Training time: {elapsed_time:.2f} seconds"
    )

if __name__ == "__main__":

    train_model()