import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "dataset/DATASET-balanced.csv"

MODEL_PATH = "model/deepfake_csv_model.pkl"

RANDOM_STATE = 42

TEST_SIZE = 0.20

NMF_COMPONENTS = 2


# ============================================================
# LOAD CSV
# ============================================================

def load_dataset():

    print("\n========================================")
    print("LOADING BALANCED CSV DATASET")
    print("========================================")

    df = pd.read_csv(CSV_PATH)

    print("Dataset shape:", df.shape)

    print("\nLabels:")
    print(df["LABEL"].value_counts())

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X = df.drop(columns=["LABEL"])

    y = df["LABEL"].map({
        "REAL": 0,
        "FAKE": 1
    })

    X = X.astype(np.float64)

    y = y.astype(int)

    print("\nFeature shape:", X.shape)

    print("Label shape:", y.shape)

    return X.values, y.values


# ============================================================
# MAIN
# ============================================================

def train():

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    X, y = load_dataset()

    # --------------------------------------------------------
    # 2. Train / test split
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAIN / TEST SPLIT")
    print("========================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print("\nTraining REAL:",
          np.sum(y_train == 0))

    print("Training FAKE:",
          np.sum(y_train == 1))

    # --------------------------------------------------------
    # 3. SMOTE
    # --------------------------------------------------------

    print("\n========================================")
    print("SMOTE BALANCING")
    print("========================================")

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train_balanced, y_train_balanced = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "Before SMOTE:",
        X_train.shape
    )

    print(
        "After SMOTE:",
        X_train_balanced.shape
    )

    print(
        "REAL:",
        np.sum(y_train_balanced == 0)
    )

    print(
        "FAKE:",
        np.sum(y_train_balanced == 1)
    )

    # --------------------------------------------------------
    # 4. GNB feature enhancement
    # --------------------------------------------------------

    print("\n========================================")
    print("GNB FEATURE ENHANCEMENT")
    print("========================================")

    gnb_feature_model = GaussianNB()

    gnb_feature_model.fit(
        X_train_balanced,
        y_train_balanced
    )

    X_train_gnb = (
        gnb_feature_model
        .predict_proba(X_train_balanced)
    )

    X_test_gnb = (
        gnb_feature_model
        .predict_proba(X_test)
    )

    print(
        "GNB training shape:",
        X_train_gnb.shape
    )

    print(
        "GNB testing shape:",
        X_test_gnb.shape
    )

    # --------------------------------------------------------
    # 5. NMF
    # --------------------------------------------------------

    print("\n========================================")
    print("NMF TRANSFORMATION")
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

    # --------------------------------------------------------
    # 6. Final GNB classifier
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL GNB CLASSIFIER")
    print("========================================")

    final_gnb = GaussianNB()

    final_gnb.fit(
        X_train_nmf,
        y_train_balanced
    )

    # --------------------------------------------------------
    # 7. Prediction
    # --------------------------------------------------------

    y_pred = final_gnb.predict(
        X_test_nmf
    )

    y_prob = final_gnb.predict_proba(
        X_test_nmf
    )[:, 1]

    # --------------------------------------------------------
    # 8. Metrics
    # --------------------------------------------------------

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # 9. Results
    # --------------------------------------------------------

    print("\n========================================")
    print("BASELINE RESULTS")
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

    # --------------------------------------------------------
    # 10. Save model
    # --------------------------------------------------------

    model_package = {

        "feature_model": gnb_feature_model,

        "nmf": nmf,

        "classifier": final_gnb,

        "feature_columns": list(
            pd.read_csv(CSV_PATH).drop(
                columns=["LABEL"]
            ).columns
        ),

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
        MODEL_PATH
    )

    print("\n========================================")
    print("MODEL SAVED")
    print("========================================")

    print(
        "Saved to:",
        MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()