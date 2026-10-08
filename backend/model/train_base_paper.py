import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.decomposition import NMF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "DATASET-balanced.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

def calculate_metrics(y_true, y_pred, y_probability):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="FAKE",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label="FAKE",
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label="FAKE",
            zero_division=0
        )
    }

    try:
        fake_index = list(y_probability.columns).index("FAKE")

        metrics["auc"] = roc_auc_score(
            (y_true == "FAKE").astype(int),
            y_probability.iloc[:, fake_index]
        )

    except Exception:
        metrics["auc"] = None

    return metrics


def print_metrics(name, metrics):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score  : {metrics['f1'] * 100:.2f}%")

    if metrics["auc"] is not None:
        print(f"AUC       : {metrics['auc']:.4f}")


def get_models():

    return {
        "GNB": GaussianNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=3
        )
    }


print("\n" + "=" * 60)
print("BASE PAPER REPLICATION")
print("=" * 60)

print("\nDataset:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nLabels:")
print(df["LABEL"].value_counts())


X = df.drop(columns=["LABEL"])
y = df["LABEL"].astype(str).str.upper()

print("\nFeature shape:", X.shape)
print("Label shape  :", y.shape)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())


print("\n\n")
print("#" * 60)
print("# EXPERIMENT A")
print("# BASELINE AUDIO FEATURES")
print("#" * 60)

baseline_models = get_models()

baseline_results = []

baseline_predictions = {}
baseline_probabilities = {}

for model_name, model in baseline_models.items():

    print("\nTraining:", model_name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = pd.DataFrame(
        model.predict_proba(X_test),
        columns=model.classes_
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability
    )

    baseline_results.append({
        "experiment": "Baseline",
        "model": model_name,
        **metrics
    })

    baseline_predictions[model_name] = y_pred
    baseline_probabilities[model_name] = y_probability

    print_metrics(model_name, metrics)

    if model_name == "GNB":

        print("\nGNB Classification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

print("\n" + "=" * 60)
print("GNB FEATURE ENHANCEMENT")
print("=" * 60)

gnb_feature_model = GaussianNB()

gnb_feature_model.fit(
    X_train,
    y_train
)

gnb_train_probability = pd.DataFrame(
    gnb_feature_model.predict_proba(X_train),
    columns=gnb_feature_model.classes_,
    index=X_train.index
)

gnb_test_probability = pd.DataFrame(
    gnb_feature_model.predict_proba(X_test),
    columns=gnb_feature_model.classes_,
    index=X_test.index
)

print(
    "GNB training probability shape:",
    gnb_train_probability.shape
)

print(
    "GNB testing probability shape:",
    gnb_test_probability.shape
)


print("\n" + "=" * 60)
print("NMF TRANSFORMATION")
print("=" * 60)


nmf_model = NMF(
    n_components=2,
    init="random",
    random_state=RANDOM_STATE,
    max_iter=1000
)

X_train_nmf = nmf_model.fit_transform(
    gnb_train_probability
)

X_test_nmf = nmf_model.transform(
    gnb_test_probability
)

print(
    "NMF training shape:",
    X_train_nmf.shape
)

print(
    "NMF testing shape:",
    X_test_nmf.shape
)

print("\n\n")
print("#" * 60)
print("# EXPERIMENT B")
print("# MFCC-GNB XTRACTNET STYLE FEATURES")
print("#" * 60)

xtract_models = get_models()

xtract_results = []

xtract_predictions = {}
xtract_probabilities = {}

for model_name, model in xtract_models.items():

    print("\nTraining:", model_name)

    model.fit(
        X_train_nmf,
        y_train
    )

    y_pred = model.predict(
        X_test_nmf
    )

    y_probability = pd.DataFrame(
        model.predict_proba(X_test_nmf),
        columns=model.classes_
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability
    )

    xtract_results.append({
        "experiment": "GNB-NMF",
        "model": model_name,
        **metrics
    })

    xtract_predictions[model_name] = y_pred
    xtract_probabilities[model_name] = y_probability

    print_metrics(model_name, metrics)

    if model_name == "GNB":

        print("\nXtractNet-style GNB Classification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = pd.DataFrame(
    baseline_results + xtract_results
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

display_results = all_results.copy()

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1"
]:
    display_results[column] = (
        display_results[column] * 100
    ).round(2)

if "auc" in display_results:
    display_results["auc"] = (
        display_results["auc"].round(4)
    )

print(display_results.to_string(index=False))


csv_path = os.path.join(
    RESULTS_DIR,
    "base_paper_results.csv"
)

all_results.to_csv(
    csv_path,
    index=False
)

print("\nResults saved to:")
print(csv_path)


json_path = os.path.join(
    RESULTS_DIR,
    "base_paper_results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_results.to_dict(orient="records"),
        file,
        indent=4
    )

print("JSON saved to:")
print(json_path)

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["REAL", "FAKE"]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(cm)

    ax.set_title(title)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["REAL", "FAKE"]
    )

    ax.set_yticklabels(
        ["REAL", "FAKE"]
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


# Baseline GNB confusion matrix
save_confusion_matrix(
    y_test,
    baseline_predictions["GNB"],
    "Baseline GNB Confusion Matrix",
    "mfcc_confusion_matrix_gnb.png"
)


# XtractNet GNB confusion matrix
save_confusion_matrix(
    y_test,
    xtract_predictions["GNB"],
    "GNB-NMF Confusion Matrix",
    "xtractnet_confusion_matrix_gnb.png"
)


plot_df = all_results.copy()

plot_df["model_label"] = (
    plot_df["experiment"]
    + " - "
    + plot_df["model"]
)

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["model_label"],
    plot_df["accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title(
    "Baseline vs GNB-NMF Model Performance"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(0, 100)

plt.tight_layout()

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", comparison_path)

plt.figure(figsize=(10, 7))

y_true_binary = (
    y_test == "FAKE"
).astype(int)

for model_name, probability in xtract_probabilities.items():

    if "FAKE" not in probability.columns:
        continue

    fake_probability = probability["FAKE"]

    fpr, tpr, _ = roc_curve(
        y_true_binary,
        fake_probability
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves - GNB-NMF Experiment"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    RESULTS_DIR,
    "roc_curves.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", roc_path)



best_result = all_results.loc[
    all_results["accuracy"].idxmax()
]

print("\n" + "=" * 60)
print("BEST RESULT")
print("=" * 60)

print(
    "Experiment:",
    best_result["experiment"]
)

print(
    "Model:",
    best_result["model"]
)

print(
    f"Accuracy: {best_result['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: {best_result['precision'] * 100:.2f}%"
)

print(
    f"Recall: {best_result['recall'] * 100:.2f}%"
)

print(
    f"F1: {best_result['f1'] * 100:.2f}%"
)

if best_result["auc"] is not None:

    print(
        f"AUC: {best_result['auc']:.4f}"
    )

print("\n" + "=" * 60)
print("BASE PAPER EXPERIMENT COMPLETED")
print("=" * 60)