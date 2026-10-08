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

CSV_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "DATASET-balanced.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results_final"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models_final"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("FINAL DEEPFAKE VOICE TRAINING")
print("========================================")

df = pd.read_csv(CSV_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nLabel distribution:")
print(df["LABEL"].value_counts())


# ============================================================
# FEATURES / LABEL
# ============================================================

X = df.drop(
    columns=["LABEL"]
)

y = df["LABEL"].map({
    "REAL": 0,
    "FAKE": 1
})

X = X.astype(np.float64)
y = y.astype(int)

feature_columns = list(X.columns)

print("\nFeature shape:")
print(X.shape)

print("\nNumber of features:")
print(len(feature_columns))


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

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

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)

print(
    "Training REAL:",
    np.sum(y_train == 0)
)

print(
    "Training FAKE:",
    np.sum(y_train == 1)
)

print(
    "Testing REAL:",
    np.sum(y_test == 0)
)

print(
    "Testing FAKE:",
    np.sum(y_test == 1)
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1
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


# ============================================================
# TRAIN
# ============================================================

results = []

trained_models = {}


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
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print(
        f"AUC       : {auc:.4f}"
    )

    print("\nClassification Report:")

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "AUC": auc

    })

    trained_models[name] = model

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_filename = (
        name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            model_filename
        )
    )


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

results_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# PLOT MODEL COMPARISON
# ============================================================

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
    "Deepfake Voice Detection - Algorithm Comparison"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=20
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


# ============================================================
# BEST MODEL
# ============================================================

best_row = results_df.iloc[0]

best_name = best_row["Model"]

best_accuracy = best_row["Accuracy"]

best_model = trained_models[
    best_name
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Algorithm:",
    best_name
)

print(
    f"Accuracy: {best_accuracy * 100:.2f}%"
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
# SAVE FEATURE CONFIG
# ============================================================

config = {

    "feature_columns": feature_columns,

    "feature_count": len(
        feature_columns
    ),

    "label_mapping": {
        "0": "REAL",
        "1": "FAKE"
    },

    "best_model": best_name,

    "best_accuracy": float(
        best_accuracy
    )

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


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "dataset_size": len(df),

    "feature_count": len(
        feature_columns
    ),

    "real_samples": int(
        np.sum(y == 0)
    ),

    "fake_samples": int(
        np.sum(y == 1)
    ),

    "training_samples": len(
        X_train
    ),

    "testing_samples": len(
        X_test
    ),

    "best_algorithm": best_name,

    "accuracy": float(
        best_accuracy
    ),

    "accuracy_percent": float(
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
    f"Accuracy   : {best_accuracy * 100:.2f}%"
)

print(
    "Model saved:"
)

print(
    os.path.join(
        MODEL_DIR,
        "best_model.joblib"
    )
)