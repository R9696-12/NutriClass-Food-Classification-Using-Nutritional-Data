import os
import glob
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report

NUMERIC_COLS = ["Calories","Protein","Carbs","Fat","Sugar","Fiber","Sodium"]
TARGET = "label"
DATA_TEST = os.path.join("data", "processed_test.csv")
MODELS_DIR = "models"
OUT_FIG_DIR = os.path.join("outputs", "figures")

def pick_model(models_dir: str) -> str | None:
    pats = glob.glob(os.path.join(models_dir, "*.joblib"))
    pats = [p for p in pats if not any(x in os.path.basename(p) for x in ("label_encoder","scaler"))]
    if not pats:
        return None
    # pick most recently modified
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pats[0]

def load_label_encoder(models_dir: str):
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    if os.path.exists(le_path):
        try:
            return joblib.load(le_path)
        except Exception:
            pass
    return None

def main():
    if not os.path.exists(DATA_TEST):
        print(f"Test file not found: {DATA_TEST}")
        return
    test = pd.read_csv(DATA_TEST)
    missing = [c for c in NUMERIC_COLS if c not in test.columns]
    if missing:
        print("Missing expected feature columns:", missing)
        return
    X_test = test[NUMERIC_COLS].values
    y_test = test[TARGET].values

    model_path = pick_model(MODELS_DIR)
    if model_path is None:
        print(f"No trained model found in {MODELS_DIR}")
        return
    model = joblib.load(model_path)
    print("Using model:", model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Optional classification report
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Confusion matrix with label names if encoder available
    le = load_label_encoder(MODELS_DIR)
    display_labels = getattr(le, "classes_", None)
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=display_labels if display_labels is not None else None,
                yticklabels=display_labels if display_labels is not None else None)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    cm_path = os.path.join(OUT_FIG_DIR, "confusion_matrix.png")
    fig.savefig(cm_path)
    print("Saved", cm_path)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.bar(NUMERIC_COLS, fi)
        ax2.set_xticklabels(NUMERIC_COLS, rotation=45, ha="right")
        ax2.set_ylabel("Importance")
        fig2.tight_layout()
        fi_path = os.path.join(OUT_FIG_DIR, "feature_importances.png")
        fig2.savefig(fi_path)
        print("Saved", fi_path)
    else:
        print("Model has no feature_importances_ attribute; skipping.")

if __name__ == "__main__":
    main()
