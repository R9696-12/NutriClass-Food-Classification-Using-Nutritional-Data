
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import save_model
import joblib

RANDOM_STATE = 42
# Use capitalized column names to match processed CSVs
NUMERIC_COLS = ["Calories","Protein","Carbs","Fat","Sugar","Fiber","Sodium"]
TARGET = "label"

def load_processed():
    train = pd.read_csv("C:\\Nutri\\data\\processed_train.csv")
    test = pd.read_csv("C:\\Nutri\\data\\processed_test.csv")
    X_train = train[NUMERIC_COLS].values
    y_train = train[TARGET].values
    X_test = test[NUMERIC_COLS].values
    y_test = test[TARGET].values
    return X_train, X_test, y_train, y_test

def build_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
    return models

def main():
    X_train, X_test, y_train, y_test = load_processed()
    models = build_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores_summary = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        scores_summary[name] = cv_scores.mean()
        print(f"{name}: CV accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # pick best by CV accuracy
    best_name = max(scores_summary, key=scores_summary.get)
    print("Best model by CV:", best_name)

    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    save_model(best_model, f"models/{best_name}.joblib")

    # Evaluate on test
    y_pred = best_model.predict(X_test)
    print("Test set classification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(cm, "outputs/confusion_matrix.joblib")
    print("Saved confusion matrix and model.")

if __name__ == "__main__":
    main()
