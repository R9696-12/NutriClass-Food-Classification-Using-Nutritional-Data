
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import encode_labels, save_model, standardize, load_data

DATA_IN = r"C:\Nutri\data\synthetic_food_dataset_imbalanced.csv"
OUT_DIR = "data"

# Match dataset casing
NUMERIC_COLS = ["Calories","Protein","Carbs","Fat","Sugar","Fiber","Sodium"]
TARGET = "label"
RANDOM_STATE = 42

def cap_outliers_iqr(df, cols):
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        df[c] = df[c].clip(lower=low, upper=high)
    return df

def main():
    # Load dataset (utils handles fallback to src/ for relative paths)
    df = load_data(DATA_IN)
    print("Loaded", len(df), "rows")
    # Normalize column names to expected casing if needed
    lower_map = {c.lower(): c for c in df.columns}
    rename_map = {}
    for lc, proper in {
        "calories": "Calories",
        "protein": "Protein",
        "carbs": "Carbs",
        "fat": "Fat",
        "sugar": "Sugar",
        "fiber": "Fiber",
        "sodium": "Sodium",
    }.items():
        if lc in lower_map and lower_map[lc] != proper:
            rename_map[lower_map[lc]] = proper
    if rename_map:
        df = df.rename(columns=rename_map)
    # Basic cleaning
    df = df.drop_duplicates()
    # If label missing, derive simple rule-based labels
    if TARGET not in df.columns:
        cal = df["Calories"].fillna(0)
        prot = df["Protein"].fillna(0)
        carbs = df["Carbs"].fillna(0)
        fat = df["Fat"].fillna(0)
        sugar = df["Sugar"].fillna(0)
        conditions = [
            (prot > 15) & (cal > 150),
            (carbs > 30) & (sugar > 10),
            (fat > 20),
        ]
        choices = ["Protein-Rich", "Carbohydrate-Rich", "Fat-Rich"]
        df[TARGET] = np.select(conditions, choices, default="Low-Calorie")

    df = df.dropna(subset=NUMERIC_COLS + [TARGET])

    # convert types if needed
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=NUMERIC_COLS)

    df = cap_outliers_iqr(df, NUMERIC_COLS)

    X = df[NUMERIC_COLS]
    y = df[TARGET]

    # Encode labels
    y_enc, le = encode_labels(y)
    os.makedirs("models", exist_ok=True)
    save_model(le, os.path.join("models", "label_encoder.joblib"))

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )

    # Standardize
    X_train_s, X_test_s, scaler = standardize(X_train, X_test)
    save_model(scaler, os.path.join("models", "scaler.joblib"))

    # Save processed
    train = pd.DataFrame(X_train_s, columns=NUMERIC_COLS)
    train[TARGET] = y_train
    test = pd.DataFrame(X_test_s, columns=NUMERIC_COLS)
    test[TARGET] = y_test
    os.makedirs(OUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUT_DIR,"processed_train.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR,"processed_test.csv"), index=False)
    print("Saved processed_train.csv and processed_test.csv")

if __name__ == "__main__":
    main()
