import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

DATA_PATH = r"C:\Nutri\src\data\synthetic_food_dataset_imbalanced.csv"

def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if not os.path.isabs(path):
        alt = os.path.join("src", path)
        if os.path.exists(alt):
            return alt
    return path

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(resolve_path(path))

def save_model(model, path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le

def standardize(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if X_test is None:
        return X_train_s, scaler
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler
