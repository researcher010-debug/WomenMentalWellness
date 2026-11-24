
import os
import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from feature_selection import select_features

SEED = 42
TEST_SIZE = 0.30
DATA_DIR = Path("Data/source")
OUTPUT_DIR = Path("Data/preprocessed")
MODEL_DIR = Path("model")
DATASETS = {
    "women_depression": {"file": "women_depression.csv", "target": "target"},
    "maternal_depression": {"file": "maternal_depression.csv", "target": "target"}
}
N_SELECTED_FEATURES = None

def reset_random(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

class TargetEncoder:
    def __init__(self, smoothing=10, folds=5):
        self.smoothing = smoothing
        self.folds = folds

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.prior_ = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y)].mean(axis=0)
        self.mapping_ = {}
        for column in X.columns:
            mapping = {}
            for category in X[column].unique():
                mask = X[column] == category
                count = mask.sum()
                mean = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y[mask])].mean(axis=0)
                mapping[category] = (count * mean + self.smoothing * self.prior_) / (count + self.smoothing)
            self.mapping_[column] = mapping
        return self

    def transform(self, X):
        result = {}
        for column in X.columns:
            encoded = np.array([self.mapping_[column].get(value, self.prior_) for value in X[column]])
            for index, cls in enumerate(self.classes_):
                result[f"{column}_target_{cls}"] = encoded[:, index]
        return pd.DataFrame(result, index=X.index)

    def fit_transform(self, X, y):
        y = np.asarray(y)
        self.fit(X, y)
        result = pd.DataFrame(index=X.index)
        if X.empty:
            return result
        min_class = pd.Series(y).value_counts().min()
        n_splits = min(self.folds, min_class)
        if n_splits < 2:
            return self.transform(X)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for column in X.columns:
            encoded = np.zeros((len(X), len(self.classes_)))
            for train_idx, valid_idx in splitter.split(X, y):
                prior = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y[train_idx])].mean(axis=0)
                train_values = X.iloc[train_idx][column].values
                valid_values = X.iloc[valid_idx][column].values
                for category in np.unique(valid_values):
                    mask = train_values == category
                    count = mask.sum()
                    if count:
                        mean = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y[train_idx][mask])].mean(axis=0)
                        value = (count * mean + self.smoothing * prior) / (count + self.smoothing)
                    else:
                        value = prior
                    encoded[valid_idx[valid_values == category]] = value
            for index, cls in enumerate(self.classes_):
                result[f"{column}_target_{cls}"] = encoded[:, index]
        return result

def preprocess_dataset(dataset_name, file_name, target_column):
    csv_path = DATA_DIR / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].astype(str).copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    if pd.Series(y).value_counts().min() < 2:
        raise ValueError("Each class must contain at least two samples for stratified splitting.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=np.number).columns.tolist()
    numeric_imputer = IterativeImputer(max_iter=10, random_state=SEED, initial_strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    if numeric_columns:
        X_train_num = pd.DataFrame(numeric_imputer.fit_transform(X_train[numeric_columns]), columns=numeric_columns)
        X_test_num = pd.DataFrame(numeric_imputer.transform(X_test[numeric_columns]), columns=numeric_columns)
        q1, q3 = X_train_num.quantile(0.25), X_train_num.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X_train_num = X_train_num.clip(lower, upper, axis=1)
        X_test_num = X_test_num.clip(lower, upper, axis=1)
    else:
        X_train_num, X_test_num = pd.DataFrame(), pd.DataFrame()
        lower, upper = pd.Series(dtype=float), pd.Series(dtype=float)
    if categorical_columns:
        X_train_cat = pd.DataFrame(categorical_imputer.fit_transform(X_train[categorical_columns]), columns=categorical_columns)
        X_test_cat = pd.DataFrame(categorical_imputer.transform(X_test[categorical_columns]), columns=categorical_columns)
        target_encoder = TargetEncoder()
        X_train_encoded = target_encoder.fit_transform(X_train_cat, y_train)
        X_test_encoded = target_encoder.transform(X_test_cat)
    else:
        target_encoder = None
        X_train_cat, X_test_cat = pd.DataFrame(), pd.DataFrame()
        X_train_encoded, X_test_encoded = pd.DataFrame(), pd.DataFrame()
    X_train_processed = pd.concat([X_train_num, X_train_encoded], axis=1)
    X_test_processed = pd.concat([X_test_num, X_test_encoded], axis=1)
    if X_train_processed.empty:
        raise ValueError("No usable features remain after preprocessing.")
    scaler = RobustScaler()
    X_train_processed = pd.DataFrame(scaler.fit_transform(X_train_processed), columns=X_train_processed.columns)
    X_test_processed = pd.DataFrame(scaler.transform(X_test_processed), columns=X_train_processed.columns)
    X_train_selected, selected_features, ranking = select_features(X_train_processed, y_train, N_SELECTED_FEATURES)
    X_test_selected = X_test_processed[selected_features].copy()
    output_path = OUTPUT_DIR / dataset_name
    artifact_path = MODEL_DIR / dataset_name / "artifacts"
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    train_df = X_train_selected.copy()
    train_df["target"] = y_train
    test_df = X_test_selected.copy()
    test_df["target"] = y_test
    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    ranking.to_csv(output_path / "feature_ranking.csv", index=False)
    artifacts = {
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "iqr_lower": lower,
        "iqr_upper": upper,
        "selected_features": selected_features,
        "classes": label_encoder.classes_
    }
    joblib.dump(artifacts, artifact_path / "preprocessing.pkl")
    print(f"\n{dataset_name}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Classes: {label_encoder.classes_.tolist()}")
    print(f"Saved to: {output_path}")
    return train_df, test_df

def main():
    reset_random()
    for dataset_name, config in DATASETS.items():
        preprocess_dataset(dataset_name, config["file"], config["target"])

if __name__ == "__main__":
    main()
