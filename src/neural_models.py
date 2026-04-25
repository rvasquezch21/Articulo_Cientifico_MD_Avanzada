
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def _feature_target_split(df: pd.DataFrame, target_col: str):
    data = df.copy()
    data = data.drop_duplicates()
    y = data[target_col]
    X = data.drop(columns=[target_col])
    return X, y

def _preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )
    return pre

def _model_configs(max_iter: int):
    return {
        "MLP Base": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=42
        ),
        "Deep MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=42
        ),
        "MLP Dropout-like L2": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=42
        ),
        "MLP Tanh": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=42
        ),
        "MLP Early Stopping": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            max_iter=max_iter,
            random_state=42
        ),
    }

def run_neural_benchmark(df: pd.DataFrame, target_col: str, test_size: float = 0.2, max_iter: int = 300):
    X, y = _feature_target_split(df, target_col)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=42,
        stratify=y_enc
    )

    pre = _preprocessor(X_train)
    configs = _model_configs(max_iter)

    rows = []
    details = {}

    for name, clf in configs.items():
        pipe = Pipeline(steps=[
            ("preprocess", pre),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

        labels = label_encoder.classes_.tolist()
        report = classification_report(
            y_test,
            y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )

        details[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "labels": labels,
            "classification_report": pd.DataFrame(report).T.round(4)
        }

    results = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    return results, details
