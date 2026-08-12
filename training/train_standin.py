"""
Train the stand-in classifier for the crisis-NLP demo.

IMPORTANT CONTEXT
-----------------
This is NOT the model from "Harnessing Natural Language Processing for Disaster
Response and Crisis Management" (NHSJS 2025). That model's checkpoint was not
available. This script trains a substitute on a *public* dataset so the demo is
runnable and its explanations are real.

What it learns is also semantically different from the paper: the public labels
mark whether a tweet is *about a real disaster*, not how *urgent* it is. The
demo UI says so explicitly.

Model choice: TF-IDF + multinomial logistic regression. Two reasons.
  1. It echoes the TF-IDF/Naive-Bayes half of the pipeline the paper describes.
  2. It is linear, so SHAP values are exact and closed-form rather than
     sampled -- per-word attributions the demo highlights are the true
     contributions, not an approximation. It also fits any free tier.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# venetis/disaster_tweets mirrors the Kaggle "Real or Not? NLP with Disaster
# Tweets" training split: id, keyword, location, text, target.
HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/venetis/disaster_tweets"
    "/parquet/default/train/0.parquet"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Matches sklearn's default token_pattern. The serving code re-uses this exact
# pattern to map feature contributions back onto spans of the raw input, so the
# two must not drift apart.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

# target 1/0 in the public data. Deliberately NOT called "urgent"/"not urgent":
# the dataset does not encode urgency.
CLASS_NAMES = ["not disaster-related", "disaster-related"]


def load_dataset() -> pd.DataFrame:
    """Fetch the public dataset, caching a CSV copy under data/."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "disaster_tweets.csv"

    if cache.exists():
        print(f"Using cached dataset at {cache}")
        return pd.read_csv(cache)

    print(f"Downloading {HF_PARQUET_URL}")
    response = requests.get(HF_PARQUET_URL, timeout=120)
    response.raise_for_status()
    frame = pd.read_parquet(io.BytesIO(response.content))

    missing = {"text", "target"} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    frame = frame[["text", "target"]].dropna()
    frame["target"] = frame["target"].astype(int)
    frame.to_csv(cache, index=False)
    print(f"Cached {len(frame)} rows to {cache}")
    return frame


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def main() -> None:
    frame = load_dataset()
    print(f"Loaded {len(frame)} rows; positive rate {frame['target'].mean():.3f}")

    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"].tolist(),
        frame["target"].to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=frame["target"],
    )
    print(f"Train {len(x_train)} / test {len(x_test)}")

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    # Background expectation E[x] over the training set. Exact SHAP for a linear
    # model is phi_j = w_j * (x_j - E[x_j]), so serving needs this vector to
    # attribute contributions without re-reading the training data.
    train_matrix = pipeline.named_steps["tfidf"].transform(x_train)
    feature_means = np.asarray(train_matrix.mean(axis=0)).ravel()

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Whatever these come out to is what gets reported. The paper's numbers
    # (89.3% / F1 0.88) describe a different model on different data and must
    # never be copied into this file or the README.
    metrics = {
        "model": "TF-IDF (1-2gram) + LogisticRegression",
        "dataset": "venetis/disaster_tweets (public Kaggle mirror)",
        "task": "binary: disaster-related vs not disaster-related",
        "n_train": len(x_train),
        "n_test": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "f1_positive": round(float(f1_score(y_test, predictions, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "per_class": classification_report(
            y_test, predictions, target_names=CLASS_NAMES, output_dict=True
        ),
        "is_standin": True,
        "not_the_published_model": (
            "Trained on public data as a substitute. Does not reproduce, and is "
            "not comparable to, the NHSJS 2025 results."
        ),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "class_names": CLASS_NAMES,
            "token_pattern": TOKEN_PATTERN,
            "feature_means": feature_means,
            "metrics": metrics,
        },
        ARTIFACT_DIR / "standin_model.joblib",
    )
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n--- held-out test metrics (stand-in model) ---")
    print(f"accuracy    {metrics['accuracy']}")
    print(f"f1 (macro)  {metrics['f1_macro']}")
    print(f"f1 (pos)    {metrics['f1_positive']}")
    print(f"roc auc     {metrics['roc_auc']}")
    print(f"\nSaved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
