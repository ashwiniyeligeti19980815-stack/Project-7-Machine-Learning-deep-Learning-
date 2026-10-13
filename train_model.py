import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


DEFAULT_DATASET = "twitter_disaster (1) (1).xlsx"
DEFAULT_MODEL = "model.joblib"


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()


def train_model(dataset_path=DEFAULT_DATASET, model_path=DEFAULT_MODEL):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {dataset_path}. Provide the labeled dataset "
            "with 'text' and 'target' columns."
        )

    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(dataset_path)
    else:
        data = pd.read_csv(dataset_path)

    missing_columns = {"text", "target"} - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    data = data[["text", "target"]].dropna()
    if data.empty:
        raise ValueError("The dataset has no rows with both text and target values.")

    target_values = pd.to_numeric(data["target"], errors="coerce")
    if target_values.isna().any() or set(target_values.unique()) != {0, 1}:
        raise ValueError("Expected exactly two target classes: 0 (non-disaster) and 1 (disaster).")

    targets = target_values.astype(int)
    class_names = {"0": "Non-disaster", "1": "Disaster"}

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=2,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=30000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    features = vectorizer.fit_transform(data["text"].map(clean_text))
    model = LogisticRegression(C=2.0, max_iter=1500, random_state=42)
    model.fit(features, targets)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "class_names": class_names},
        model_path,
    )
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train and serialize the disaster tweet classifier.")
    parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET, help="CSV or Excel dataset path")
    parser.add_argument("--output", default=DEFAULT_MODEL, help="Output model artifact path")
    arguments = parser.parse_args()
    artifact_path = train_model(arguments.dataset, arguments.output)
    print(f"Saved trained model to {artifact_path}")


if __name__ == "__main__":
    main()
    