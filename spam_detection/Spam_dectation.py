import argparse
import os
import sys
from typing import List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = "spam_pipeline.joblib"
RANDOM_STATE = 42


def load_dataset(path: str) -> pd.DataFrame:
    """Load a TSV dataset with columns: label, message."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Many public copies of this dataset are tab‑separated with no header
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="latin-1")

    # Basic cleaning
    df = df.dropna(subset=["label", "message"]).reset_index(drop=True)

    # Normalize labels to 0/1
    df["label"] = df["label"].str.strip().str.lower()
    label_map = {"ham": 0, "spam": 1}
    if not set(df["label"]).issubset(set(label_map.keys())):
        bad = sorted(set(df["label"]) - set(label_map.keys()))
        raise ValueError(f"Unexpected labels found: {bad}. Expected only 'ham' and 'spam'.")
    df["target"] = df["label"].map(label_map).astype(int)

    return df[["message", "target"]]


def build_pipeline() -> Pipeline:
    """Create a text classification pipeline: TF‑IDF + MultinomialNB."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    return pipe


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    X = df["message"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== Evaluation =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Save trained pipeline
    joblib.dump(pipe, MODEL_PATH)
    print(f"\n✅ Saved model to: {MODEL_PATH}")

    return pipe


def load_model(path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. Train first with: python {os.path.basename(__file__)} --data sms.tsv --train"
        )
    return joblib.load(path)


def predict_texts(pipe: Pipeline, messages: List[str], threshold: float = 0.3):
    try:
        proba = pipe.predict_proba(messages)[:, 1]  # spam prob
    except Exception:
        proba = None
        preds = pipe.predict(messages)

    for i, msg in enumerate(messages):
        if proba is not None:
            spam_prob = proba[i]
            label = "SPAM" if spam_prob >= threshold else "HAM"
            print(f"{label} ({spam_prob:.2f}) → {msg}")
        else:
            label = "SPAM" if preds[i] == 1 else "HAM"
            print(f"{label} → {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spam SMS Classifier")
    parser.add_argument("--data", type=str, default="sms.tsv", help="Path to sms.tsv (label TAB message)")
    parser.add_argument("--train", action="store_true", help="Train & evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict using the saved model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Single message to classify")
    group.add_argument("--file", type=str, help="Path to a .txt file (one message per line)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        df = load_dataset(args.data)
        print(f"Loaded dataset: {args.data} → {df.shape[0]} rows")
        class_counts = df["target"].value_counts().sort_index()
        print(f"Class counts (ham=0, spam=1):\n{class_counts}")
        train_and_evaluate(df)

    if args.predict:
        pipe = load_model(MODEL_PATH)
        messages: List[str] = []
        if args.text:
            messages = [args.text]
        elif args.file:
            if not os.path.exists(args.file):
                print(f"File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
                messages = [line.strip() for line in f if line.strip()]
        else:
            # Read from STDIN (handy for piping)
            print("Enter messages (Ctrl+D/Ctrl+Z to end):", file=sys.stderr)
            for line in sys.stdin:
                line = line.strip()
                if line:
                    messages.append(line)
        if not messages:
            print("No messages provided for prediction.", file=sys.stderr)
            sys.exit(1)
        predict_texts(pipe, messages)

    if not args.train and not args.predict:
        # Default help if no action chosen
        print(__doc__)


if __name__ == "__main__":
    main()
