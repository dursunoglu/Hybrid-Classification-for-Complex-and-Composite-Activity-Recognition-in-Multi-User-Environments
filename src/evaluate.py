"""Evaluate prediction files."""

from __future__ import annotations

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))
    print()
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
