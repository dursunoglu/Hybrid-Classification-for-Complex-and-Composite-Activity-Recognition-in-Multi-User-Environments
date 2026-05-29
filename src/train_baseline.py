"""Train a baseline HAR model.

Examples:
    python src/train_baseline.py --data_dir data --task atomic_activity
    python src/train_baseline.py --data_dir data --task atomic_activity --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from config import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP, RANDOM_STATE
from dataset import load_raw_samples, load_session_splits, attach_split
from preprocessing import impute_missing, create_windows
from baseline_models import make_random_forest
from utils import ensure_dir, save_json


def make_quick_splits(raw: pd.DataFrame) -> pd.DataFrame:
    """Create temporary splits from whatever sessions are available.

    This is useful for smoke testing with preview_first_2000_rows.csv.
    For publication experiments, use the official session_splits.csv file.
    """
    sessions = sorted(raw["session_id"].unique())
    if len(sessions) < 3:
        # Fall back to same sessions for train/test in tiny smoke tests.
        return pd.DataFrame({
            "session_id": sessions,
            "split": ["train" for _ in sessions],
        })

    train_sessions, temp_sessions = train_test_split(
        sessions, test_size=0.30, random_state=RANDOM_STATE
    )
    val_sessions, test_sessions = train_test_split(
        temp_sessions, test_size=0.50, random_state=RANDOM_STATE
    )
    rows = []
    for s in train_sessions:
        rows.append({"session_id": s, "split": "train"})
    for s in val_sessions:
        rows.append({"session_id": s, "split": "validation"})
    for s in test_sessions:
        rows.append({"session_id": s, "split": "test"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--task", type=str, default="atomic_activity",
                        choices=["atomic_activity", "composite_activity", "interaction_type", "user_id"])
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=DEFAULT_WINDOW_STEP)
    parser.add_argument("--quick", action="store_true",
                        help="Create temporary splits from loaded data for smoke testing.")
    args = parser.parse_args()

    results_dir = ensure_dir(args.results_dir)

    print("Loading data...")
    raw = load_raw_samples(args.data_dir)

    if args.quick:
        splits = make_quick_splits(raw)
    else:
        splits = load_session_splits(args.data_dir)

    raw = attach_split(raw, splits)
    raw = impute_missing(raw)

    # If quick mode has only one session, create row-level splits after windowing.
    train_df = raw[raw["split"] == "train"].copy()
    val_df = raw[raw["split"] == "validation"].copy()
    test_df = raw[raw["split"] == "test"].copy()

    print("Creating windows...")

    if args.quick and (len(val_df) == 0 or len(test_df) == 0):
        X_all, y_all = create_windows(train_df, args.task, args.window_size, args.step)
        if len(X_all) < 3:
            raise ValueError("Not enough windows for quick smoke test. Use a larger preview or full dataset.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y_all, test_size=0.30, random_state=RANDOM_STATE, stratify=None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=None
        )
    else:
        X_train, y_train = create_windows(train_df, args.task, args.window_size, args.step)
        X_val, y_val = create_windows(val_df, args.task, args.window_size, args.step)
        X_test, y_test = create_windows(test_df, args.task, args.window_size, args.step)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"No windows found for train/test. Train windows={len(X_train)}, test windows={len(X_test)}. "
            "Use the full dataset or run with --quick for preview files."
        )

    id_cols = ["session_id", "user_id", "start_sample", "end_sample"]
    feature_cols = [c for c in X_train.columns if c not in id_cols]

    model = make_random_forest(random_state=RANDOM_STATE)
    print(f"Training Random Forest baseline for task: {args.task}")
    model.fit(X_train[feature_cols], y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test[feature_cols])

    metrics = {
        "task": args.task,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "num_train_windows": int(len(X_train)),
        "num_validation_windows": int(len(X_val)),
        "num_test_windows": int(len(X_test)),
        "window_size": args.window_size,
        "step": args.step,
        "quick_mode": bool(args.quick),
    }

    pred_df = X_test[id_cols].copy()
    pred_df["true_label"] = y_test.values
    pred_df["predicted_label"] = y_pred
    pred_path = results_dir / f"predictions_{args.task}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = results_dir / f"metrics_{args.task}.json"
    save_json(metrics, metrics_path)

    report_path = results_dir / f"classification_report_{args.task}.txt"
    report_path.write_text(classification_report(y_test, y_pred), encoding="utf-8")

    model_path = results_dir / f"rf_baseline_{args.task}.joblib"
    joblib.dump(model, model_path)

    print("Done.")
    print(metrics)
    print(f"Saved predictions to {pred_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
