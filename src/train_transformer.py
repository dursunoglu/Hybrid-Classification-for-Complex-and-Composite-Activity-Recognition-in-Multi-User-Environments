"""Train a Transformer baseline for HAR.

Examples:
    python src/train_transformer.py --data_dir data --task atomic_activity
    python src/train_transformer.py --data_dir data --task atomic_activity --quick --epochs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP, RANDOM_STATE
from dataset import load_raw_samples, load_session_splits, attach_split
from preprocessing import impute_missing
from sequence_dataset import create_sequence_windows
from utils import ensure_dir, save_json
from transformer_model import TransformerHAR


def set_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_quick_splits(raw: pd.DataFrame) -> pd.DataFrame:
    """Create temporary splits from available sessions."""
    sessions = sorted(raw["session_id"].unique())
    if len(sessions) < 3:
        return pd.DataFrame({"session_id": sessions, "split": ["train" for _ in sessions]})

    train_sessions, temp_sessions = train_test_split(
        sessions, test_size=0.30, random_state=RANDOM_STATE
    )
    val_sessions, test_sessions = train_test_split(
        temp_sessions, test_size=0.50, random_state=RANDOM_STATE
    )

    rows = []
    rows += [{"session_id": s, "split": "train"} for s in train_sessions]
    rows += [{"session_id": s, "split": "validation"} for s in val_sessions]
    rows += [{"session_id": s, "split": "test"} for s in test_sessions]
    return pd.DataFrame(rows)


def standardize_windows(X_train, X_val, X_test):
    """Standardize sequence windows channel-wise using train statistics."""
    n_train, seq_len, n_channels = X_train.shape

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_channels)
    scaler.fit(X_train_2d)

    def transform(X):
        if len(X) == 0:
            return X
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    labels = []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(pred.tolist())
        labels.extend(y.numpy().tolist())
    return np.array(labels), np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--task", type=str, default="atomic_activity",
                        choices=["atomic_activity", "composite_activity", "interaction_type", "user_id"])
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step", type=int, default=DEFAULT_WINDOW_STEP)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="Create temporary splits from loaded data for smoke testing.")
    args = parser.parse_args()

    set_seed()
    results_dir = ensure_dir(args.results_dir)

    print("Loading data...")
    raw = load_raw_samples(args.data_dir)
    splits = make_quick_splits(raw) if args.quick else load_session_splits(args.data_dir)
    raw = attach_split(raw, splits)
    raw = impute_missing(raw)

    train_df = raw[raw["split"] == "train"].copy()
    val_df = raw[raw["split"] == "validation"].copy()
    test_df = raw[raw["split"] == "test"].copy()

    print("Creating sequence windows...")
    if args.quick and (len(val_df) == 0 or len(test_df) == 0):
        X_all, y_all, meta_all = create_sequence_windows(train_df, args.task, args.window_size, args.step)
        if len(X_all) < 3:
            raise ValueError("Not enough windows for quick smoke test. Use a larger preview or full dataset.")

        idx = np.arange(len(X_all))
        train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_STATE)

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        meta_test = meta_all.iloc[test_idx].reset_index(drop=True)
    else:
        X_train, y_train, _ = create_sequence_windows(train_df, args.task, args.window_size, args.step)
        X_val, y_val, _ = create_sequence_windows(val_df, args.task, args.window_size, args.step)
        X_test, y_test, meta_test = create_sequence_windows(test_df, args.task, args.window_size, args.step)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"No windows found for train/test. Train={len(X_train)}, test={len(X_test)}. "
            "Use the full dataset or run with --quick for preview files."
        )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val) if len(y_val) else np.array([], dtype=int)
    y_test_enc = label_encoder.transform(y_test)

    X_train, X_val, X_test, scaler = standardize_windows(X_train, X_val, X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train_enc, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test_enc, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TransformerHAR(
        input_dim=X_train.shape[-1],
        num_classes=len(label_encoder.classes_),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Training Transformer baseline for task: {args.task}")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}/{args.epochs} - loss={loss:.4f}")

    print("Evaluating...")
    y_true_enc, y_pred_enc = predict(model, test_loader, device)

    y_true = label_encoder.inverse_transform(y_true_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    metrics = {
        "model": "TransformerHAR",
        "task": args.task,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "num_train_windows": int(len(X_train)),
        "num_validation_windows": int(len(X_val)),
        "num_test_windows": int(len(X_test)),
        "window_size": args.window_size,
        "step": args.step,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "quick_mode": bool(args.quick),
    }

    pred_df = meta_test.copy()
    pred_df["true_label"] = y_true
    pred_df["predicted_label"] = y_pred
    pred_path = results_dir / f"predictions_transformer_{args.task}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = results_dir / f"metrics_transformer_{args.task}.json"
    save_json(metrics, metrics_path)

    report_path = results_dir / f"classification_report_transformer_{args.task}.txt"
    report_path.write_text(classification_report(y_true, y_pred), encoding="utf-8")

    checkpoint_path = results_dir / f"transformer_{args.task}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_classes": label_encoder.classes_.tolist(),
            "metrics": metrics,
            "input_dim": int(X_train.shape[-1]),
            "args": vars(args),
        },
        checkpoint_path,
    )

    joblib.dump(scaler, results_dir / f"transformer_scaler_{args.task}.joblib")
    joblib.dump(label_encoder, results_dir / f"transformer_label_encoder_{args.task}.joblib")

    print("Done.")
    print(metrics)
    print(f"Saved predictions to {pred_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
