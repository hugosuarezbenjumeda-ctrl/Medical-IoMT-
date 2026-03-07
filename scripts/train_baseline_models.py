#!/usr/bin/env python3
"""Train baseline IDS models for CICIoMT2024 merged flow features.

Models:
- Logistic Regression
- Random Forest
- HistGradientBoosting
- Simple MLP (1 hidden layer)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stable_hash_u64(text: str) -> int:
    h = 1469598103934665603
    for ch in text:
        h ^= ord(ch)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


@dataclass
class Reservoir:
    capacity: int
    rng: np.random.Generator
    seen: int = 0
    rows: List[Tuple[np.ndarray, int, str, str]] | None = None

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = []

    def add(self, row: Tuple[np.ndarray, int, str, str]) -> None:
        self.seen += 1
        if len(self.rows) < self.capacity:
            self.rows.append(row)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.capacity:
            self.rows[j] = row


def parse_float(v: str) -> float:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return 0.0


def load_feature_columns(path: str) -> List[str]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    return header[20:]


def build_train_val_samples(
    train_csv: str,
    feature_cols: Sequence[str],
    seed: int,
    train_cap_per_class: int,
    val_cap_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    res_train = {
        0: Reservoir(train_cap_per_class, np.random.default_rng(seed + 11)),
        1: Reservoir(train_cap_per_class, np.random.default_rng(seed + 12)),
    }
    res_val = {
        0: Reservoir(val_cap_per_class, np.random.default_rng(seed + 21)),
        1: Reservoir(val_cap_per_class, np.random.default_rng(seed + 22)),
    }
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y = int(row["label"])
            x = np.array([parse_float(row[c]) for c in feature_cols], dtype=np.float32)
            protocol = row.get("protocol_hint", "unknown")
            family = row.get("attack_family", "unknown")
            k = f"{row.get('source_relpath','')}#{row.get('source_row_index','0')}#{seed}"
            if stable_hash_u64(k) % 5 == 0:
                res_val[y].add((x, y, protocol, family))
            else:
                res_train[y].add((x, y, protocol, family))

    rows_train = (res_train[0].rows or []) + (res_train[1].rows or [])
    rows_val = (res_val[0].rows or []) + (res_val[1].rows or [])
    np.random.default_rng(seed + 31).shuffle(rows_train)
    np.random.default_rng(seed + 32).shuffle(rows_val)
    x_train = np.vstack([r[0] for r in rows_train])
    y_train = np.array([r[1] for r in rows_train], dtype=np.int32)
    x_val = np.vstack([r[0] for r in rows_val])
    y_val = np.array([r[1] for r in rows_val], dtype=np.int32)
    return x_train, y_train, x_val, y_val


def build_test_samples(
    test_csv: str,
    feature_cols: Sequence[str],
    seed: int,
    test_cap_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    res = {
        0: Reservoir(test_cap_per_class, np.random.default_rng(seed + 41)),
        1: Reservoir(test_cap_per_class, np.random.default_rng(seed + 42)),
    }
    with open(test_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y = int(row["label"])
            x = np.array([parse_float(row[c]) for c in feature_cols], dtype=np.float32)
            protocol = row.get("protocol_hint", "unknown")
            family = row.get("attack_family", "unknown")
            res[y].add((x, y, protocol, family))
    rows = (res[0].rows or []) + (res[1].rows or [])
    np.random.default_rng(seed + 43).shuffle(rows)
    x = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.int32)
    protocol = np.array([r[2] for r in rows], dtype=object)
    family = np.array([r[3] for r in rows], dtype=object)
    return x, y, protocol, family


def select_threshold_recall_at_fpr(y_true: np.ndarray, proba: np.ndarray, max_fpr: float = 0.01) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, proba)
    valid = np.where(fpr <= max_fpr)[0]
    if len(valid) == 0:
        return {"threshold": 1.0, "val_recall": 0.0, "val_fpr": 0.0}
    best_idx = valid[np.argmax(tpr[valid])]
    return {
        "threshold": float(thr[best_idx]),
        "val_recall": float(tpr[best_idx]),
        "val_fpr": float(fpr[best_idx]),
    }


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def evaluate_by_slice(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    slice_values: np.ndarray,
    slice_name: str,
    model_name: str,
) -> List[Dict[str, float | str | int]]:
    out = []
    for val in sorted(set(slice_values.tolist())):
        m = slice_values == val
        if int(m.sum()) == 0:
            continue
        metrics = evaluate(y_true[m], proba[m], threshold)
        metrics.update({"model": model_name, slice_name: val, "n": int(m.sum())})
        out.append(metrics)
    return out


def save_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_train.csv")
    ap.add_argument("--test-csv", default="/home/capstone15/data/ciciomt2024/merged/metadata_test.csv")
    ap.add_argument("--output-dir", default="/home/capstone15/reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-cap-per-class", type=int, default=50000)
    ap.add_argument("--val-cap-per-class", type=int, default=15000)
    ap.add_argument("--test-cap-per-class", type=int, default=30000)
    args = ap.parse_args()

    feature_cols = load_feature_columns(args.train_csv)
    x_train, y_train, x_val, y_val = build_train_val_samples(
        args.train_csv,
        feature_cols,
        args.seed,
        args.train_cap_per_class,
        args.val_cap_per_class,
    )
    x_test, y_test, test_protocol, test_family = build_test_samples(
        args.test_csv,
        feature_cols,
        args.seed,
        args.test_cap_per_class,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"baseline_models_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=args.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=args.seed,
            max_depth=8,
            learning_rate=0.08,
            max_iter=250,
        ),
        "mlp_baseline": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.1,
                        max_iter=40,
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
    }

    summary_rows: List[Dict[str, object]] = []
    protocol_rows: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []
    full = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]
        thr_info = select_threshold_recall_at_fpr(y_val, p_val, max_fpr=0.01)
        test_m = evaluate(y_test, p_test, thr_info["threshold"])
        test_m.update(
            {
                "model": name,
                "threshold": thr_info["threshold"],
                "val_recall_at_fpr_le_1pct": thr_info["val_recall"],
                "val_fpr_at_selected_threshold": thr_info["val_fpr"],
            }
        )
        summary_rows.append(test_m)

        p_rows = evaluate_by_slice(y_test, p_test, thr_info["threshold"], test_protocol, "protocol_hint", name)
        f_rows = evaluate_by_slice(y_test, p_test, thr_info["threshold"], test_family, "attack_family", name)
        protocol_rows.extend(p_rows)
        family_rows.extend(f_rows)

        full[name] = {
            "global_test": test_m,
            "protocol_slices": p_rows,
            "attack_family_slices": f_rows,
        }
        joblib.dump(model, model_dir / f"{name}.joblib")

    summary_rows.sort(key=lambda r: float(r["f1"]), reverse=True)
    best_model = summary_rows[0]["model"] if summary_rows else None

    save_csv(
        out_dir / "metrics_summary.csv",
        summary_rows,
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "fpr",
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
            "val_recall_at_fpr_le_1pct",
            "val_fpr_at_selected_threshold",
        ],
    )
    save_csv(
        out_dir / "slice_metrics_protocol.csv",
        protocol_rows,
        ["model", "protocol_hint", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"],
    )
    save_csv(
        out_dir / "slice_metrics_attack_family.csv",
        family_rows,
        ["model", "attack_family", "n", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"],
    )

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "seed": args.seed,
                "feature_columns": feature_cols,
                "sample_sizes": {
                    "train": int(len(y_train)),
                    "val": int(len(y_val)),
                    "test": int(len(y_test)),
                    "train_pos_ratio": float(y_train.mean()),
                    "val_pos_ratio": float(y_val.mean()),
                    "test_pos_ratio": float(y_test.mean()),
                },
                "best_model_by_f1": best_model,
                "results": full,
            },
            f,
            indent=2,
        )

    with (out_dir / "RUN_SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("Baseline model training completed.\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Best model by F1: {best_model}\n")
        f.write(f"Train/Val/Test samples: {len(y_train)}/{len(y_val)}/{len(y_test)}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")

    print(out_dir)


if __name__ == "__main__":
    main()
