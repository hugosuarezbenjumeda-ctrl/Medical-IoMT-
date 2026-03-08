#!/usr/bin/env python3
"""Build static simulation data for the Vite React IDS replay UI."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from xgb_protocol_ids_utils import (
    discover_latest_hpo_run,
    global_feature_importance,
    load_feature_columns,
    load_thresholds_by_protocol,
    load_xgb_models_by_protocol,
    local_feature_contributions,
    prepare_feature_matrix,
    routed_predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to full_gpu_hpo_models_* directory (defaults to latest under reports).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/merged/metadata_test.csv",
        help="Path to metadata_test.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/ids_react_simulation/simulation_data.json",
        help="Output JSON for the React UI.",
    )
    parser.add_argument(
        "--window-rows",
        type=int,
        default=5000,
        help="Number of flows per simulation window.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=10,
        help="Simulated seconds represented by each window.",
    )
    parser.add_argument(
        "--max-alerts-per-window",
        type=int,
        default=8,
        help="Maximum number of sampled alert cards stored per window.",
    )
    parser.add_argument(
        "--max-local-features",
        type=int,
        default=10,
        help="Top local explanation features per sampled alert.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="If >0, cap number of windows for faster generation.",
    )
    parser.add_argument(
        "--replay-order",
        type=str,
        default="interleave-protocol-source",
        choices=["sequential", "shuffle", "interleave-source", "interleave-protocol-source"],
        help="How rows are ordered before windowing.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed used when replay order is shuffle/interleave-source.",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="balanced-protocol",
        choices=["top-score", "balanced-protocol"],
        help="How sampled alerts are selected within each window.",
    )
    return parser.parse_args()


def safe_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def interleave_row_groups(groups: List[pd.DataFrame], shuffle_seed: int) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame()
    rng = np.random.default_rng(shuffle_seed)
    order = np.arange(len(groups), dtype=int)
    rng.shuffle(order)
    groups = [groups[i] for i in order]

    max_len = max(len(g) for g in groups)
    out_parts = []
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                out_parts.append(g.iloc[[i]])
    return pd.concat(out_parts, ignore_index=True)


def interleave_by_source(df: pd.DataFrame, shuffle_seed: int) -> pd.DataFrame:
    if "source_relpath" not in df.columns:
        return df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    groups = [g.reset_index(drop=True) for _, g in df.groupby("source_relpath", sort=False)]
    if not groups:
        return df.reset_index(drop=True)
    return interleave_row_groups(groups, shuffle_seed)


def interleave_by_protocol_source(df: pd.DataFrame, shuffle_seed: int) -> pd.DataFrame:
    if "protocol_hint" not in df.columns:
        return interleave_by_source(df, shuffle_seed)

    protocol_streams = []
    for _, proto_df in df.groupby("protocol_hint", sort=False):
        protocol_streams.append(interleave_by_source(proto_df.reset_index(drop=True), shuffle_seed))
    if not protocol_streams:
        return df.reset_index(drop=True)
    return interleave_row_groups(protocol_streams, shuffle_seed)


def reorder_rows(df: pd.DataFrame, replay_order: str, shuffle_seed: int) -> pd.DataFrame:
    if df.empty or replay_order == "sequential":
        return df.reset_index(drop=True)

    if replay_order == "shuffle":
        return df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    if replay_order == "interleave-source":
        return interleave_by_source(df, shuffle_seed)
    if replay_order == "interleave-protocol-source":
        return interleave_by_protocol_source(df, shuffle_seed)
    return df.reset_index(drop=True)


def pick_sampled_alert_indices(
    scored: pd.DataFrame,
    alert_idx: np.ndarray,
    max_alerts_per_window: int,
    strategy: str,
) -> List[int]:
    limit = max(1, int(max_alerts_per_window))
    if len(alert_idx) == 0:
        return []

    if strategy == "top-score":
        scores = scored.loc[alert_idx, "score_attack"].to_numpy(np.float64, copy=False)
        order = np.argsort(scores)[::-1]
        return alert_idx[order][:limit].tolist()

    proto_buckets: Dict[str, List[int]] = defaultdict(list)
    for ridx in alert_idx.tolist():
        proto = safe_str(scored.iloc[ridx].get("protocol_used"))
        proto_buckets[proto].append(ridx)

    for proto, ridxs in proto_buckets.items():
        proto_buckets[proto] = sorted(
            ridxs,
            key=lambda i: safe_float(scored.iloc[i].get("score_attack")),
            reverse=True,
        )

    selected: List[int] = []
    protocol_order = sorted(proto_buckets.keys(), key=lambda p: (-len(proto_buckets[p]), p))
    while len(selected) < limit:
        progressed = False
        for proto in protocol_order:
            bucket = proto_buckets[proto]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def build_global_payload(
    run_dir: Path,
    models_by_protocol: Dict[str, object],
    feature_columns: List[str],
    thresholds_by_protocol: Dict[str, float],
) -> Dict[str, object]:
    explain_path = run_dir / "xgb_explainability" / "global_feature_importance.csv"
    if explain_path.exists():
        global_df = pd.read_csv(explain_path)
    else:
        rows = []
        for proto, model in models_by_protocol.items():
            imp = global_feature_importance(model, feature_columns)
            imp["protocol_hint"] = proto
            rows.append(imp)
        global_df = pd.concat(rows, ignore_index=True)

    protocol_payload = []
    feature_score = defaultdict(float)
    for proto in sorted(models_by_protocol.keys()):
        dfp = global_df[global_df["protocol_hint"].astype(str) == proto].copy()
        if dfp.empty:
            continue
        metric = "mean_abs_contribution" if "mean_abs_contribution" in dfp.columns else "gain"
        dfp = dfp.sort_values(metric, ascending=False).head(20).reset_index(drop=True)

        top_features = []
        n = len(dfp)
        for i, row in dfp.iterrows():
            fname = safe_str(row.get("feature"))
            points = float(max(1, n - i))
            feature_score[fname] += points
            top_features.append(
                {
                    "feature": fname,
                    "mean_abs_contribution": safe_float(row.get("mean_abs_contribution")),
                    "gain": safe_float(row.get("gain")),
                    "weight": safe_float(row.get("weight")),
                }
            )
        protocol_payload.append(
            {
                "protocol": proto,
                "threshold": safe_float(thresholds_by_protocol.get(proto, 0.5), 0.5),
                "top_features": top_features,
            }
        )

    overall_top = [
        {"feature": k, "score": float(v)}
        for k, v in sorted(feature_score.items(), key=lambda kv: kv[1], reverse=True)[:25]
    ]
    return {
        "protocols": protocol_payload,
        "overall_top_features": overall_top,
    }


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else discover_latest_hpo_run("reports")
    if run_dir is None:
        raise RuntimeError("No run directory found. Pass --run-dir explicitly.")
    run_dir = run_dir.resolve()
    test_csv = Path(args.test_csv).resolve()
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(run_dir)
    models_by_protocol = load_xgb_models_by_protocol(run_dir)
    thresholds_by_protocol = load_thresholds_by_protocol(run_dir)

    metadata_cols = ["protocol_hint", "attack_family", "source_relpath", "source_row_index"]
    header = pd.read_csv(test_csv, nrows=0).columns.astype(str).tolist()
    usecols = [c for c in metadata_cols if c in header] + [c for c in feature_columns if c in header]

    global_payload = build_global_payload(
        run_dir=run_dir,
        models_by_protocol=models_by_protocol,
        feature_columns=feature_columns,
        thresholds_by_protocol=thresholds_by_protocol,
    )

    full_df = pd.read_csv(test_csv, usecols=usecols)
    replay_df = reorder_rows(full_df, args.replay_order, args.shuffle_seed)

    windows = []
    total_rows = 0
    total_alerts = 0
    total_protocol_alerts = defaultdict(int)

    chunk_size = max(1, args.window_rows)
    total_chunks = (len(replay_df) + chunk_size - 1) // chunk_size
    for widx in range(total_chunks):
        if args.max_windows > 0 and widx >= args.max_windows:
            break

        start = widx * chunk_size
        end = min((widx + 1) * chunk_size, len(replay_df))
        chunk = replay_df.iloc[start:end].copy()
        chunk = chunk.reset_index(drop=True)
        if chunk.empty:
            continue
        x_chunk = prepare_feature_matrix(chunk.copy(), feature_columns)
        pred = routed_predict(
            df=chunk.copy(),
            feature_columns=feature_columns,
            models_by_protocol=models_by_protocol,
            thresholds_by_protocol=thresholds_by_protocol,
        )
        scored = pd.concat([chunk, pred], axis=1)
        scored["prediction_label"] = np.where(scored["prediction"] == 1, "attack", "benign")

        n_rows = int(len(scored))
        total_rows += n_rows

        alert_mask = scored["prediction"].to_numpy(np.int8, copy=False) == 1
        alert_idx = np.where(alert_mask)[0]
        n_alerts = int(len(alert_idx))
        total_alerts += n_alerts

        proto_counts = (
            scored["protocol_used"].astype(str).value_counts().to_dict()
            if "protocol_used" in scored.columns
            else {}
        )
        proto_alert_counts = (
            scored.loc[alert_mask, "protocol_used"].astype(str).value_counts().to_dict()
            if "protocol_used" in scored.columns and n_alerts > 0
            else {}
        )
        for k, v in proto_alert_counts.items():
            total_protocol_alerts[k] += int(v)

        sampled_alerts = []
        if n_alerts > 0:
            selected_idx = pick_sampled_alert_indices(
                scored=scored,
                alert_idx=alert_idx,
                max_alerts_per_window=args.max_alerts_per_window,
                strategy=args.sampling_strategy,
            )
            for ridx in selected_idx:
                row = scored.iloc[ridx]
                proto = safe_str(row.get("protocol_used"))
                booster = models_by_protocol.get(proto)
                if booster is None:
                    continue

                local_df = local_feature_contributions(
                    booster=booster,
                    row_features=x_chunk[ridx],
                    feature_columns=feature_columns,
                    top_n=max(1, args.max_local_features),
                )
                local_features = []
                for _, lr in local_df.iterrows():
                    local_features.append(
                        {
                            "feature": safe_str(lr.get("feature")),
                            "contribution": safe_float(lr.get("contribution")),
                            "abs_contribution": safe_float(lr.get("abs_contribution")),
                            "direction": safe_str(lr.get("direction")),
                        }
                    )

                sampled_alerts.append(
                    {
                        "id": f"w{widx:05d}_r{ridx:05d}",
                        "window_index": int(widx),
                        "row_in_window": int(ridx),
                        "global_row_index": int(widx * args.window_rows + ridx),
                        "protocol": proto,
                        "score_attack": safe_float(row.get("score_attack")),
                        "threshold": safe_float(row.get("threshold"), 0.5),
                        "attack_family": safe_str(row.get("attack_family")),
                        "source_relpath": safe_str(row.get("source_relpath")),
                        "source_row_index": safe_int(row.get("source_row_index")),
                        "local_explanation": local_features,
                    }
                )

        windows.append(
            {
                "window_index": int(widx),
                "sim_time_sec": int(widx * args.window_seconds),
                "start_row": int(widx * args.window_rows),
                "end_row_exclusive": int(widx * args.window_rows + n_rows),
                "flow_count": n_rows,
                "alert_count": n_alerts,
                "alert_ratio": float(n_alerts / max(1, n_rows)),
                "protocol_flow_counts": {str(k): int(v) for k, v in proto_counts.items()},
                "protocol_alert_counts": {str(k): int(v) for k, v in proto_alert_counts.items()},
                "sampled_alerts": sampled_alerts,
            }
        )

        print(
            f"[window {widx}] rows={n_rows} alerts={n_alerts} "
            f"sampled={len(sampled_alerts)} total_rows={total_rows}"
        )

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "test_csv": str(test_csv),
            "window_rows": int(args.window_rows),
            "window_seconds": int(args.window_seconds),
            "replay_order": str(args.replay_order),
            "shuffle_seed": int(args.shuffle_seed),
            "sampling_strategy": str(args.sampling_strategy),
            "max_alerts_per_window": int(args.max_alerts_per_window),
            "max_local_features": int(args.max_local_features),
            "total_rows": int(total_rows),
            "total_windows": int(len(windows)),
            "total_alerts": int(total_alerts),
            "total_alert_ratio": float(total_alerts / max(1, total_rows)),
            "total_protocol_alerts": {str(k): int(v) for k, v in sorted(total_protocol_alerts.items())},
        },
        "global_explanations": global_payload,
        "windows": windows,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)

    print(f"Wrote simulation data: {output_json}")


if __name__ == "__main__":
    main()
