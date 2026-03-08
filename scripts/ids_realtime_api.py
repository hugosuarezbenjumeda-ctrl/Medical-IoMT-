#!/usr/bin/env python3
"""Local realtime IDS API for the MIoT IDS Prototype React UI."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from generate_ids_react_simulation_data import build_global_payload, reorder_rows
from xgb_protocol_ids_utils import (
    discover_latest_hpo_run,
    load_feature_columns,
    load_thresholds_by_protocol,
    load_xgb_models_by_protocol,
    local_feature_contributions,
    prepare_feature_matrix,
    routed_predict,
)


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def safe_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v)


class LiveIDSRuntime:
    def __init__(
        self,
        run_dir: Path,
        test_csv: Path,
        rows_per_second: int,
        local_top_n: int,
        replay_order: str,
        shuffle_seed: int,
        max_recent_alerts: int,
    ) -> None:
        self.run_dir = run_dir
        self.test_csv = test_csv
        self.rows_per_second = max(1, int(rows_per_second))
        self.local_top_n = max(1, int(local_top_n))
        self.replay_order = replay_order
        self.shuffle_seed = int(shuffle_seed)
        self.max_recent_alerts = max(1, int(max_recent_alerts))

        self.lock = threading.Lock()

        self.feature_columns = load_feature_columns(run_dir)
        self.models_by_protocol = load_xgb_models_by_protocol(run_dir)
        self.thresholds_by_protocol = load_thresholds_by_protocol(run_dir)
        self.global_explanations = build_global_payload(
            run_dir=run_dir,
            models_by_protocol=self.models_by_protocol,
            feature_columns=self.feature_columns,
            thresholds_by_protocol=self.thresholds_by_protocol,
        )

        metadata_cols = ["protocol_hint", "attack_family", "source_relpath", "source_row_index"]
        header = pd.read_csv(test_csv, nrows=0).columns.astype(str).tolist()
        usecols = [c for c in metadata_cols if c in header] + [c for c in self.feature_columns if c in header]
        full_df = pd.read_csv(test_csv, usecols=usecols)
        self.df = reorder_rows(full_df, replay_order=self.replay_order, shuffle_seed=self.shuffle_seed)
        self.total_rows = int(len(self.df))

        self._reset_locked()

    def _reset_locked(self) -> None:
        self.running = False
        self.ended = False
        self.cursor = 0
        self.sim_seconds = 0
        self.flows_processed = 0
        self.alerts_detected = 0
        self.alerts_surfaced = 0
        self.first_alert_sim_second = None
        self.first_alert_row = None
        self.recent_alerts: List[Dict[str, object]] = []
        self.protocol_flow_counts = defaultdict(int)
        self.protocol_alert_counts = defaultdict(int)
        self.last_update = time.monotonic()

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def start(self) -> None:
        with self.lock:
            if not self.ended:
                self.running = True
                self.last_update = time.monotonic()

    def pause(self) -> None:
        with self.lock:
            self._advance_locked()
            self.running = False

    def _build_alert_payload(
        self,
        row: pd.Series,
        x_row: np.ndarray,
        row_index: int,
    ) -> Dict[str, object]:
        proto = safe_str(row.get("protocol_used"))
        booster = self.models_by_protocol.get(proto)
        local_features = []
        if booster is not None:
            local_df = local_feature_contributions(
                booster=booster,
                row_features=x_row,
                feature_columns=self.feature_columns,
                top_n=self.local_top_n,
            )
            for _, lr in local_df.iterrows():
                local_features.append(
                    {
                        "feature": safe_str(lr.get("feature")),
                        "contribution": safe_float(lr.get("contribution")),
                        "abs_contribution": safe_float(lr.get("abs_contribution")),
                        "direction": safe_str(lr.get("direction")),
                    }
                )

        return {
            "id": f"row_{row_index:07d}",
            "global_row_index": row_index,
            "protocol": proto,
            "score_attack": safe_float(row.get("score_attack")),
            "threshold": safe_float(row.get("threshold"), 0.5),
            "attack_family": safe_str(row.get("attack_family")),
            "source_relpath": safe_str(row.get("source_relpath")),
            "source_row_index": safe_int(row.get("source_row_index")),
            "sim_second": int(self.sim_seconds),
            "local_explanation": local_features,
        }

    def _advance_one_second_locked(self) -> None:
        if self.ended:
            return

        start = self.cursor
        end = min(self.total_rows, start + self.rows_per_second)
        if start >= end:
            self.ended = True
            self.running = False
            return

        chunk = self.df.iloc[start:end].copy().reset_index(drop=True)
        x_chunk = prepare_feature_matrix(chunk.copy(), self.feature_columns)
        pred = routed_predict(
            df=chunk.copy(),
            feature_columns=self.feature_columns,
            models_by_protocol=self.models_by_protocol,
            thresholds_by_protocol=self.thresholds_by_protocol,
        )
        scored = pd.concat([chunk, pred], axis=1)

        second_alerts: List[Dict[str, object]] = []
        for i in range(len(scored)):
            row = scored.iloc[i]
            proto = safe_str(row.get("protocol_used"))
            self.protocol_flow_counts[proto] += 1
            if safe_int(row.get("prediction")) == 1:
                self.alerts_detected += 1
                self.alerts_surfaced += 1
                self.protocol_alert_counts[proto] += 1
                row_index = start + i
                if self.first_alert_sim_second is None:
                    self.first_alert_sim_second = int(self.sim_seconds + 1)
                    self.first_alert_row = int(row_index)
                second_alerts.append(self._build_alert_payload(row=row, x_row=x_chunk[i], row_index=row_index))

        if second_alerts:
            self.recent_alerts = (second_alerts + self.recent_alerts)[: self.max_recent_alerts]

        self.cursor = end
        self.flows_processed = end
        self.sim_seconds += 1
        if self.cursor >= self.total_rows:
            self.ended = True
            self.running = False

    def _advance_locked(self) -> None:
        if not self.running or self.ended:
            self.last_update = time.monotonic()
            return

        now = time.monotonic()
        elapsed = now - self.last_update
        ticks = int(elapsed)
        if ticks <= 0:
            return

        ticks = min(ticks, 60)
        for _ in range(ticks):
            self._advance_one_second_locked()
            if self.ended:
                break
        self.last_update += ticks

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            self._advance_locked()
            return {
                "running": bool(self.running),
                "ended": bool(self.ended),
                "sim_seconds": int(self.sim_seconds),
                "cursor": int(self.cursor),
                "flows_processed": int(self.flows_processed),
                "alerts_detected": int(self.alerts_detected),
                "alerts_surfaced": int(self.alerts_surfaced),
                "first_alert_sim_second": self.first_alert_sim_second,
                "first_alert_row": self.first_alert_row,
                "recent_alerts": self.recent_alerts,
                "protocol_flow_counts": {k: int(v) for k, v in sorted(self.protocol_flow_counts.items())},
                "protocol_alert_counts": {k: int(v) for k, v in sorted(self.protocol_alert_counts.items())},
            }

    def init_payload(self) -> Dict[str, object]:
        snap = self.snapshot()
        return {
            "meta": {
                "run_dir": str(self.run_dir),
                "test_csv": str(self.test_csv),
                "rows_per_second": int(self.rows_per_second),
                "replay_order": self.replay_order,
                "shuffle_seed": int(self.shuffle_seed),
                "local_top_n": int(self.local_top_n),
                "total_rows": int(self.total_rows),
                "thresholds_by_protocol": {
                    str(k): float(v) for k, v in sorted(self.thresholds_by_protocol.items())
                },
            },
            "global_explanations": self.global_explanations,
            "state": snap,
        }


class IDSRequestHandler(BaseHTTPRequestHandler):
    runtime: LiveIDSRuntime | None = None

    def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _send_not_found(self) -> None:
        self._send_json({"error": "Not found"}, status=404)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        runtime = IDSRequestHandler.runtime
        if runtime is None:
            self._send_json({"error": "Runtime not initialized"}, status=500)
            return

        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/api/init":
            self._send_json(runtime.init_payload())
            return
        if path == "/api/state":
            self._send_json({"state": runtime.snapshot()})
            return
        self._send_not_found()

    def do_POST(self) -> None:  # noqa: N802
        runtime = IDSRequestHandler.runtime
        if runtime is None:
            self._send_json({"error": "Runtime not initialized"}, status=500)
            return

        path = urlparse(self.path).path
        if path == "/api/start":
            runtime.start()
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/pause":
            runtime.pause()
            self._send_json({"state": runtime.snapshot()})
            return
        if path == "/api/reset":
            runtime.reset()
            self._send_json({"state": runtime.snapshot()})
            return

        self._send_not_found()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
        help="Path to metadata_test.csv",
    )
    parser.add_argument(
        "--rows-per-second",
        type=int,
        default=1,
        help="How many CSV rows to score per simulated second.",
    )
    parser.add_argument(
        "--local-top-n",
        type=int,
        default=8,
        help="Top local explanation features for each surfaced alert.",
    )
    parser.add_argument(
        "--replay-order",
        type=str,
        default="shuffle",
        choices=["sequential", "shuffle", "interleave-source", "interleave-protocol-source"],
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--max-recent-alerts", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else discover_latest_hpo_run("reports")
    if run_dir is None:
        raise RuntimeError("No run directory found. Pass --run-dir explicitly.")
    run_dir = run_dir.resolve()
    test_csv = Path(args.test_csv).resolve()
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    runtime = LiveIDSRuntime(
        run_dir=run_dir,
        test_csv=test_csv,
        rows_per_second=args.rows_per_second,
        local_top_n=args.local_top_n,
        replay_order=args.replay_order,
        shuffle_seed=args.shuffle_seed,
        max_recent_alerts=args.max_recent_alerts,
    )
    IDSRequestHandler.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), IDSRequestHandler)
    print(f"MIoT IDS realtime API listening at http://{args.host}:{args.port}")
    print(f"Rows loaded: {runtime.total_rows} | rows_per_second={runtime.rows_per_second}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
