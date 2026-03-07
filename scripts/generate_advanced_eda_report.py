#!/usr/bin/env python3
"""Dependency-free advanced EDA report generator for CICIoMT merged CSVs."""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METADATA_COLUMNS = [
    "source_relpath",
    "source_filename",
    "source_modality",
    "source_group",
    "source_split_folder",
    "assigned_split",
    "split_strategy",
    "sample_name",
    "sample_core_name",
    "protocol_scope",
    "protocol_hint",
    "device",
    "scenario",
    "attack_name",
    "attack_family",
    "attack_variant",
    "is_attack",
    "is_benign",
    "label",
    "source_row_index",
]

DEFAULT_SAMPLE_RATE = 0.006
DEFAULT_RANDOM_SEED = 20260305


@dataclass
class Paths:
    train_csv: Path
    test_csv: Path
    output_dir: Path
    tables_dir: Path
    plots_dir: Path


def ensure_dirs(paths: Paths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)


def safe_float(raw: str) -> Tuple[float, bool]:
    text = (raw or "").strip()
    if text == "":
        return 0.0, True
    try:
        return float(text), False
    except ValueError:
        return 0.0, True


def maybe_int(raw: str, default: int = 0) -> int:
    text = (raw or "").strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def pooled_std(vals0: Sequence[float], vals1: Sequence[float]) -> float:
    n0, n1 = len(vals0), len(vals1)
    if n0 < 2 or n1 < 2:
        return 0.0
    v0 = statistics.pvariance(vals0)
    v1 = statistics.pvariance(vals1)
    num = ((n0 - 1) * v0) + ((n1 - 1) * v1)
    den = (n0 + n1 - 2)
    if den <= 0:
        return 0.0
    return math.sqrt(num / den)


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return (1 - w) * sorted_vals[lo] + w * sorted_vals[hi]


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return 0.0
    return num / den


def save_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">{body}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def axis_and_title(
    title: str,
    width: int,
    height: int,
    margin_l: int,
    margin_r: int,
    margin_t: int,
    margin_b: int,
) -> str:
    x0 = margin_l
    y0 = height - margin_b
    x1 = width - margin_r
    y1 = margin_t
    return (
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>'
        f'<text x="{width//2}" y="26" text-anchor="middle" font-size="16" '
        f'font-family="sans-serif" font-weight="600">{escape(title)}</text>'
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" stroke-width="1"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" stroke-width="1"/>'
    )


def save_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    color: str = "#2a9d8f",
    rotate_labels: bool = False,
) -> None:
    width, height = 1120, 680
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 170 if rotate_labels else 120
    body = axis_and_title(title, width, height, margin_l, margin_r, margin_t, margin_b)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0

    n = max(1, len(values))
    bar_w = plot_w / n * 0.72
    step = plot_w / n

    for i, (label, v) in enumerate(zip(labels, values)):
        h = (v / max_v) * plot_h
        x = margin_l + i * step + (step - bar_w) / 2
        y = height - margin_b - h
        body += f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>'
        body += (
            f'<text x="{x + bar_w/2:.2f}" y="{y - 6:.2f}" text-anchor="middle" '
            f'font-size="10" font-family="sans-serif">{v:.0f}</text>'
        )
        lx = x + bar_w / 2
        ly = height - margin_b + 16
        if rotate_labels:
            body += (
                f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="end" '
                f'font-size="10" font-family="sans-serif" transform="rotate(-45 {lx:.2f} {ly:.2f})">'
                f'{escape(label)}</text>'
            )
        else:
            body += (
                f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" '
                f'font-size="10" font-family="sans-serif">{escape(label)}</text>'
            )

    for i in range(6):
        frac = i / 5
        y = height - margin_b - frac * plot_h
        val = frac * max_v
        body += f'<line x1="{margin_l}" y1="{y:.2f}" x2="{width-margin_r}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>'
        body += (
            f'<text x="{margin_l-8}" y="{y+4:.2f}" text-anchor="end" '
            f'font-size="10" font-family="sans-serif">{val:.0f}</text>'
        )

    save_svg(path, width, height, body)


def save_stacked_bar_chart(
    path: Path,
    title: str,
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    colors: Dict[str, str],
    rotate_labels: bool = False,
) -> None:
    width, height = 1120, 700
    margin_l, margin_r, margin_t, margin_b = 90, 60, 70, 180 if rotate_labels else 130
    body = axis_and_title(title, width, height, margin_l, margin_r, margin_t, margin_b)

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    totals = [0.0] * len(categories)
    for svals in series.values():
        for i, v in enumerate(svals):
            totals[i] += v
    max_v = max(totals) if totals else 1.0
    if max_v <= 0:
        max_v = 1.0

    n = max(1, len(categories))
    step = plot_w / n
    bar_w = step * 0.64

    for i, cat in enumerate(categories):
        x = margin_l + i * step + (step - bar_w) / 2
        y_cursor = height - margin_b
        for sname, svals in series.items():
            v = svals[i]
            h = (v / max_v) * plot_h
            y = y_cursor - h
            body += (
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
                f'fill="{colors.get(sname, "#999")}"/>'
            )
            y_cursor = y

        lx = x + bar_w / 2
        ly = height - margin_b + 16
        if rotate_labels:
            body += (
                f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="end" '
                f'font-size="10" font-family="sans-serif" transform="rotate(-45 {lx:.2f} {ly:.2f})">'
                f'{escape(cat)}</text>'
            )
        else:
            body += (
                f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" '
                f'font-size="10" font-family="sans-serif">{escape(cat)}</text>'
            )

    for i in range(6):
        frac = i / 5
        y = height - margin_b - frac * plot_h
        val = frac * max_v
        body += f'<line x1="{margin_l}" y1="{y:.2f}" x2="{width-margin_r}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>'
        body += (
            f'<text x="{margin_l-8}" y="{y+4:.2f}" text-anchor="end" '
            f'font-size="10" font-family="sans-serif">{val:.0f}</text>'
        )

    legend_x = width - margin_r - 180
    legend_y = margin_t + 8
    y_off = 0
    for sname in series.keys():
        body += (
            f'<rect x="{legend_x}" y="{legend_y + y_off}" width="14" height="14" '
            f'fill="{colors.get(sname, "#999")}"/>'
        )
        body += (
            f'<text x="{legend_x + 20}" y="{legend_y + y_off + 12}" '
            f'font-size="11" font-family="sans-serif">{escape(sname)}</text>'
        )
        y_off += 20

    save_svg(path, width, height, body)


def save_hist_overlay(
    path: Path,
    title: str,
    vals_benign: Sequence[float],
    vals_attack: Sequence[float],
    bins: int = 36,
) -> None:
    width, height = 1120, 680
    margin_l, margin_r, margin_t, margin_b = 90, 60, 70, 120
    body = axis_and_title(title, width, height, margin_l, margin_r, margin_t, margin_b)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    if not vals_benign and not vals_attack:
        save_svg(path, width, height, body)
        return

    all_vals = list(vals_benign) + list(vals_attack)
    lo = min(all_vals)
    hi = max(all_vals)
    if lo == hi:
        hi = lo + 1.0
    bw = (hi - lo) / bins

    hist_b = [0] * bins
    hist_a = [0] * bins

    for v in vals_benign:
        idx = min(bins - 1, max(0, int((v - lo) / bw)))
        hist_b[idx] += 1
    for v in vals_attack:
        idx = min(bins - 1, max(0, int((v - lo) / bw)))
        hist_a[idx] += 1

    max_count = max(hist_b + hist_a) if (hist_b or hist_a) else 1
    if max_count <= 0:
        max_count = 1

    bar_w = plot_w / bins
    for i in range(bins):
        h_b = (hist_b[i] / max_count) * plot_h
        h_a = (hist_a[i] / max_count) * plot_h
        x = margin_l + i * bar_w
        yb = height - margin_b - h_b
        ya = height - margin_b - h_a
        body += f'<rect x="{x:.2f}" y="{yb:.2f}" width="{bar_w:.2f}" height="{h_b:.2f}" fill="#4e79a7" fill-opacity="0.45"/>'
        body += f'<rect x="{x:.2f}" y="{ya:.2f}" width="{bar_w:.2f}" height="{h_a:.2f}" fill="#e15759" fill-opacity="0.45"/>'

    for i in range(6):
        frac = i / 5
        y = height - margin_b - frac * plot_h
        val = frac * max_count
        body += f'<line x1="{margin_l}" y1="{y:.2f}" x2="{width-margin_r}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>'
        body += f'<text x="{margin_l-8}" y="{y+4:.2f}" text-anchor="end" font-size="10" font-family="sans-serif">{val:.0f}</text>'

    body += (
        f'<text x="{margin_l}" y="{height-28}" font-size="11" font-family="sans-serif">'
        f'Range: {lo:.3f} to {hi:.3f}</text>'
    )
    legend_x = width - margin_r - 180
    legend_y = margin_t + 8
    body += f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="#4e79a7" fill-opacity="0.55"/>'
    body += f'<text x="{legend_x+20}" y="{legend_y+12}" font-size="11" font-family="sans-serif">Benign</text>'
    body += f'<rect x="{legend_x}" y="{legend_y+22}" width="14" height="14" fill="#e15759" fill-opacity="0.55"/>'
    body += f'<text x="{legend_x+20}" y="{legend_y+34}" font-size="11" font-family="sans-serif">Attack</text>'

    save_svg(path, width, height, body)


def main() -> None:
    random.seed(DEFAULT_RANDOM_SEED)

    base = Path("/home/capstone15/data/ciciomt2024/merged")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(f"/home/capstone15/reports/eda_advanced_{now}")

    paths = Paths(
        train_csv=base / "metadata_train.csv",
        test_csv=base / "metadata_test.csv",
        output_dir=out_root,
        tables_dir=out_root / "tables",
        plots_dir=out_root / "plots",
    )
    ensure_dirs(paths)

    if not paths.train_csv.exists() or not paths.test_csv.exists():
        raise SystemExit("Missing merged train/test CSV files.")

    with paths.train_csv.open("r", newline="", encoding="utf-8", errors="replace") as f:
        fieldnames = next(csv.reader(f))

    feature_columns = [c for c in fieldnames if c not in METADATA_COLUMNS]

    label_counts = Counter()
    label_by_file_split = Counter()
    protocol_counts = Counter()
    protocol_by_label = Counter()
    protocol_by_split = Counter()
    device_counts = Counter()
    family_counts = Counter()
    family_by_protocol = Counter()
    attack_name_counts = Counter()
    source_group_counts = Counter()
    source_file_counts = Counter()
    scenario_counts = Counter()

    sampled_feature_values = {
        0: {f: array("f") for f in feature_columns},
        1: {f: array("f") for f in feature_columns},
    }
    sampled_missing_counts = {
        0: {f: 0 for f in feature_columns},
        1: {f: 0 for f in feature_columns},
    }
    sampled_zero_counts = {
        0: {f: 0 for f in feature_columns},
        1: {f: 0 for f in feature_columns},
    }
    sampled_row_count = Counter()

    file_map = [("train", paths.train_csv), ("test", paths.test_csv)]

    total_rows = 0

    for logical_split, csv_path in file_map:
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                label = maybe_int(row.get("label", "0"), default=0)
                label = 1 if label == 1 else 0

                label_counts[label] += 1
                label_by_file_split[(logical_split, label)] += 1

                protocol = (row.get("protocol_hint", "") or "unknown").strip() or "unknown"
                device = (row.get("device", "") or "unknown").strip() or "unknown"
                family = (row.get("attack_family", "") or "unknown").strip() or "unknown"
                attack_name = (row.get("attack_name", "") or "unknown").strip() or "unknown"
                source_group = (row.get("source_group", "") or "unknown").strip() or "unknown"
                source_relpath = (row.get("source_relpath", "") or "unknown").strip() or "unknown"
                scenario = (row.get("scenario", "") or "unknown").strip() or "unknown"

                protocol_counts[protocol] += 1
                protocol_by_label[(protocol, label)] += 1
                protocol_by_split[(logical_split, protocol)] += 1
                device_counts[device] += 1
                family_counts[family] += 1
                family_by_protocol[(family, protocol)] += 1
                source_group_counts[source_group] += 1
                scenario_counts[scenario] += 1
                source_file_counts[source_relpath] += 1
                if label == 1:
                    attack_name_counts[attack_name] += 1

                if random.random() < DEFAULT_SAMPLE_RATE:
                    sampled_row_count[label] += 1
                    for fcol in feature_columns:
                        val, missing = safe_float(row.get(fcol, ""))
                        sampled_feature_values[label][fcol].append(val)
                        if missing:
                            sampled_missing_counts[label][fcol] += 1
                        if val == 0.0:
                            sampled_zero_counts[label][fcol] += 1

    benign_total = label_counts[0]
    attack_total = label_counts[1]
    attack_share = (attack_total / total_rows) if total_rows else 0.0
    benign_share = (benign_total / total_rows) if total_rows else 0.0

    # Core tables
    write_csv(
        paths.tables_dir / "class_balance.csv",
        ["label", "rows", "share"],
        [
            ["benign", benign_total, f"{benign_share:.6f}"],
            ["attack", attack_total, f"{attack_share:.6f}"],
        ],
    )

    protocol_rows = []
    for protocol, n in protocol_counts.most_common():
        benign_n = protocol_by_label[(protocol, 0)]
        attack_n = protocol_by_label[(protocol, 1)]
        total_n = benign_n + attack_n
        benign_ratio = benign_n / total_n if total_n else 0.0
        attack_ratio = attack_n / total_n if total_n else 0.0
        protocol_rows.append([protocol, total_n, benign_n, attack_n, f"{benign_ratio:.6f}", f"{attack_ratio:.6f}"])
    write_csv(
        paths.tables_dir / "protocol_label_counts.csv",
        ["protocol_hint", "rows", "benign_rows", "attack_rows", "benign_ratio", "attack_ratio"],
        protocol_rows,
    )

    family_rows = []
    for fam, n in family_counts.most_common():
        family_rows.append([fam, n, f"{(n / total_rows) if total_rows else 0.0:.6f}"])
    write_csv(paths.tables_dir / "attack_family_counts.csv", ["attack_family", "rows", "share"], family_rows)

    attack_name_rows = [[name, n] for name, n in attack_name_counts.most_common(25)]
    write_csv(paths.tables_dir / "top_attack_names.csv", ["attack_name", "rows"], attack_name_rows)

    split_rows = []
    for split in ("train", "test"):
        n0 = label_by_file_split[(split, 0)]
        n1 = label_by_file_split[(split, 1)]
        nt = n0 + n1
        split_rows.append([split, nt, n0, n1, f"{(n0/nt) if nt else 0.0:.6f}", f"{(n1/nt) if nt else 0.0:.6f}"])
    write_csv(
        paths.tables_dir / "split_label_balance.csv",
        ["split", "rows", "benign_rows", "attack_rows", "benign_ratio", "attack_ratio"],
        split_rows,
    )

    protocol_shift_rows = []
    train_total = sum(protocol_by_split[("train", p)] for p in protocol_counts)
    test_total = sum(protocol_by_split[("test", p)] for p in protocol_counts)
    for p in protocol_counts:
        tr = protocol_by_split[("train", p)]
        te = protocol_by_split[("test", p)]
        tr_share = (tr / train_total) if train_total else 0.0
        te_share = (te / test_total) if test_total else 0.0
        protocol_shift_rows.append([p, tr, te, f"{tr_share:.6f}", f"{te_share:.6f}", f"{(te_share - tr_share):.6f}"])
    protocol_shift_rows.sort(key=lambda r: abs(float(r[5])), reverse=True)
    write_csv(
        paths.tables_dir / "train_test_protocol_shift.csv",
        ["protocol_hint", "train_rows", "test_rows", "train_share", "test_share", "share_diff_test_minus_train"],
        protocol_shift_rows,
    )

    source_rows = [[src, n] for src, n in source_file_counts.most_common(25)]
    write_csv(paths.tables_dir / "top_source_files_by_rows.csv", ["source_relpath", "rows"], source_rows)

    scenario_rows = [[sc, n] for sc, n in scenario_counts.most_common(25)]
    write_csv(paths.tables_dir / "scenario_counts.csv", ["scenario", "rows"], scenario_rows)

    # Feature separability (from sampled rows)
    feature_sep = []
    for fcol in feature_columns:
        benign_vals = sampled_feature_values[0][fcol]
        attack_vals = sampled_feature_values[1][fcol]
        if len(benign_vals) < 10 or len(attack_vals) < 10:
            continue

        b_list = benign_vals.tolist()
        a_list = attack_vals.tolist()

        m0 = statistics.fmean(b_list)
        m1 = statistics.fmean(a_list)
        s0 = statistics.pstdev(b_list) if len(b_list) > 1 else 0.0
        s1 = statistics.pstdev(a_list) if len(a_list) > 1 else 0.0
        ps = pooled_std(b_list, a_list)
        d = ((m1 - m0) / ps) if ps > 0 else 0.0

        miss0 = sampled_missing_counts[0][fcol]
        miss1 = sampled_missing_counts[1][fcol]
        z0 = sampled_zero_counts[0][fcol]
        z1 = sampled_zero_counts[1][fcol]
        n0 = len(b_list)
        n1 = len(a_list)

        feature_sep.append(
            [
                fcol,
                n0,
                n1,
                f"{m0:.6f}",
                f"{m1:.6f}",
                f"{s0:.6f}",
                f"{s1:.6f}",
                f"{d:.6f}",
                f"{(z0/n0) if n0 else 0.0:.6f}",
                f"{(z1/n1) if n1 else 0.0:.6f}",
                f"{(miss0/n0) if n0 else 0.0:.6f}",
                f"{(miss1/n1) if n1 else 0.0:.6f}",
            ]
        )

    feature_sep.sort(key=lambda r: abs(float(r[7])), reverse=True)
    write_csv(
        paths.tables_dir / "feature_separability_sampled.csv",
        [
            "feature",
            "benign_sample_n",
            "attack_sample_n",
            "benign_mean",
            "attack_mean",
            "benign_std",
            "attack_std",
            "cohens_d_attack_minus_benign",
            "benign_zero_rate",
            "attack_zero_rate",
            "benign_missing_rate",
            "attack_missing_rate",
        ],
        feature_sep,
    )

    top_features = [row[0] for row in feature_sep[:8]]

    # Detailed summaries for selected features
    detail_rows = []
    for fcol in top_features:
        b_vals = sorted(sampled_feature_values[0][fcol].tolist())
        a_vals = sorted(sampled_feature_values[1][fcol].tolist())
        if len(b_vals) < 10 or len(a_vals) < 10:
            continue
        detail_rows.append(
            [
                fcol,
                "benign",
                len(b_vals),
                f"{quantile(b_vals, 0.10):.6f}",
                f"{quantile(b_vals, 0.25):.6f}",
                f"{quantile(b_vals, 0.50):.6f}",
                f"{quantile(b_vals, 0.75):.6f}",
                f"{quantile(b_vals, 0.90):.6f}",
                f"{b_vals[0]:.6f}",
                f"{b_vals[-1]:.6f}",
            ]
        )
        detail_rows.append(
            [
                fcol,
                "attack",
                len(a_vals),
                f"{quantile(a_vals, 0.10):.6f}",
                f"{quantile(a_vals, 0.25):.6f}",
                f"{quantile(a_vals, 0.50):.6f}",
                f"{quantile(a_vals, 0.75):.6f}",
                f"{quantile(a_vals, 0.90):.6f}",
                f"{a_vals[0]:.6f}",
                f"{a_vals[-1]:.6f}",
            ]
        )
    write_csv(
        paths.tables_dir / "feature_quantiles_top8_sampled.csv",
        ["feature", "class", "sample_n", "p10", "p25", "p50", "p75", "p90", "min", "max"],
        detail_rows,
    )

    # Correlation matrix for top features (sampled, combined classes)
    corr_rows = []
    combined = {}
    for fcol in top_features:
        combined[fcol] = sampled_feature_values[0][fcol].tolist() + sampled_feature_values[1][fcol].tolist()

    for i, fi in enumerate(top_features):
        for j, fj in enumerate(top_features):
            if j < i:
                continue
            c = pearson_corr(combined[fi], combined[fj]) if fi in combined and fj in combined else 0.0
            corr_rows.append([fi, fj, f"{c:.6f}"])
    write_csv(paths.tables_dir / "top_feature_correlations_sampled.csv", ["feature_i", "feature_j", "pearson_corr"], corr_rows)

    # Deeper iterative exploration: dominant attack family by protocol
    dominant_family = family_counts.most_common(1)[0][0] if family_counts else "unknown"
    dominant_family_share = (family_counts[dominant_family] / total_rows) if total_rows else 0.0
    deep_rows = []
    top_fams = [fam for fam, _ in family_counts.most_common(10)]
    top_protocols = [p for p, _ in protocol_counts.most_common(8)]
    for fam in top_fams:
        for proto in top_protocols:
            deep_rows.append([fam, proto, family_by_protocol[(fam, proto)]])
    write_csv(paths.tables_dir / "attack_family_by_protocol_top10x8.csv", ["attack_family", "protocol_hint", "rows"], deep_rows)

    # Plots
    save_bar_chart(
        paths.plots_dir / "01_class_balance.svg",
        "Class Balance (All Rows)",
        ["Benign", "Attack"],
        [benign_total, attack_total],
        color="#2a9d8f",
    )

    top_protocols_plot = [p for p, _ in protocol_counts.most_common(10)]
    benign_proto_vals = [protocol_by_label[(p, 0)] for p in top_protocols_plot]
    attack_proto_vals = [protocol_by_label[(p, 1)] for p in top_protocols_plot]
    save_stacked_bar_chart(
        paths.plots_dir / "02_top_protocols_by_label.svg",
        "Top Protocols with Benign/Attack Composition",
        top_protocols_plot,
        {"Benign": benign_proto_vals, "Attack": attack_proto_vals},
        {"Benign": "#4e79a7", "Attack": "#e15759"},
        rotate_labels=True,
    )

    top_families_plot = [f for f, _ in family_counts.most_common(12)]
    top_family_vals = [family_counts[f] for f in top_families_plot]
    save_bar_chart(
        paths.plots_dir / "03_top_attack_families.svg",
        "Top Attack Families / Traffic Families",
        top_families_plot,
        top_family_vals,
        color="#e76f51",
        rotate_labels=True,
    )

    shift_plot_protocols = [r[0] for r in protocol_shift_rows[:10]]
    train_vals = [protocol_by_split[("train", p)] for p in shift_plot_protocols]
    test_vals = [protocol_by_split[("test", p)] for p in shift_plot_protocols]
    save_stacked_bar_chart(
        paths.plots_dir / "04_protocol_train_test_counts.svg",
        "Protocol Presence by Split (Top Shift Candidates)",
        shift_plot_protocols,
        {"Train": train_vals, "Test": test_vals},
        {"Train": "#59a14f", "Test": "#f28e2b"},
        rotate_labels=True,
    )

    d_labels = [r[0] for r in feature_sep[:12]]
    d_vals = [abs(float(r[7])) for r in feature_sep[:12]]
    save_bar_chart(
        paths.plots_dir / "05_top_feature_effect_sizes.svg",
        "Top Features by |Cohen's d| (Sampled)",
        d_labels,
        d_vals,
        color="#9c755f",
        rotate_labels=True,
    )

    for idx, feat in enumerate(top_features[:3], start=6):
        b = sampled_feature_values[0][feat].tolist()
        a = sampled_feature_values[1][feat].tolist()
        save_hist_overlay(
            paths.plots_dir / f"0{idx}_hist_{feat.replace(' ', '_')}.svg",
            f"Distribution Overlay: {feat} (Sampled)",
            b,
            a,
            bins=36,
        )

    # Report markdown with interpretations
    top_protocol_line = ", ".join([f"{p} ({n:,})" for p, n in protocol_counts.most_common(5)])
    top_family_line = ", ".join([f"{f} ({n:,})" for f, n in family_counts.most_common(5)])
    top_attack_name_line = ", ".join([f"{n} ({c:,})" for n, c in attack_name_counts.most_common(5)])

    split_attack_rates = {
        s[0]: float(s[5]) for s in split_rows
    }
    split_rate_delta = split_attack_rates.get("test", 0.0) - split_attack_rates.get("train", 0.0)

    interpret_lines = []
    interpret_lines.append(
        f"1. Severe class imbalance persists: attack share is **{attack_share:.2%}** vs benign **{benign_share:.2%}**. "
        "For IDS thresholding, prioritize PR-AUC/FPR-constrained recall rather than raw accuracy."
    )
    interpret_lines.append(
        f"2. Protocol mix is concentrated: top protocols are {top_protocol_line}. "
        "Model evaluation should include per-protocol slices to avoid over-crediting dominant traffic types."
    )
    interpret_lines.append(
        f"3. Traffic/attack-family concentration is high: {top_family_line}. "
        "This can bias learning toward frequent families and suppress recall on rare attacks."
    )
    interpret_lines.append(
        f"4. Attack-name dominance (top): {top_attack_name_line}. "
        "Use macro metrics and family-wise confusion matrices to track long-tail behavior."
    )
    interpret_lines.append(
        f"5. Train-test attack-rate delta is **{split_rate_delta:+.2%}** (test minus train). "
        "Any threshold policy selected on train/validation should be rechecked on test under fixed FPR targets."
    )

    if dominant_family_share > 0.30:
        interpret_lines.append(
            f"6. Iterative deep dive triggered: dominant family '{dominant_family}' occupies **{dominant_family_share:.2%}** of all rows. "
            "The report includes family-by-protocol tables to inspect concentration pockets that may drive shortcut learning."
        )

    if abs(split_rate_delta) > 0.03:
        interpret_lines.append(
            "7. Additional shift alert: noticeable split-level attack-rate drift suggests careful validation design (e.g., stratified CV by family/protocol) before final model claims."
        )

    report = []
    report.append("# Advanced EDA Report: CICIoMT2024 Merged Metadata Dataset")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Thesis Objective Alignment")
    report.append(
        "This EDA is framed for a **flow-based IoMT IDS (benign vs malicious)** and focuses on class imbalance, protocol/attack composition, split drift, and feature separability to inform thresholding, robustness checks, and explainability priorities."
    )
    report.append("")
    report.append("## Dataset Scope")
    report.append(f"- Total rows analyzed (full metadata pass): **{total_rows:,}**")
    report.append(f"- Feature columns: **{len(feature_columns)}**")
    report.append(f"- Metadata columns: **{len(METADATA_COLUMNS)}**")
    report.append(f"- Sampled rows for feature-level analysis: benign={sampled_row_count[0]:,}, attack={sampled_row_count[1]:,}")
    report.append("")
    report.append("## Key Plots")
    report.append("- ![Class Balance](plots/01_class_balance.svg)")
    report.append("- ![Top Protocols by Label](plots/02_top_protocols_by_label.svg)")
    report.append("- ![Top Attack Families](plots/03_top_attack_families.svg)")
    report.append("- ![Protocol Train Test Counts](plots/04_protocol_train_test_counts.svg)")
    report.append("- ![Top Feature Effect Sizes](plots/05_top_feature_effect_sizes.svg)")

    hist_files = sorted([p.name for p in paths.plots_dir.glob("0[6-9]_hist_*.svg")])
    for hf in hist_files:
        report.append(f"- ![Feature Distribution {hf}](plots/{hf})")

    report.append("")
    report.append("## Key Tables")
    report.append("- `tables/class_balance.csv`")
    report.append("- `tables/split_label_balance.csv`")
    report.append("- `tables/protocol_label_counts.csv`")
    report.append("- `tables/attack_family_counts.csv`")
    report.append("- `tables/top_attack_names.csv`")
    report.append("- `tables/train_test_protocol_shift.csv`")
    report.append("- `tables/feature_separability_sampled.csv`")
    report.append("- `tables/feature_quantiles_top8_sampled.csv`")
    report.append("- `tables/top_feature_correlations_sampled.csv`")
    report.append("- `tables/attack_family_by_protocol_top10x8.csv`")
    report.append("")
    report.append("## Interpretation")
    for line in interpret_lines:
        report.append(f"- {line}")

    report.append("")
    report.append("## Modeling Implications for This Thesis")
    report.append(
        "- Use **class-aware evaluation** (PR-AUC, recall@FPR<=1%, F1) and not accuracy-dominated decisions."
    )
    report.append(
        "- Calibrate thresholds using validation slices by protocol/family; then freeze and evaluate once on test."
    )
    report.append(
        "- For robustness experiments, prioritize high-impact features from `feature_separability_sampled.csv` and keep perturbations physically plausible."
    )
    report.append(
        "- For explainability, first report global importance for top discriminative features, then local explanations on false positives and missed attacks."
    )

    report.append("")
    report.append("## Reproducibility")
    report.append(f"- Random seed: `{DEFAULT_RANDOM_SEED}`")
    report.append(f"- Feature sample rate: `{DEFAULT_SAMPLE_RATE}`")
    report.append("- Data sources: `data/ciciomt2024/merged/metadata_train.csv`, `metadata_test.csv`")

    (paths.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "output_dir": str(paths.output_dir),
        "total_rows": total_rows,
        "attack_rows": attack_total,
        "benign_rows": benign_total,
        "attack_share": attack_share,
        "benign_share": benign_share,
        "sampled_benign_rows": sampled_row_count[0],
        "sampled_attack_rows": sampled_row_count[1],
        "top_protocols": protocol_counts.most_common(5),
        "top_families": family_counts.most_common(5),
        "top_attack_names": attack_name_counts.most_common(5),
        "top_features_by_abs_d": [row[0] for row in feature_sep[:8]],
        "dominant_family": dominant_family,
        "dominant_family_share": dominant_family_share,
    }
    write_csv(
        paths.output_dir / "RUN_SUMMARY.csv",
        ["key", "value"],
        [[k, v] for k, v in summary.items()],
    )

    print("EDA report generated:", paths.output_dir)
    print("Total rows:", total_rows)
    print("Attack share:", f"{attack_share:.4f}")
    print("Sampled rows (benign/attack):", sampled_row_count[0], sampled_row_count[1])


if __name__ == "__main__":
    main()
