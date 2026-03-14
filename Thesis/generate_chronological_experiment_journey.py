from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

ROOT = Path(r"c:\Users\Hugo\Desktop\Thesis\Medical-IoMT-")
REPORTS = ROOT / "reports"
OUT_PATH = ROOT / "Thesis" / "CHRONOLOGICAL_EXPERIMENT_JOURNEY.md"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a"}:
        return None
    if text.lower() in {"inf", "+inf", "infinity"}:
        return math.inf
    if text.lower() in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return None


def escape_md(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_cell(value, column=""):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value).strip()
    if text == "":
        return ""
    if text.lower() in {"nan", "na", "n/a"}:
        return "NA"
    if text.lower() in {"inf", "+inf", "infinity"}:
        return "inf"
    if text.lower() in {"-inf", "-infinity"}:
        return "-inf"
    col = column.lower()
    if col in {"run_id", "candidate_group", "candidate_group_key"}:
        return escape_md(text)
    number = to_float(text)
    if number is None:
        return escape_md(text)
    if col in {"rows", "row", "n", "tp", "tn", "fp", "fn", "rank", "seed", "final_rank", "candidate_rows", "gate_pass_candidates", "stable_groups", "consistent_pass_groups", "seeds_checked", "rows_used", "locked_features", "mutable_features", "unique_files", "rows_moved_to_train", "triggered"} or col.endswith("_rows") or col.endswith("_count") or col.endswith("_groups") or col.endswith("_features") or col.endswith("_files") or col.startswith("n_") or col.startswith("num_"):
        if abs(number - round(number)) < 1e-9:
            return f"{int(round(number)):,}"
        return f"{number:,.2f}"
    if any(token in col for token in ["precision", "recall", "f1", "fpr", "auc", "ratio", "share", "rate", "threshold", "score", "objective", "epsilon", "delta", "resolution"]):
        return f"{number:.4f}"
    if abs(number) >= 1000 and abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    return f"{number:.4f}"


def md_table(rows, columns, headers=None):
    headers = headers or {}
    lines = [
        "| " + " | ".join(headers.get(column, column) for column in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column, ""), column) for column in columns) + " |")
    return "\n".join(lines)


def find_row(rows, key, value):
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row for {key}={value}")


def first_n(rows, n):
    return rows[:n]


def best_by(rows, group_key, score_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    output = []
    for key in sorted(grouped):
        output.append(max(grouped[key], key=lambda item: to_float(item[score_key]) or float("-inf")))
    return output


eda_class_balance = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "class_balance.csv")
eda_split_balance = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "split_label_balance.csv")
eda_protocol_counts = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "protocol_label_counts.csv")
eda_attack_family_counts = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "attack_family_counts.csv")
eda_top_attack_names = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "top_attack_names.csv")
eda_attack_family_by_protocol = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "attack_family_by_protocol_top10x8.csv")
eda_protocol_shift = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "train_test_protocol_shift.csv")
eda_feature_sep = read_csv(REPORTS / "eda_advanced_20260305_231027" / "tables" / "feature_separability_sampled.csv")

baseline_metrics = read_csv(REPORTS / "baseline_models_stdlib_20260305_234858" / "metrics_summary.csv")
baseline_protocol = read_csv(REPORTS / "baseline_models_stdlib_20260305_234858" / "slice_metrics_protocol.csv")
baseline_families = read_csv(REPORTS / "baseline_models_stdlib_20260305_234858" / "slice_metrics_attack_family.csv")

full_gpu_metrics = read_csv(REPORTS / "full_gpu_models_20260306_001638" / "metrics_summary.csv")
full_gpu_protocol = read_csv(REPORTS / "full_gpu_models_20260306_001638" / "slice_metrics_protocol.csv")

hpo_134806_metrics = read_csv(REPORTS / "full_gpu_hpo_models_20260306_134806" / "metrics_summary.csv")
hpo_134806_protocol = read_csv(REPORTS / "full_gpu_hpo_models_20260306_134806" / "slice_metrics_protocol.csv")

hpo_153556_metrics = read_csv(REPORTS / "full_gpu_hpo_models_20260306_153556" / "metrics_summary.csv")
hpo_153556_protocol_models = read_csv(REPORTS / "full_gpu_hpo_models_20260306_153556" / "metrics_summary_per_protocol_models.csv")

hpo_172441_protocol_models = read_csv(REPORTS / "full_gpu_hpo_models_20260306_172441" / "metrics_summary_per_protocol_models.csv")
hpo_180951_protocol_models = read_csv(REPORTS / "full_gpu_hpo_models_20260306_180951" / "metrics_summary_per_protocol_models.csv")

leakguard_metrics = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851" / "metrics_summary.csv")
leakguard_protocol_models = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851" / "metrics_summary_per_protocol_models.csv")
explainability_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851" / "xgb_explainability" / "global_feature_importance.csv")
robust_query_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851" / "xgb_robustness_realistic_full_20260308_212054" / "robustness_query_metrics_global.csv")
robust_query_protocol = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851" / "xgb_robustness_realistic_full_20260308_212054" / "robustness_query_metrics_protocol.csv")
robust_summary = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851" / "xgb_robustness_realistic_full_20260308_212054" / "summary.json")

wifi_hardening_103936 = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_103936" / "hardening_summary.json")
wifi_hardening_135449 = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_135449" / "hardening_summary.json")
wifi_hardening_180250 = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_180250" / "hardening_summary.json")
wifi_hard_negative_180250 = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_180250" / "hard_negative_stats.csv")

wifi_rebalance_decision = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_wifi_rebalance_matrix_v1_20260309_204053" / "decision_table.csv")
wifi_rebalance_stability = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_wifi_rebalance_matrix_v1_20260309_204053" / "stability_check.csv")

proto_20260310_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260310_105248" / "decision_table_global.csv")
proto_121948_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_121948" / "decision_table_global.csv")
proto_121948_wifi = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_121948" / "decision_table_protocol_wifi.csv")
proto_121948_mqtt = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_121948" / "decision_table_protocol_mqtt.csv")
proto_121948_bt = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_121948" / "decision_table_protocol_bluetooth.csv")
proto_180757_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_180757" / "decision_table_global.csv")
proto_180757_stability = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_180757" / "stability_consistency_summary.csv")
proto_200922_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_200922" / "decision_table_global.csv")
proto_200922_stability = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_200922" / "stability_consistency_summary.csv")

data_audit_protocol_balance = read_csv(REPORTS / "data_audit_20260313_124851" / "protocol_label_balance_train_test.csv")
data_audit_protocol_shift = read_csv(REPORTS / "data_audit_20260313_124851" / "protocol_share_shift_train_vs_test.csv")
data_audit_feature_drift = read_csv(REPORTS / "data_audit_20260313_124851" / "feature_drift_top20.csv")

proto_003108_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108" / "decision_table_global.csv")
proto_003108_stability = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108" / "stability_consistency_summary.csv")
proto_003108_wifi = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108" / "decision_table_protocol_wifi.csv")
proto_003108_mqtt = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108" / "decision_table_protocol_mqtt.csv")
proto_003108_bt = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108" / "decision_table_protocol_bluetooth.csv")

proto_112105_global = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "decision_table_global.csv")
proto_112105_stability = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "stability_consistency_summary.csv")
proto_112105_test_metrics = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "catboost_E_test_predictions_metrics_with_thresholds.csv")
proto_112105_wifi = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "decision_table_protocol_wifi.csv")
proto_112105_mqtt = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "decision_table_protocol_mqtt.csv")
proto_112105_bt = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "decision_table_protocol_bluetooth.csv")
proto_112105_matrix_summary = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "matrix_summary.json")
proto_112105_escalation = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "escalation_recommendation.json")
proto_112105_stability_check = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "stability_check_global.csv")
proto_112105_data_cap = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "data_cap_summary.csv")
proto_112105_external_benign = read_csv(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "external_benign_data_summary.csv")
proto_112105_realism_profile = read_json(REPORTS / "full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105" / "hardening_realism_profile.json")


def explainability_top_features():
    grouped = defaultdict(list)
    for row in explainability_global:
        if int(float(row["rank"])) <= 5:
            grouped[row["protocol_hint"]].append(row["feature"])
    return [
        {"protocol": protocol, "top_1": grouped[protocol][0], "top_2": grouped[protocol][1], "top_3": grouped[protocol][2], "top_4": grouped[protocol][3], "top_5": grouped[protocol][4]}
        for protocol in ["bluetooth", "mqtt", "wifi"]
    ]


def hardening_comparison():
    runs = [
        ("20260309_103936", wifi_hardening_103936, "Conservative FPR-first hardening"),
        ("20260309_135449", wifi_hardening_135449, "Constraint-aware pilot"),
        ("20260309_180250", wifi_hardening_180250, "Full-budget hardening"),
    ]
    output = []
    for run_id, summary, note in runs:
        selected = summary["thresholds"]["selection"]["selected_metrics"]
        output.append({
            "run_id": run_id,
            "note": note,
            "train_flip_rows": summary["hard_negative"].get("train_flip_rows"),
            "train_selected_rows": summary["hard_negative"].get("train_selected_rows", summary["hard_negative"].get("train_flip_rows")),
            "wifi_base_threshold": summary["thresholds"]["wifi_base"],
            "wifi_new_threshold": summary["thresholds"]["wifi_new"],
            "clean_fpr_new": summary["validation_metrics"]["clean_new_threshold"]["fpr"],
            "attacked_benign_fpr_new": selected.get("attacked_benign_fpr"),
            "attacked_malicious_recall_new": selected.get("attacked_malicious_recall"),
            "delta_clean_f1": summary["validation_metrics"]["delta_clean_f1"],
        })
    return output


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def find_protocol_row(rows, model, protocol):
    for row in rows:
        if row.get("model") == model and row.get("protocol_hint") == protocol:
            return row
    raise KeyError(f"Missing row for model={model}, protocol={protocol}")


def top_ranked(rows):
    ranked = sorted(rows, key=lambda row: to_float(row.get("final_rank")) if to_float(row.get("final_rank")) is not None else math.inf)
    return ranked[0]


def protocol_summary(rows, protocol):
    gate_rows = [row for row in rows if row["gate_pass"] == "True"]
    best = gate_rows[0] if gate_rows else rows[0]
    return {
        "protocol": protocol,
        "gate_pass_candidates": len(gate_rows),
        "best_candidate": f"{best['model_name']}__{best['family_id']}",
        "clean_f1": best["clean_f1"],
        "clean_fpr": best["clean_fpr"],
        "adv_malicious_recall": best["adv_malicious_recall"],
        "robust_f1": best["robust_f1"],
        "gate_pass": best["gate_pass"],
    }


def finalist_rows(rows_by_protocol, finalists):
    output = []
    for protocol, rows in rows_by_protocol:
        for label, model_name, family_id in finalists:
            for row in rows:
                if row["model_name"] == model_name and row["family_id"] == family_id and row["seed"] == "42":
                    output.append({
                        "protocol": protocol,
                        "candidate": label,
                        "clean_f1": row["clean_f1"],
                        "clean_fpr": row["clean_fpr"],
                        "attacked_benign_fpr": row["attacked_benign_fpr"],
                        "adv_malicious_recall": row["adv_malicious_recall"],
                        "robust_f1": row["robust_f1"],
                        "gate_pass": row["gate_pass"],
                    })
                    break
    return output


def catboost_e_rows(rows_by_protocol):
    output = []
    for protocol, rows in rows_by_protocol:
        for row in rows:
            if row["model_name"] == "catboost" and row["family_id"] == "E" and row["seed"] == "42":
                output.append({
                    "protocol": protocol,
                    "selected_threshold": row["selected_threshold"],
                    "clean_f1": row["clean_f1"],
                    "clean_fpr": row["clean_fpr"],
                    "attacked_benign_fpr": row["attacked_benign_fpr"],
                    "adv_malicious_recall": row["adv_malicious_recall"],
                    "robust_f1": row["robust_f1"],
                    "gate_pass": row["gate_pass"],
                })
                break
    return output


def early_protocol_progression():
    output = []
    for protocol in ["wifi", "mqtt", "bluetooth"]:
        baseline_best = max(
            [row for row in baseline_protocol if row["protocol_hint"] == protocol],
            key=lambda row: ((to_float(row["f1"]) or float("-inf")), -(to_float(row["fpr"]) or float("inf"))),
        )
        full_best = max(
            [row for row in full_gpu_protocol if row["protocol_hint"] == protocol],
            key=lambda row: ((to_float(row["f1"]) or float("-inf")), -(to_float(row["fpr"]) or float("inf"))),
        )
        hpo_best = max(
            [row for row in hpo_134806_protocol if row["protocol_hint"] == protocol],
            key=lambda row: ((to_float(row["f1"]) or float("-inf")), -(to_float(row["fpr"]) or float("inf"))),
        )
        output.append({
            "protocol": protocol,
            "baseline_model": baseline_best["model"],
            "baseline_f1": baseline_best["f1"],
            "baseline_fpr": baseline_best["fpr"],
            "full_gpu_model": full_best["model"],
            "full_gpu_f1": full_best["f1"],
            "full_gpu_fpr": full_best["fpr"],
            "first_hpo_model": hpo_best["model"],
            "first_hpo_f1": hpo_best["f1"],
            "first_hpo_fpr": hpo_best["fpr"],
        })
    return output


def bluetooth_progression():
    routed_bt = find_row([row for row in hpo_153556_protocol_models if row["protocol_hint"] == "bluetooth" and row["model"] == "xgboost_tuned__bluetooth"], "protocol_hint", "bluetooth")
    split_repair_bt = find_row([row for row in hpo_180951_protocol_models if row["protocol_hint"] == "bluetooth" and row["model"] == "xgboost_tuned__bluetooth"], "protocol_hint", "bluetooth")
    leakguard_bt = find_row([row for row in leakguard_protocol_models if row["protocol_hint"] == "bluetooth" and row["model"] == "xgboost_tuned__bluetooth"], "protocol_hint", "bluetooth")
    final_bt = [row for row in catboost_e_rows([("bluetooth", proto_112105_bt)]) if row["protocol"] == "bluetooth"][0]
    return [
        {
            "stage": "Full-data GPU",
            "model": "complex_mlp",
            "threshold": full_gpu_mlp["threshold"],
            "f1": find_protocol_row(full_gpu_protocol, "complex_mlp", "bluetooth")["f1"],
            "fpr": find_protocol_row(full_gpu_protocol, "complex_mlp", "bluetooth")["fpr"],
            "note": "Best early clean slice, but still too many benign alarms for deployment.",
        },
        {
            "stage": "First global HPO",
            "model": "complex_mlp_tuned",
            "threshold": hpo_134806_mlp["threshold"],
            "f1": find_protocol_row(hpo_134806_protocol, "complex_mlp_tuned", "bluetooth")["f1"],
            "fpr": find_protocol_row(hpo_134806_protocol, "complex_mlp_tuned", "bluetooth")["fpr"],
            "note": "Optimization improved little for the real Bluetooth problem.",
        },
        {
            "stage": "Initial protocol routing",
            "model": routed_bt["model"],
            "threshold": routed_bt["threshold"],
            "f1": routed_bt["f1"],
            "fpr": routed_bt["fpr"],
            "note": "Routing made Bluetooth nearly look solved, which triggered skepticism.",
        },
        {
            "stage": "Split repair",
            "model": split_repair_bt["model"],
            "threshold": split_repair_bt["threshold"],
            "f1": split_repair_bt["f1"],
            "fpr": split_repair_bt["fpr"],
            "note": "After repairing split logic, Bluetooth remained strong with defensible validation.",
        },
        {
            "stage": "Leakguard base",
            "model": leakguard_bt["model"],
            "threshold": leakguard_bt["threshold"],
            "f1": leakguard_bt["f1"],
            "fpr": leakguard_bt["fpr"],
            "note": "Leakguard preserved the improvement under stricter overlap controls.",
        },
        {
            "stage": "Final robust matrix",
            "model": "catboost__E",
            "threshold": final_bt["selected_threshold"],
            "f1": final_bt["clean_f1"],
            "fpr": final_bt["clean_fpr"],
            "note": "Final robust deployment candidate kept Bluetooth clean and stable.",
        },
    ]


def wifi_robustness_progression():
    wifi_query_mal = find_row([row for row in robust_query_protocol if row["attack_method"] == "query_sparse_hillclimb" and row["protocol_hint"] == "wifi"], "epsilon", "0.1")
    wifi_query_benign = find_row([row for row in robust_query_protocol if row["attack_method"] == "query_sparse_hillclimb_benign" and row["protocol_hint"] == "wifi"], "epsilon", "0.1")
    wifi_family_a = find_row(wifi_rebalance_decision, "family_id", "A")
    final_wifi = [row for row in catboost_e_rows([("wifi", proto_112105_wifi)]) if row["protocol"] == "wifi"][0]
    return [
        {
            "stage": "Realistic baseline",
            "representative": "xgboost leakguard base",
            "clean_fpr": robust_summary["baseline_by_protocol"]["wifi"]["fpr"],
            "attacked_benign_fpr": 0.0,
            "malicious_side_recall": robust_summary["baseline_by_protocol"]["wifi"]["recall"],
            "note": "Clean WiFi behavior looked excellent before benign-side attack pressure.",
        },
        {
            "stage": "Benign query attack eps=0.1",
            "representative": "query_sparse_hillclimb_benign",
            "clean_fpr": robust_summary["baseline_by_protocol"]["wifi"]["fpr"],
            "attacked_benign_fpr": wifi_query_benign["fpr"],
            "malicious_side_recall": wifi_query_mal["recall"],
            "note": "Operational problem emerged as benign drift, not malicious evasion.",
        },
        {
            "stage": "Conservative WiFi hardening",
            "representative": "wifi_robust_v1_103936",
            "clean_fpr": wifi_hardening_103936["validation_metrics"]["clean_new_threshold"]["fpr"],
            "attacked_benign_fpr": wifi_hardening_103936["thresholds"]["selection"]["selected_metrics"]["attacked_benign_fpr"],
            "malicious_side_recall": wifi_hardening_103936["thresholds"]["selection"]["selected_metrics"].get("attacked_malicious_recall", wifi_hardening_103936["thresholds"]["selection"]["selected_metrics"].get("clean_recall")),
            "note": "First hardening pass improved robustness without wrecking the clean slice.",
        },
        {
            "stage": "Full-budget WiFi hardening",
            "representative": "wifi_robust_v1_180250",
            "clean_fpr": wifi_hardening_180250["validation_metrics"]["clean_new_threshold"]["fpr"],
            "attacked_benign_fpr": wifi_hardening_180250["thresholds"]["selection"]["selected_metrics"]["attacked_benign_fpr"],
            "malicious_side_recall": wifi_hardening_180250["thresholds"]["selection"]["selected_metrics"].get("attacked_malicious_recall", wifi_hardening_180250["thresholds"]["selection"]["selected_metrics"].get("clean_recall")),
            "note": "More aggressive recovery made the clean FPR trade-off unacceptable.",
        },
        {
            "stage": "WiFi rebalance seed-42 winner",
            "representative": "family A",
            "clean_fpr": wifi_family_a["clean_fpr"],
            "attacked_benign_fpr": wifi_family_a["attacked_benign_fpr"],
            "malicious_side_recall": wifi_family_a["adv_malicious_recall"],
            "note": "This was the temporary answer that failed stability on rerun.",
        },
        {
            "stage": "Final robust matrix",
            "representative": "catboost__E",
            "clean_fpr": final_wifi["clean_fpr"],
            "attacked_benign_fpr": final_wifi["attacked_benign_fpr"],
            "malicious_side_recall": final_wifi["adv_malicious_recall"],
            "note": "The final deployment baseline restored a viable WiFi trade-off under stability.",
        },
    ]


def matrix_campaign_evolution():
    campaigns = [
        ("20260310_105248", "Feasibility pass", proto_20260310_global, None, "Original formulation was too narrow and infeasible."),
        ("20260312_121948", "All-model coarse pass", proto_121948_global, None, "First all-protocol gate-pass candidates appeared."),
        ("20260312_180757", "Targeted recovery", proto_180757_global, proto_180757_stability, "Recovery families widened the candidate set but not stable promotion."),
        ("20260312_200922", "Longer stabilization sweep", proto_200922_global, proto_200922_stability, "Any-pass groups existed, but none were consistent."),
        ("20260314_003108", "Near-final stabilized pass", proto_003108_global, proto_003108_stability, "Two stable finalists existed, but robust artifacts were missing."),
        ("20260314_112105", "Artifact-persistent final pass", proto_112105_global, proto_112105_stability, "Only one fully stable and saved deployment candidate remained."),
    ]
    output = []
    for run_id, label, rows, stability_rows, note in campaigns:
        top = top_ranked(rows)
        stable_groups = None if stability_rows is None else sum(1 for row in stability_rows if is_true(row["consistent_gate_pass"]))
        output.append({
            "run_id": run_id,
            "focus": label,
            "candidate_rows": len(rows),
            "gate_pass_candidates": sum(1 for row in rows if is_true(row["gate_pass"])),
            "stable_groups": stable_groups,
            "rank_1_candidate": top.get("candidate_group_key") or top.get("candidate_key"),
            "note": note,
        })
    return output


def family_definition_rows():
    return [
        {
            "family_id": family["family_id"],
            "family_name": family["family_name"],
            "family_description": family["family_description"],
            "hardneg_weight": family["hardneg_weight"],
            "maladv_weight": family["maladv_weight"],
            "extra_topk_per_epsilon": family["extra_topk_per_epsilon"],
        }
        for family in proto_112105_matrix_summary["families"]
    ]


def gate_definition_rows():
    gates = proto_112105_matrix_summary["gates"]
    return [
        {"gate": "clean_fpr_max", "value": gates["clean_fpr_max"], "meaning": "Upper bound for clean benign false-positive rate."},
        {"gate": "attacked_benign_fpr_max", "value": gates["attacked_benign_fpr_max"], "meaning": "Upper bound for benign-side FPR after attack."},
        {"gate": "adv_malicious_recall_min", "value": gates["adv_malicious_recall_min"], "meaning": "Minimum acceptable malicious recall under attack."},
        {"gate": "threshold_gate_attacked_benign_margin", "value": gates["threshold_gate_attacked_benign_margin"], "meaning": "Internal threshold margin used during gate-preserving selection."},
        {"gate": "strict_fpr_feasibility_check", "value": gates["strict_fpr_feasibility_check"], "meaning": "Rejects claims when FPR cannot be resolved precisely enough."},
        {"gate": "min_val_benign_for_fpr_gate", "value": gates["min_val_benign_for_fpr_gate"], "meaning": "Minimum benign validation rows required to enforce FPR gates."},
    ]


def realism_summary_rows():
    output = []
    for protocol in ["wifi", "mqtt", "bluetooth"]:
        info = proto_112105_realism_profile["protocols"][protocol]
        output.append({
            "protocol": protocol,
            "rows_used": info["rows_used"],
            "locked_features": info["locked_features"],
            "mutable_features": info["mutable_features"],
            "tot_num_ratio_low": info["ratio_tot_num_avg"]["low"],
            "tot_num_ratio_high": info["ratio_tot_num_avg"]["high"],
            "rate_ratio_low": info["ratio_rate"]["low"],
            "rate_ratio_high": info["ratio_rate"]["high"],
        })
    return output


def add_protocol_counts(counts):
    grouped = defaultdict(int)
    for key, value in counts.items():
        grouped[key.split("::")[0]] += int(value)
    return grouped


def robust_sampling_rows():
    output = []
    for label, section, per_target in [
        ("Surrogate training", robust_summary["train_sampling"], robust_summary["train_sampling"]["target_per_protocol"]),
        ("Evaluation", robust_summary["eval_sampling"], "attack<=25000, benign<=25000"),
        ("Realistic evaluation", robust_summary["realistic_eval_sampling"], "attack<=5000, benign<=5000"),
    ]:
        counts = add_protocol_counts(section["counts_sampled"])
        output.append({
            "stage": label,
            "wifi_rows": counts.get("wifi", 0),
            "mqtt_rows": counts.get("mqtt", 0),
            "bluetooth_rows": counts.get("bluetooth", 0),
            "total_rows": section["rows_sampled_total"],
            "per_protocol_target": per_target,
        })
    return output


def data_cap_rows():
    output = []
    for row in proto_112105_data_cap:
        label = row["label"]
        if str(label) == "0":
            label = "benign"
        elif str(label) == "1":
            label = "attack"
        output.append({
            "protocol": row["protocol"],
            "label_name": label,
            "rows": row["rows"],
            "unique_files": row["unique_files"],
            "largest_file_share": row["largest_file_share"],
            "fallback_type": row["fallback_type"],
            "val_rows": row["val_rows"],
            "fpr_resolution": row["fpr_resolution"],
            "floor_applied": row["floor_applied"],
            "rows_moved_to_train": row["rows_moved_to_train"],
        })
    return output


def stability_envelope_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate_group_key"]].append(row)
    output = []
    for key in sorted(grouped):
        bucket = grouped[key]
        output.append({
            "candidate_group": key,
            "all_seed_gate_pass": all(is_true(row["global_gate_pass"]) for row in bucket),
            "seeds_checked": len(bucket),
            "min_clean_fpr": min(to_float(row["worst_clean_fpr"]) or float("inf") for row in bucket),
            "max_clean_fpr": max(to_float(row["worst_clean_fpr"]) or float("-inf") for row in bucket),
            "min_attacked_benign_fpr": min(to_float(row["worst_attacked_benign_fpr"]) or float("inf") for row in bucket),
            "max_attacked_benign_fpr": max(to_float(row["worst_attacked_benign_fpr"]) or float("-inf") for row in bucket),
            "min_adv_malicious_recall": min(to_float(row["worst_adv_malicious_recall"]) or float("inf") for row in bucket),
            "max_adv_malicious_recall": max(to_float(row["worst_adv_malicious_recall"]) or float("-inf") for row in bucket),
            "primary_fail_reason": bucket[0]["primary_fail_reason"],
        })
    output.sort(key=lambda row: (not row["all_seed_gate_pass"], row["candidate_group"]))
    return output


def stable_candidate_shift_rows():
    near_final_stable = [row["candidate_group_key"] for row in proto_003108_stability if is_true(row["consistent_gate_pass"])]
    final_stable = [row["candidate_group_key"] for row in proto_112105_stability if is_true(row["consistent_gate_pass"])]
    return [
        {
            "run_id": "20260314_003108",
            "consistent_pass_groups": len(near_final_stable),
            "stable_candidates": ", ".join(near_final_stable),
            "artifact_persistence": "No",
            "decision": "Useful analytical shortlist, but not deployable.",
        },
        {
            "run_id": "20260314_112105",
            "consistent_pass_groups": len(final_stable),
            "stable_candidates": ", ".join(final_stable),
            "artifact_persistence": "Yes",
            "decision": "Single stable deployment baseline remained.",
        },
    ]


def thesis_caveat_rows():
    attack_row = find_row(eda_class_balance, "label", "attack")
    wifi_row = find_row(eda_protocol_counts, "protocol_hint", "wifi")
    bt_cap_row = find_row([row for row in proto_112105_data_cap if row["protocol"] == "bluetooth" and row["label"] == "0"], "protocol", "bluetooth")
    mqtt_test_row = find_row([row for row in data_audit_protocol_balance if row["split"] == "test" and row["protocol"] == "mqtt"], "split", "test")
    return [
        {
            "caveat": "Attack-heavy corpus",
            "evidence": f"attack share={format_cell(attack_row['share'], 'share')}",
            "implication": "Accuracy would overstate performance; FPR-constrained evaluation is mandatory.",
        },
        {
            "caveat": "WiFi dominates the merged dataset",
            "evidence": f"wifi rows={format_cell(wifi_row['rows'], 'rows')}, share={format_cell(to_float(wifi_row['rows']) / 9320045, 'share')}",
            "implication": "Protocol slicing is needed to stop WiFi from masking minority-protocol failures.",
        },
        {
            "caveat": "MQTT lacks benign negatives in test",
            "evidence": f"test mqtt benign={format_cell(mqtt_test_row['benign'], 'rows')}",
            "implication": "MQTT FPR and ROC-AUC on the final test split are unavailable, not zero.",
        },
        {
            "caveat": "Bluetooth benign data is highly concentrated",
            "evidence": f"largest benign file share={format_cell(bt_cap_row['largest_file_share'], 'share')}, benign floor applied={bt_cap_row['floor_applied']}",
            "implication": "Bluetooth claims must be interpreted with file concentration and floor repairs in mind.",
        },
        {
            "caveat": "No external benign augmentation in final run",
            "evidence": f"external benign enabled={proto_112105_external_benign[0]['enabled']}",
            "implication": "Final deployment claims remain in-distribution rather than cross-source calibrated.",
        },
    ]


baseline_logistic = find_row(baseline_metrics, "model", "logistic_sgd")
full_gpu_mlp = find_row(full_gpu_metrics, "model", "complex_mlp")
hpo_134806_mlp = find_row(hpo_134806_metrics, "model", "complex_mlp_tuned")
hpo_153556_xgb = find_row(hpo_153556_metrics, "model", "xgboost_tuned__protocol_routed")
leakguard_ensemble = find_row(leakguard_metrics, "model", "weighted_ensemble__protocol_routed")
robust_global_benign_eps01 = find_row([row for row in robust_query_global if row["attack_method"] == "query_sparse_hillclimb_benign"], "epsilon", "0.1")
robust_global_mal_eps01 = find_row([row for row in robust_query_global if row["attack_method"] == "query_sparse_hillclimb"], "epsilon", "0.1")
best_leakguard_by_protocol = best_by([row for row in leakguard_protocol_models if "__protocol_routed" not in row["model"]], "protocol_hint", "selection_score")
parts = []
chronology_rows = [
    {"date": "2026-03-05", "phase": "Data merge and advanced EDA", "artifact": "reports/eda_advanced_20260305_231027", "next": "Establish a fast baseline before scaling up."},
    {"date": "2026-03-05", "phase": "Reduced-sample baseline matrix", "artifact": "reports/baseline_models_stdlib_20260305_234858", "next": "Move to full-data GPU training."},
    {"date": "2026-03-06", "phase": "Full-data GPU training and first HPO", "artifact": "reports/full_gpu_models_20260306_001638 and reports/full_gpu_hpo_models_20260306_134806", "next": "Make training protocol-aware and repair FPR behavior."},
    {"date": "2026-03-06", "phase": "Protocol-routed HPO, FPR-fix, split repair, leakguard", "artifact": "reports/full_gpu_hpo_models_20260306_153556 -> ...172441 -> ...180951 -> ...195851", "next": "Use the leakguard base run for explanation and robustness analysis."},
    {"date": "2026-03-07", "phase": "Explainability and UI", "artifact": "reports/full_gpu_hpo_models_20260306_195851/xgb_explainability", "next": "Attack the model under realistic constraints."},
    {"date": "2026-03-08 to 2026-03-09", "phase": "Realistic robustness and WiFi hardening", "artifact": "reports/...xgb_robustness_realistic_full_20260308_212054 and wifi_robust_v1_*", "next": "Rebalance hardening strategies and then move to protocol-wide robust selection."},
    {"date": "2026-03-09", "phase": "WiFi rebalance matrix", "artifact": "reports/full_gpu_hpo_models_20260306_195851_wifi_rebalance_matrix_v1_20260309_204053", "next": "Build a protocol-wide robust matrix."},
    {"date": "2026-03-10 to 2026-03-14", "phase": "Protocol-wide robust matrix and stability passes", "artifact": "reports/...20260310_105248 -> ...121948 -> ...180757 -> ...200922 -> ...003108 -> ...112105", "next": "Promote the only stable artifact-persistent deployment baseline."},
    {"date": "2026-03-13", "phase": "Data-first audit", "artifact": "reports/data_audit_20260313_124851", "next": "Use data reality to justify stability-aware selection and explicit thesis caveats."},
]
decision_log_rows = [
    {"date": "2026-03-05", "observation": "EDA showed 93% attack share and overwhelming WiFi dominance.", "decision": "Use protocol slices and FPR-aware thresholding from the start.", "why_next": "Global accuracy would be misleading."},
    {"date": "2026-03-05", "observation": "Small stdlib baselines already achieved strong F1.", "decision": "Scale to full-data GPU models instead of polishing shallow baselines.", "why_next": "Learnability was confirmed quickly and cheaply."},
    {"date": "2026-03-06", "observation": "Full-data global metrics were excellent, but Bluetooth FPR remained uncomfortable.", "decision": "Move to constrained HPO under lower-FPR objectives.", "why_next": "Operational benign alarms mattered more than another decimal in global F1."},
    {"date": "2026-03-06", "observation": "First HPO still left the tuned MLP above the desired FPR regime.", "decision": "Abandon the single-global-model assumption and route by protocol.", "why_next": "Different traffic regimes were behaving as different problems."},
    {"date": "2026-03-06", "observation": "Protocol routing made difficult slices look almost perfect.", "decision": "Audit split logic, leakage, and fallback behavior before trusting the result.", "why_next": "A suspiciously easy win is a warning, not a conclusion."},
    {"date": "2026-03-06", "observation": "First FPR-fix run collapsed to WiFi-only output.", "decision": "Repair split construction with protocol- and class-aware handling.", "why_next": "Evaluation integrity had become the main blocker."},
    {"date": "2026-03-06", "observation": "Leakguard produced the first trusted routed baseline.", "decision": "Use that run as the base for explainability and robustness.", "why_next": "The project needed trustworthy failure analysis, not another clean-only score."},
    {"date": "2026-03-08", "observation": "Realistic query attacks barely hurt malicious recall but exploded WiFi benign FPR.", "decision": "Target WiFi hardening directly.", "why_next": "Benign-side robustness became the operational bottleneck."},
    {"date": "2026-03-09", "observation": "WiFi rebalance found a seed-42 answer that failed stability.", "decision": "Generalize the search into a protocol-wide robust matrix.", "why_next": "A local WiFi-only fix was not thesis-grade evidence."},
    {"date": "2026-03-10 to 2026-03-14", "observation": "Coarse winners kept failing stability or artifact persistence.", "decision": "Keep rerunning until one candidate satisfied gates, stability, and saved-model requirements simultaneously.", "why_next": "Deployment claims needed more than a single-seed leaderboard."},
]
engineering_intervention_rows = [
    {"date": "2026-03-10", "intervention": "Stability grouping and deterministic seed logic were corrected.", "why": "Cross-seed consistency was being misread by earlier campaign code.", "effect": "Stability results became scientifically interpretable."},
    {"date": "2026-03-10", "intervention": "CPU threading and startup overhead were tuned for the Slurm jobs.", "why": "Robust matrix runs were too slow and underutilizing the machine.", "effect": "Later campaigns could evaluate more candidates within feasible wall-clock."},
    {"date": "2026-03-11", "intervention": "Per-candidate timing and batched query scoring were added.", "why": "The attack loop was CPU-bound and opaque during long runs.", "effect": "Runtime became diagnosable and significantly more tractable."},
    {"date": "2026-03-12", "intervention": "Recovery family pack and Bluetooth-specific controls were implemented.", "why": "Bluetooth remained the protocol that determined whether any candidate could pass globally.", "effect": "Families C, D, and E became plausible finalists instead of speculative ideas."},
    {"date": "2026-03-12", "intervention": "A fast-run crash from a function-signature mismatch was repaired.", "why": "Large scheduled runs were failing before producing usable results.", "effect": "The next matrix submission became execution-ready instead of wasting queue time."},
    {"date": "2026-03-14", "intervention": "Robust candidate artifact saving was enabled by default.", "why": "A stable analytical winner without saved models was not deployable.", "effect": "The final rerun could promote an actual deployment baseline rather than only a report row."},
]

parts.append(dedent(f"""
# Chronological Thesis Narrative of the IoMT IDS Development Journey

This document reconstructs the project as a thesis-style research journey using two authoritative sources: the dated reasoning in `context if youre an ai.md` and the saved results under `reports/`. Each section explains what problem was being addressed at that point, what the saved results showed, and why those results forced the next step.

The point of this document is not only to state the final answer. It is to make the path to the final answer defensible.

## Chronology at a glance

{md_table(chronology_rows, ['date', 'phase', 'artifact', 'next'], {'date': 'Date', 'phase': 'Phase', 'artifact': 'Artifact', 'next': 'Next step'})}

## 1. Advanced EDA defined the real problem before any serious training

The merged metadata table immediately showed that this was not a balanced, homogeneous binary classification problem. The saved EDA tables below explain why the later modeling pipeline became protocol-aware, threshold-aware, and robustness-aware.

### Table 1. Global class balance
{md_table(eda_class_balance, ['label', 'rows', 'share'], {'label': 'Label', 'rows': 'Rows', 'share': 'Share'})}

### Table 2. Protocol composition by label
{md_table(eda_protocol_counts, ['protocol_hint', 'rows', 'benign_rows', 'attack_rows', 'benign_ratio', 'attack_ratio'], {'protocol_hint': 'Protocol', 'rows': 'Rows', 'benign_rows': 'Benign rows', 'attack_rows': 'Attack rows', 'benign_ratio': 'Benign ratio', 'attack_ratio': 'Attack ratio'})}

### Table 3. Attack-family concentration
{md_table(first_n(eda_attack_family_counts, 7), ['attack_family', 'rows', 'share'], {'attack_family': 'Attack family', 'rows': 'Rows', 'share': 'Share'})}

### Table 4. Train-test protocol shift
{md_table(eda_protocol_shift, ['protocol_hint', 'train_rows', 'test_rows', 'train_share', 'test_share', 'share_diff_test_minus_train'], {'protocol_hint': 'Protocol', 'train_rows': 'Train rows', 'test_rows': 'Test rows', 'train_share': 'Train share', 'test_share': 'Test share', 'share_diff_test_minus_train': 'Test minus train'})}

### Table 5. Most separable sampled features
{md_table(first_n(eda_feature_sep, 8), ['feature', 'benign_mean', 'attack_mean', 'cohens_d_attack_minus_benign', 'benign_zero_rate', 'attack_zero_rate'], {'feature': 'Feature', 'benign_mean': 'Benign mean', 'attack_mean': 'Attack mean', 'cohens_d_attack_minus_benign': "Cohen's d", 'benign_zero_rate': 'Benign zero rate', 'attack_zero_rate': 'Attack zero rate'})}

The EDA results established three facts that remained true for the rest of the project. First, the dataset was attack-heavy, so low FPR thresholding had to matter at least as much as raw F1. Second, WiFi dominated the corpus, so protocol slices were necessary to keep the dominant traffic from hiding minority-protocol failure modes. Third, the feature space clearly contained usable signal, so a flow-based IDS was worth pursuing without payload inspection.

The next step followed logically: before committing full GPU budget, confirm quickly that the problem is learnable with a small baseline matrix.

## 2. Reduced-sample baselines showed that the feature space already carried real IDS signal

### Table 6. Reduced-sample global baselines
{md_table(baseline_metrics, ['model', 'threshold', 'precision', 'recall', 'f1', 'fpr', 'roc_auc'], {'model': 'Model', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR', 'roc_auc': 'ROC-AUC'})}

### Table 7. Reduced-sample protocol slices
{md_table(baseline_protocol, ['model', 'protocol_hint', 'n', 'precision', 'recall', 'f1', 'fpr'], {'model': 'Model', 'protocol_hint': 'Protocol', 'n': 'Rows', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

### Table 8. Family-wise behavior of the reduced-sample winner
{md_table(sorted([row for row in baseline_families if row['model'] == 'logistic_sgd'], key=lambda row: row['attack_family'])[:7], ['attack_family', 'n', 'precision', 'recall', 'f1', 'fpr'], {'attack_family': 'Attack family', 'n': 'Rows', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

`logistic_sgd` reached an F1 of {format_cell(baseline_logistic['f1'], 'f1')} with FPR {format_cell(baseline_logistic['fpr'], 'fpr')}. That did not mean the thesis was solved. It meant the feature space was already good enough that stronger families and full-data training were worth the cost. The next step was therefore to scale the training pipeline rather than spend more effort squeezing tiny gains out of shallow baselines.

## 3. Full-data GPU training exposed the real operational weakness: false positives on specific protocols

### Table 9. First full-data GPU global metrics
{md_table(full_gpu_metrics, ['model', 'threshold', 'precision', 'recall', 'f1', 'fpr', 'roc_auc'], {'model': 'Model', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR', 'roc_auc': 'ROC-AUC'})}

### Table 10. First full-data GPU protocol slices
{md_table(full_gpu_protocol, ['model', 'protocol_hint', 'n', 'precision', 'recall', 'f1', 'fpr'], {'model': 'Model', 'protocol_hint': 'Protocol', 'n': 'Rows', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

The global results were outstanding. The complex MLP reached F1 {format_cell(full_gpu_mlp['f1'], 'f1')}. But the protocol slices showed why global scores were insufficient for an IDS thesis. Bluetooth behaved very differently from WiFi and MQTT, and false-positive behavior remained much less comfortable there than the global row suggested.

That is why the project did not stop at "full-data training works." The next move was to tune model families under an explicit low-FPR objective.

## 4. First HPO pass improved optimization sophistication, not deployment credibility

### Table 11. First constrained HPO global metrics
{md_table(hpo_134806_metrics, ['model', 'selection_score', 'threshold', 'precision', 'recall', 'f1', 'fpr'], {'model': 'Model', 'selection_score': 'Selection score', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

### Table 12. First constrained HPO protocol slices
{md_table(hpo_134806_protocol, ['model', 'protocol_hint', 'n', 'precision', 'recall', 'f1', 'fpr'], {'model': 'Model', 'protocol_hint': 'Protocol', 'n': 'Rows', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

The HPO machinery became better, but the deployment story did not become good enough. The selected tuned MLP still had FPR {format_cell(hpo_134806_mlp['fpr'], 'fpr')}, which was too high for the low-FPR thesis target. The context log also recorded that CatBoost search had configuration issues and the ensemble mostly mirrored XGBoost.

That forced the first major structural decision of the project: stop assuming a single global model family should solve all protocol regimes in the same way. The next stage became protocol-routed training.

## 5. Protocol-routed HPO was the first big structural breakthrough

### Table 13. Initial protocol-routed global metrics
{md_table(hpo_153556_metrics, ['model', 'selection_score', 'precision', 'recall', 'f1'], {'model': 'Model', 'selection_score': 'Selection score', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'})}

### Table 14. Initial protocol-routed per-protocol model table
{md_table(hpo_153556_protocol_models, ['protocol_hint', 'model', 'selection_score', 'threshold', 'precision', 'recall', 'f1', 'fpr'], {'protocol_hint': 'Protocol', 'model': 'Model', 'selection_score': 'Selection score', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

Routing by protocol changed the picture dramatically. The routed XGBoost model reached F1 {format_cell(hpo_153556_xgb['f1'], 'f1')}, and the Bluetooth slice became almost clean enough to look solved. That was exactly why the next step had to be skeptical rather than celebratory. When a difficult slice suddenly becomes near-perfect, the correct response is to audit the validation logic and leakage controls.

## 6. FPR-fix, split repair, and leakguard turned impressive metrics into trustworthy metrics

### Table 15. WiFi-only collapse in the first FPR-fix run
{md_table(hpo_172441_protocol_models, ['protocol_hint', 'model', 'selection_score', 'threshold', 'precision', 'recall', 'f1'], {'protocol_hint': 'Protocol', 'model': 'Model', 'selection_score': 'Selection score', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'})}

### Table 16. Split-repaired protocol/class-aware run
{md_table(hpo_180951_protocol_models, ['protocol_hint', 'model', 'selection_score', 'threshold', 'precision', 'recall', 'f1', 'fpr'], {'protocol_hint': 'Protocol', 'model': 'Model', 'selection_score': 'Selection score', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

### Table 17. Leakguard global metrics
{md_table(leakguard_metrics, ['model', 'selection_score', 'val_weighted_objective', 'precision', 'recall', 'f1', 'fpr'], {'model': 'Model', 'selection_score': 'Selection score', 'val_weighted_objective': 'Validation objective', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

### Table 18. Best leakguard protocol-specific models
{md_table(best_leakguard_by_protocol, ['protocol_hint', 'model', 'selection_score', 'threshold', 'precision', 'recall', 'f1', 'fpr'], {'protocol_hint': 'Protocol', 'model': 'Model', 'selection_score': 'Selection score', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

Table 15 is a necessary failure in the story. The first FPR-fix run collapsed to WiFi-only output, which exposed a split-construction defect. Table 16 shows the repair: once the validation logic became protocol- and class-aware with deterministic fallback, all protocols returned again. Table 17 then becomes the true anchor for the rest of the project because leakguard enforced file-level overlap checks and validation-only ranking. The project chose methodological trust over cleaner-looking but weaker guarantees.

That is why the next phase did not add yet another training family. The next phase moved into explainability and adversarial robustness on top of the leakguard base run.

## 7. Explainability made the routed model intelligible enough for thesis use

### Table 19. Top explainability features by protocol
{md_table(explainability_top_features(), ['protocol', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5'], {'protocol': 'Protocol', 'top_1': 'Top feature 1', 'top_2': 'Top feature 2', 'top_3': 'Top feature 3', 'top_4': 'Top feature 4', 'top_5': 'Top feature 5'})}

The explainability artifacts confirmed that the protocol-routed architecture was not only operationally useful but also interpretable. Different protocols were driven by different feature groups, which supported the earlier decision to route the traffic rather than force a single global decision function.

The next step was to stop asking "why is the model making these decisions?" and start asking "how does the model fail under adaptive pressure?"

## 8. Realistic robustness changed the project from a performance study into a deployment study

### Table 20. Realistic query-attack global metrics
{md_table(robust_query_global, ['attack_method', 'epsilon', 'attack_objective', 'query_budget', 'targeted_rows', 'targeted_success_rate', 'precision', 'recall', 'f1', 'fpr', 'delta_f1'], {'attack_method': 'Attack method', 'epsilon': 'Epsilon', 'attack_objective': 'Objective', 'query_budget': 'Budget', 'targeted_rows': 'Targeted rows', 'targeted_success_rate': 'Targeted success rate', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR', 'delta_f1': 'Delta F1'})}

### Table 21. Protocol query metrics at baseline and epsilon 0.1
{md_table([row for row in robust_query_protocol if row['epsilon'] in {'0.0', '0.1'}], ['attack_method', 'protocol_hint', 'epsilon', 'attack_objective', 'targeted_success_rate', 'precision', 'recall', 'f1', 'fpr', 'delta_f1'], {'attack_method': 'Attack method', 'protocol_hint': 'Protocol', 'epsilon': 'Epsilon', 'attack_objective': 'Objective', 'targeted_success_rate': 'Targeted success rate', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR', 'delta_f1': 'Delta F1'})}

The robust metrics changed the research question. Malicious evasion stayed comparatively controlled, with targeted success rate {format_cell(robust_global_mal_eps01['targeted_success_rate'], 'targeted_success_rate')} at epsilon 0.1. Benign-side drift was the real deployment problem. At epsilon 0.1 the global benign-side query attack pushed FPR to {format_cell(robust_global_benign_eps01['fpr'], 'fpr')}, and the protocol table showed that WiFi was the main driver. MQTT also lacked benign test rows entirely, which became an explicit limitation.

That result forced the next phase: targeted WiFi hardening rather than generic retraining.
""").strip())
parts.append(dedent(f"""
## 9. WiFi hardening taught the project how fragile the FPR/recall trade-off really was

### Table 22. Comparison of the three WiFi hardening runs
{md_table(hardening_comparison(), ['run_id', 'note', 'train_flip_rows', 'train_selected_rows', 'wifi_base_threshold', 'wifi_new_threshold', 'clean_fpr_new', 'attacked_benign_fpr_new', 'attacked_malicious_recall_new', 'delta_clean_f1'], {'run_id': 'Run', 'note': 'Intent', 'train_flip_rows': 'Train flip rows', 'train_selected_rows': 'Train selected rows', 'wifi_base_threshold': 'Base threshold', 'wifi_new_threshold': 'New threshold', 'clean_fpr_new': 'Clean FPR', 'attacked_benign_fpr_new': 'Attacked benign FPR', 'attacked_malicious_recall_new': 'Attacked malicious recall', 'delta_clean_f1': 'Delta clean F1'})}

### Table 23. Full-budget hard-negative generation summary
{md_table(wifi_hard_negative_180250, ['split', 'epsilon', 'targeted_rows', 'flip_rows', 'flip_rate', 'selected_rows', 'queries_mean', 'queries_p95'], {'split': 'Split', 'epsilon': 'Epsilon', 'targeted_rows': 'Targeted rows', 'flip_rows': 'Flip rows', 'flip_rate': 'Flip rate', 'selected_rows': 'Selected rows', 'queries_mean': 'Queries mean', 'queries_p95': 'Queries p95'})}

The conservative run looked good. The small-sample constrained run looked even better. But the full-budget run exposed the cost of trying to recover adversarial malicious recall aggressively: the threshold fell to {format_cell(wifi_hardening_180250['thresholds']['wifi_new'], 'threshold')}, clean FPR rose to {format_cell(wifi_hardening_180250['validation_metrics']['clean_new_threshold']['fpr'], 'fpr')}, and attacked benign FPR rose to {format_cell(wifi_hardening_180250['thresholds']['selection']['selected_metrics']['attacked_benign_fpr'], 'fpr')}. That is the moment where the project learned that the WiFi problem could not be solved by a single hardening recipe.

The next step therefore became a family rebalance matrix with explicit gate checking.

## 10. The WiFi rebalance matrix found a temporary answer, then invalidated it through stability

### Table 24. WiFi rebalance decision table
{md_table(wifi_rebalance_decision, ['family_id', 'family_name', 'selected_threshold', 'clean_fpr', 'attacked_benign_fpr', 'adv_malicious_recall'], {'family_id': 'Family', 'family_name': 'Family name', 'selected_threshold': 'Threshold', 'clean_fpr': 'Clean FPR', 'attacked_benign_fpr': 'Attacked benign FPR', 'adv_malicious_recall': 'Adv. malicious recall'})}

### Table 25. WiFi rebalance stability check
{md_table(wifi_rebalance_stability, ['seed', 'family_id', 'family_name', 'gate_pass', 'clean_fpr', 'attacked_benign_fpr', 'adv_malicious_recall', 'source'], {'seed': 'Seed', 'family_id': 'Family', 'family_name': 'Family name', 'gate_pass': 'Gate pass', 'clean_fpr': 'Clean FPR', 'attacked_benign_fpr': 'Attacked benign FPR', 'adv_malicious_recall': 'Adv. malicious recall', 'source': 'Source'})}

Family A finally passed the WiFi gate on the primary seed. Then it failed stability, with adversarial malicious recall collapsing to {format_cell(wifi_rebalance_stability[1]['adv_malicious_recall'], 'recall')} on the rerun. That single table explains why the project moved away from WiFi-only remediation and into a protocol-wide robust matrix. A thesis that ignored this stability failure would be overstating its defense.

## 11. The first protocol-wide robust matrix run proved the original formulation was infeasible

### Table 26. First protocol-wide robust matrix global table
{md_table(proto_20260310_global, ['candidate_key', 'model_name', 'family_id', 'gate_failure_category', 'worst_clean_fpr', 'worst_attacked_benign_fpr', 'worst_adv_malicious_recall', 'mean_clean_f1', 'mean_robust_f1'], {'candidate_key': 'Candidate', 'model_name': 'Model', 'family_id': 'Family', 'gate_failure_category': 'Failure category', 'worst_clean_fpr': 'Worst clean FPR', 'worst_attacked_benign_fpr': 'Worst attacked benign FPR', 'worst_adv_malicious_recall': 'Worst adv. recall', 'mean_clean_f1': 'Mean clean F1', 'mean_robust_f1': 'Mean robust F1'})}

Every candidate in the first protocol-wide matrix failed, and the run only covered WiFi with XGBoost families. That failure was useful because it justified the heavy runtime and search-profile redesign work that immediately followed.

## 12. The first successful all-model coarse protocol matrix proved the idea could work

### Table 27. All-model coarse protocol matrix global ranking
{md_table(proto_121948_global, ['candidate_key', 'model_name', 'family_id', 'gate_pass', 'gate_failure_category', 'worst_clean_fpr', 'worst_attacked_benign_fpr', 'worst_adv_malicious_recall', 'mean_clean_f1', 'mean_robust_f1', 'final_rank'], {'candidate_key': 'Candidate', 'model_name': 'Model', 'family_id': 'Family', 'gate_pass': 'Gate pass', 'gate_failure_category': 'Failure category', 'worst_clean_fpr': 'Worst clean FPR', 'worst_attacked_benign_fpr': 'Worst attacked benign FPR', 'worst_adv_malicious_recall': 'Worst adv. recall', 'mean_clean_f1': 'Mean clean F1', 'mean_robust_f1': 'Mean robust F1', 'final_rank': 'Rank'})}

### Table 28. Protocol bottleneck summary from the coarse pass
{md_table([protocol_summary(proto_121948_wifi, 'wifi'), protocol_summary(proto_121948_mqtt, 'mqtt'), protocol_summary(proto_121948_bt, 'bluetooth')], ['protocol', 'gate_pass_candidates', 'best_candidate', 'clean_f1', 'clean_fpr', 'adv_malicious_recall', 'robust_f1', 'gate_pass'], {'protocol': 'Protocol', 'gate_pass_candidates': 'Gate-pass candidates', 'best_candidate': 'Best candidate', 'clean_f1': 'Clean F1', 'clean_fpr': 'Clean FPR', 'adv_malicious_recall': 'Adv. malicious recall', 'robust_f1': 'Robust F1', 'gate_pass': 'Gate pass'})}

This run mattered because `xgboost__B__42` became the first all-protocol gate-pass candidate. But Bluetooth still acted as the limiting protocol, and there was no stable final answer yet. That is why targeted recovery and multi-seed stability still had to happen.

## 13. Targeted recovery and early stabilization showed coarse winners were still not enough

### Table 29. Targeted V1 recovery run
{md_table(proto_180757_global, ['candidate_key', 'model_name', 'family_id', 'gate_pass', 'gate_failure_category', 'worst_clean_fpr', 'worst_attacked_benign_fpr', 'worst_adv_malicious_recall', 'mean_clean_f1', 'mean_robust_f1', 'final_rank'], {'candidate_key': 'Candidate', 'model_name': 'Model', 'family_id': 'Family', 'gate_pass': 'Gate pass', 'gate_failure_category': 'Failure category', 'worst_clean_fpr': 'Worst clean FPR', 'worst_attacked_benign_fpr': 'Worst attacked benign FPR', 'worst_adv_malicious_recall': 'Worst adv. recall', 'mean_clean_f1': 'Mean clean F1', 'mean_robust_f1': 'Mean robust F1', 'final_rank': 'Rank'})}

### Table 30. Stability summary for the targeted recovery run
{md_table(proto_180757_stability, ['candidate_group_key', 'model_name', 'family_id', 'consistent_gate_pass', 'num_seeds_checked', 'any_seed_pass'], {'candidate_group_key': 'Candidate group', 'model_name': 'Model', 'family_id': 'Family', 'consistent_gate_pass': 'Consistent pass', 'num_seeds_checked': 'Seeds checked', 'any_seed_pass': 'Any seed passed'})}

### Table 31. Longer stabilization sweep
{md_table(proto_200922_global, ['candidate_key', 'model_name', 'family_id', 'gate_pass', 'gate_failure_category', 'worst_clean_fpr', 'worst_attacked_benign_fpr', 'worst_adv_malicious_recall', 'mean_clean_f1', 'mean_robust_f1', 'final_rank'], {'candidate_key': 'Candidate', 'model_name': 'Model', 'family_id': 'Family', 'gate_pass': 'Gate pass', 'gate_failure_category': 'Failure category', 'worst_clean_fpr': 'Worst clean FPR', 'worst_attacked_benign_fpr': 'Worst attacked benign FPR', 'worst_adv_malicious_recall': 'Worst adv. recall', 'mean_clean_f1': 'Mean clean F1', 'mean_robust_f1': 'Mean robust F1', 'final_rank': 'Rank'})}

### Table 32. Stability summary for the longer sweep
{md_table(proto_200922_stability, ['candidate_group_key', 'model_name', 'family_id', 'consistent_gate_pass', 'num_seeds_checked', 'any_seed_pass'], {'candidate_group_key': 'Candidate group', 'model_name': 'Model', 'family_id': 'Family', 'consistent_gate_pass': 'Consistent pass', 'num_seeds_checked': 'Seeds checked', 'any_seed_pass': 'Any seed passed'})}

These tables explain why the project did not promote an early CatBoost winner. Coarse candidates existed, but none survived the stability check. This is the exact gap that the later final stabilized run had to close.

## 14. The data audit justified more disciplined stability-aware selection

### Table 33. Train/test protocol balance from the audit
{md_table(data_audit_protocol_balance, ['split', 'protocol', 'attack', 'benign', 'total', 'benign_ratio', 'attack_ratio'], {'split': 'Split', 'protocol': 'Protocol', 'attack': 'Attack', 'benign': 'Benign', 'total': 'Total', 'benign_ratio': 'Benign ratio', 'attack_ratio': 'Attack ratio'})}

### Table 34. Protocol share shift from the audit
{md_table(data_audit_protocol_shift, ['protocol', 'train_rows', 'test_rows', 'train_share', 'test_share', 'share_diff_test_minus_train'], {'protocol': 'Protocol', 'train_rows': 'Train rows', 'test_rows': 'Test rows', 'train_share': 'Train share', 'test_share': 'Test share', 'share_diff_test_minus_train': 'Test minus train'})}

### Table 35. Largest audited feature shifts
{md_table(first_n(data_audit_feature_drift, 10), ['feature', 'mean_train', 'mean_test', 'std_train', 'std_test', 'mean_shift_abs', 'smd_like'], {'feature': 'Feature', 'mean_train': 'Train mean', 'mean_test': 'Test mean', 'std_train': 'Train std', 'std_test': 'Test std', 'mean_shift_abs': 'Absolute mean shift', 'smd_like': 'SMD-like'})}

The audit confirmed that WiFi dominance, MQTT test benign absence, and measurable feature drift were real structural pressures. That made stability-aware model selection more important, not less.

## 15. The first near-final answer: stable finalists without saved robust artifacts

### Table 36. Stabilized finalist global decision table
{md_table(proto_003108_global, ['candidate_key', 'model_name', 'family_id', 'gate_pass', 'gate_failure_category', 'worst_clean_fpr', 'worst_attacked_benign_fpr', 'worst_adv_malicious_recall', 'mean_clean_f1', 'mean_robust_f1', 'final_rank'], {'candidate_key': 'Candidate', 'model_name': 'Model', 'family_id': 'Family', 'gate_pass': 'Gate pass', 'gate_failure_category': 'Failure category', 'worst_clean_fpr': 'Worst clean FPR', 'worst_attacked_benign_fpr': 'Worst attacked benign FPR', 'worst_adv_malicious_recall': 'Worst adv. recall', 'mean_clean_f1': 'Mean clean F1', 'mean_robust_f1': 'Mean robust F1', 'final_rank': 'Rank'})}

### Table 37. Stability consistency for the finalist run
{md_table(proto_003108_stability, ['candidate_group_key', 'model_name', 'family_id', 'consistent_gate_pass', 'num_seeds_checked', 'any_seed_pass'], {'candidate_group_key': 'Candidate group', 'model_name': 'Model', 'family_id': 'Family', 'consistent_gate_pass': 'Consistent pass', 'num_seeds_checked': 'Seeds checked', 'any_seed_pass': 'Any seed passed'})}

### Table 38. Protocol comparison of the two stable finalists
{md_table(finalist_rows([('wifi', proto_003108_wifi), ('mqtt', proto_003108_mqtt), ('bluetooth', proto_003108_bt)], [('xgboost__E', 'xgboost', 'E'), ('catboost__C', 'catboost', 'C')]), ['protocol', 'candidate', 'clean_f1', 'clean_fpr', 'attacked_benign_fpr', 'adv_malicious_recall', 'robust_f1', 'gate_pass'], {'protocol': 'Protocol', 'candidate': 'Candidate', 'clean_f1': 'Clean F1', 'clean_fpr': 'Clean FPR', 'attacked_benign_fpr': 'Attacked benign FPR', 'adv_malicious_recall': 'Adv. malicious recall', 'robust_f1': 'Robust F1', 'gate_pass': 'Gate pass'})}

This run gave the first stable finalists: `xgboost__E` and `catboost__C`. But the robust artifacts were not saved. That meant the best analytical answer still was not a deployable answer. The only honest next step was to rerun the final pass with artifact persistence enabled.

## 16. The artifact-persistent final run promoted `catboost__E` as the deployment baseline

### Table 39. Final artifact-persistent global decision table
{md_table(proto_112105_global, ['candidate_key', 'model_name', 'family_id', 'gate_pass', 'gate_failure_category', 'worst_clean_fpr', 'worst_attacked_benign_fpr', 'worst_adv_malicious_recall', 'mean_clean_f1', 'mean_robust_f1', 'final_rank'], {'candidate_key': 'Candidate', 'model_name': 'Model', 'family_id': 'Family', 'gate_pass': 'Gate pass', 'gate_failure_category': 'Failure category', 'worst_clean_fpr': 'Worst clean FPR', 'worst_attacked_benign_fpr': 'Worst attacked benign FPR', 'worst_adv_malicious_recall': 'Worst adv. recall', 'mean_clean_f1': 'Mean clean F1', 'mean_robust_f1': 'Mean robust F1', 'final_rank': 'Rank'})}

### Table 40. Final stability consistency summary
{md_table(proto_112105_stability, ['candidate_group_key', 'model_name', 'family_id', 'consistent_gate_pass', 'num_seeds_checked', 'any_seed_pass'], {'candidate_group_key': 'Candidate group', 'model_name': 'Model', 'family_id': 'Family', 'consistent_gate_pass': 'Consistent pass', 'num_seeds_checked': 'Seeds checked', 'any_seed_pass': 'Any seed passed'})}

### Table 41. Final protocol-level robust metrics for `catboost__E`
{md_table(catboost_e_rows([('wifi', proto_112105_wifi), ('mqtt', proto_112105_mqtt), ('bluetooth', proto_112105_bt)]), ['protocol', 'selected_threshold', 'clean_f1', 'clean_fpr', 'attacked_benign_fpr', 'adv_malicious_recall', 'robust_f1', 'gate_pass'], {'protocol': 'Protocol', 'selected_threshold': 'Selected threshold', 'clean_f1': 'Clean F1', 'clean_fpr': 'Clean FPR', 'attacked_benign_fpr': 'Attacked benign FPR', 'adv_malicious_recall': 'Adv. malicious recall', 'robust_f1': 'Robust F1', 'gate_pass': 'Gate pass'})}

### Table 42. Final full-test routed prediction metrics
{md_table(proto_112105_test_metrics, ['scope', 'model', 'threshold', 'precision', 'recall', 'f1', 'fpr', 'roc_auc', 'pr_auc', 'tp', 'tn', 'fp', 'fn', 'n_rows'], {'scope': 'Scope', 'model': 'Model', 'threshold': 'Threshold', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR', 'roc_auc': 'ROC-AUC', 'pr_auc': 'PR-AUC', 'tp': 'TP', 'tn': 'TN', 'fp': 'FP', 'fn': 'FN', 'n_rows': 'Rows'})}

This run is the final answer because it combines all the requirements that earlier candidates failed to combine: protocol coverage, gate pass, stability, and saved artifacts. The final stability summary shows that only `catboost__E` remained an all-seed consistent pass. The coarse-ranked XGBoost families no longer mattered more than that stability fact.

The final routed full-test metrics show why that decision is defensible. The global routed `catboost__E` row retained near-perfect precision and recall, with extremely low false-positive behavior. The one mandatory caveat remains MQTT: because the MQTT test split contains no benign negatives, MQTT FPR and ROC-AUC are not estimable on that split and must be stated as unavailable rather than silently implied.

## 17. Core synthesis

The complete journey supports a stronger thesis claim than any single run could support. The project did not simply find a high-performing IoMT IDS. It built a reproducible decision process that started from data reality, treated protocol heterogeneity as first-order information, replaced clean-score optimism with leakguard and split controls, discovered that benign-side adversarial false positives were the real operational danger, and used multi-seed stability to prevent premature promotion of coarse winners.

### Table 43. Milestone comparison from first baseline to final deployment baseline
{md_table([
    {'phase': 'Reduced-sample baseline winner', 'run': 'baseline_models_stdlib_20260305_234858', 'model': baseline_logistic['model'], 'precision': baseline_logistic['precision'], 'recall': baseline_logistic['recall'], 'f1': baseline_logistic['f1'], 'fpr': baseline_logistic['fpr']},
    {'phase': 'First full-data GPU winner', 'run': 'full_gpu_models_20260306_001638', 'model': full_gpu_mlp['model'], 'precision': full_gpu_mlp['precision'], 'recall': full_gpu_mlp['recall'], 'f1': full_gpu_mlp['f1'], 'fpr': full_gpu_mlp['fpr']},
    {'phase': 'First HPO winner', 'run': 'full_gpu_hpo_models_20260306_134806', 'model': hpo_134806_mlp['model'], 'precision': hpo_134806_mlp['precision'], 'recall': hpo_134806_mlp['recall'], 'f1': hpo_134806_mlp['f1'], 'fpr': hpo_134806_mlp['fpr']},
    {'phase': 'Initial protocol-routed winner', 'run': 'full_gpu_hpo_models_20260306_153556', 'model': hpo_153556_xgb['model'], 'precision': hpo_153556_xgb['precision'], 'recall': hpo_153556_xgb['recall'], 'f1': hpo_153556_xgb['f1'], 'fpr': hpo_153556_xgb['fpr']},
    {'phase': 'Leakguard base run', 'run': 'full_gpu_hpo_models_20260306_195851', 'model': leakguard_ensemble['model'], 'precision': leakguard_ensemble['precision'], 'recall': leakguard_ensemble['recall'], 'f1': leakguard_ensemble['f1'], 'fpr': leakguard_ensemble['fpr']},
    {'phase': 'Final deployment baseline', 'run': 'full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105', 'model': 'catboost__E', 'precision': proto_112105_test_metrics[-1]['precision'], 'recall': proto_112105_test_metrics[-1]['recall'], 'f1': proto_112105_test_metrics[-1]['f1'], 'fpr': proto_112105_test_metrics[-1]['fpr']},
], ['phase', 'run', 'model', 'precision', 'recall', 'f1', 'fpr'], {'phase': 'Phase', 'run': 'Run', 'model': 'Model', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'fpr': 'FPR'})}

The most important conclusion is that the final promoted model was not chosen because it had the single flashiest coarse score. It was chosen because it was the only candidate that survived the full chain of evidence: data realism, protocol-aware evaluation, adversarial stress, stability, and artifact persistence.

The remaining sections deepen that claim with appendix-level support. They are included because a thesis defense rarely fails on the headline metric; it fails on whether the intermediate pivots, constraints, and caveats were documented carefully enough.
""").strip())

parts.append(dedent(f"""
## 18. Detailed decision log extracted from the project timeline

### Table 44. Decision log from observation to next move
{md_table(decision_log_rows, ['date', 'observation', 'decision', 'why_next'], {'date': 'Date', 'observation': 'Observed result', 'decision': 'Decision taken', 'why_next': 'Why that forced the next step'})}

This table matters because it shows that the project did not move randomly from one experiment to the next. Each pivot responded to a specific failure mode. The chronology begins with a data-imbalance problem, moves into a protocol-heterogeneity problem, then into a robustness problem, and finally into a stability-and-artifact problem. That sequence is exactly why the final thesis claim is stronger than a standard "best test F1" story.

Another way to read Table 44 is as a record of increasing methodological strictness. Early in the timeline the main question was whether the feature space could separate benign from malicious traffic. Midway through the timeline the question became whether the separation still held after leakage controls, protocol-aware routing, and adversarial perturbations. By the end, the question was whether any candidate still looked good when all of those filters were applied simultaneously and repeatedly across seeds.

## 19. Additional EDA details that shaped all later modeling choices

### Table 45. Split-level label balance
{md_table(eda_split_balance, ['split', 'rows', 'benign_rows', 'attack_rows', 'benign_ratio', 'attack_ratio'], {'split': 'Split', 'rows': 'Rows', 'benign_rows': 'Benign rows', 'attack_rows': 'Attack rows', 'benign_ratio': 'Benign ratio', 'attack_ratio': 'Attack ratio'})}

### Table 46. Most frequent concrete attack names
{md_table(first_n(eda_top_attack_names, 10), ['attack_name', 'rows'], {'attack_name': 'Attack name', 'rows': 'Rows'})}

### Table 47. Attack-family hotspots by protocol
{md_table(first_n(eda_attack_family_by_protocol, 12), ['attack_family', 'protocol_hint', 'rows'], {'attack_family': 'Attack family', 'protocol_hint': 'Protocol', 'rows': 'Rows'})}

Table 45 explains why threshold calibration could not be treated as a one-time generic choice. The benign ratio fell from {format_cell(eda_split_balance[0]['benign_ratio'], 'ratio')} in train to {format_cell(eda_split_balance[1]['benign_ratio'], 'ratio')} in test. In other words, the project was validating in one operating mix and deploying into another. That is exactly the kind of shift that inflates apparent performance when thresholds are selected carelessly.

Table 46 shows that "attack" was not a smooth, homogeneous class. It was dominated by specific DDoS attack names, several of them in the same family cluster. The implication for the thesis was straightforward: a model could look globally excellent while partly learning the signatures of a few dominant scenarios. That is why the project kept returning to per-family and per-protocol slices whenever a result looked too clean.

Table 47 adds the key protocol insight. WiFi contained the massive DDoS and DoS concentration, MQTT mixed benign with several flood-oriented attacks, and Bluetooth had a much smaller but structurally different distribution. That distributional asymmetry is the direct reason why protocol-aware routing became more than a convenience. It became the correct problem formulation.

## 20. Cross-stage metric progressions make the bottlenecks visible

### Table 48. Early protocol progression from baseline to first HPO
{md_table(early_protocol_progression(), ['protocol', 'baseline_model', 'baseline_f1', 'baseline_fpr', 'full_gpu_model', 'full_gpu_f1', 'full_gpu_fpr', 'first_hpo_model', 'first_hpo_f1', 'first_hpo_fpr'], {'protocol': 'Protocol', 'baseline_model': 'Baseline best model', 'baseline_f1': 'Baseline F1', 'baseline_fpr': 'Baseline FPR', 'full_gpu_model': 'Full GPU best model', 'full_gpu_f1': 'Full GPU F1', 'full_gpu_fpr': 'Full GPU FPR', 'first_hpo_model': 'First HPO best model', 'first_hpo_f1': 'First HPO F1', 'first_hpo_fpr': 'First HPO FPR'})}

### Table 49. Bluetooth from bottleneck to controlled slice
{md_table(bluetooth_progression(), ['stage', 'model', 'threshold', 'f1', 'fpr', 'note'], {'stage': 'Stage', 'model': 'Representative model', 'threshold': 'Threshold', 'f1': 'F1', 'fpr': 'FPR', 'note': 'Interpretation'})}

### Table 50. WiFi robustness trade-off across the hardening sequence
{md_table(wifi_robustness_progression(), ['stage', 'representative', 'clean_fpr', 'attacked_benign_fpr', 'malicious_side_recall', 'note'], {'stage': 'Stage', 'representative': 'Representative candidate', 'clean_fpr': 'Clean FPR', 'attacked_benign_fpr': 'Attacked benign FPR', 'malicious_side_recall': 'Malicious-side recall', 'note': 'Interpretation'})}

Table 48 shows that the three protocols were not improving in the same way. MQTT was strong almost immediately, WiFi improved in already-small margins, and Bluetooth remained the place where model family choice visibly changed the operational risk profile. That asymmetry is the strongest compact argument for why a single global threshold or family comparison would have been incomplete.

Table 49 makes the Bluetooth story concrete. The slice began with acceptable but still operationally expensive false-positive behavior under the best full-data clean model, then became nearly perfect under routing, and finally stayed controlled under leakguard and the final robust matrix. The important point is not that Bluetooth eventually reached FPR 0.0 in multiple tables. The important point is that it only became believable after the suspiciously easy version survived split repair and leak controls.

Table 50 captures the WiFi lesson that changed the entire second half of the thesis. Clean WiFi FPR was never the issue. The issue was benign-side attack sensitivity. Once the epsilon 0.1 benign query attack pushed attacked benign FPR to {format_cell(find_row([row for row in robust_query_protocol if row['attack_method'] == 'query_sparse_hillclimb_benign' and row['protocol_hint'] == 'wifi'], 'epsilon', '0.1')['fpr'], 'fpr')}, the project had to stop thinking in terms of clean-score optimization and start thinking in terms of controlled operating envelopes. That single transition is why the later campaign was framed around gate passing, not only ranking.

## 21. The robust-matrix campaign became a study in search design, not just model ranking

### Table 51. Evolution of the protocol-wide robust matrix campaign
{md_table(matrix_campaign_evolution(), ['run_id', 'focus', 'candidate_rows', 'gate_pass_candidates', 'stable_groups', 'rank_1_candidate', 'note'], {'run_id': 'Run', 'focus': 'Campaign focus', 'candidate_rows': 'Candidate rows', 'gate_pass_candidates': 'Gate-pass candidates', 'stable_groups': 'Stable groups', 'rank_1_candidate': 'Rank-1 candidate', 'note': 'Outcome'})}

### Table 52. Final robust family definitions
{md_table(family_definition_rows(), ['family_id', 'family_name', 'family_description', 'hardneg_weight', 'maladv_weight', 'extra_topk_per_epsilon'], {'family_id': 'Family', 'family_name': 'Name', 'family_description': 'Description', 'hardneg_weight': 'Hard-negative weight', 'maladv_weight': 'Malicious-adv weight', 'extra_topk_per_epsilon': 'Extra top-k per epsilon'})}

### Table 53. Final gate definitions
{md_table(gate_definition_rows(), ['gate', 'value', 'meaning'], {'gate': 'Gate', 'value': 'Value', 'meaning': 'Meaning'})}

Table 51 clarifies why the project kept running new matrix passes instead of choosing the earliest rank-1 candidate. The number of gate-pass candidates was not the problem after the search widened. The problem was that gate passing on a single seed was cheap, while stable gate passing across seeds with saved artifacts was rare. That is why the later runs look repetitive in directory structure but not in scientific meaning.

Table 52 makes the family logic explicit. Families A and B were basically precision- and balance-oriented references. Families C, D, and E were not arbitrary extra variants; they were targeted attempts to recover Bluetooth without breaking the global robust constraints. The final answer coming from family E is therefore consistent with the project history. The family itself was created because the bottleneck demanded it.

Table 53 is important thesis evidence because it states the acceptance criteria directly. A candidate had to stay under {format_cell(proto_112105_matrix_summary['gates']['clean_fpr_max'], 'fpr')} clean FPR, under {format_cell(proto_112105_matrix_summary['gates']['attacked_benign_fpr_max'], 'fpr')} attacked benign FPR, and above {format_cell(proto_112105_matrix_summary['gates']['adv_malicious_recall_min'], 'recall')} adversarial malicious recall. Once these constraints are written plainly, the later decision between XGBoost and CatBoost families becomes a constrained-selection question rather than a beauty contest.

## 22. Feasibility and realism constraints are part of the result, not just implementation detail

### Table 54. Final realism profile by protocol
{md_table(realism_summary_rows(), ['protocol', 'rows_used', 'locked_features', 'mutable_features', 'tot_num_ratio_low', 'tot_num_ratio_high', 'rate_ratio_low', 'rate_ratio_high'], {'protocol': 'Protocol', 'rows_used': 'Rows used', 'locked_features': 'Locked features', 'mutable_features': 'Mutable features', 'tot_num_ratio_low': 'Tot/num ratio low', 'tot_num_ratio_high': 'Tot/num ratio high', 'rate_ratio_low': 'Rate ratio low', 'rate_ratio_high': 'Rate ratio high'})}

### Table 55. Final data-cap and FPR-feasibility summary
{md_table(data_cap_rows(), ['protocol', 'label_name', 'rows', 'unique_files', 'largest_file_share', 'fallback_type', 'val_rows', 'fpr_resolution', 'floor_applied', 'rows_moved_to_train'], {'protocol': 'Protocol', 'label_name': 'Label', 'rows': 'Rows', 'unique_files': 'Unique files', 'largest_file_share': 'Largest file share', 'fallback_type': 'Fallback type', 'val_rows': 'Validation rows', 'fpr_resolution': 'FPR resolution', 'floor_applied': 'Floor applied', 'rows_moved_to_train': 'Rows moved to train'})}

### Table 56. Sampling profile of the realistic robustness campaign
{md_table(robust_sampling_rows(), ['stage', 'wifi_rows', 'mqtt_rows', 'bluetooth_rows', 'total_rows', 'per_protocol_target'], {'stage': 'Stage', 'wifi_rows': 'WiFi rows', 'mqtt_rows': 'MQTT rows', 'bluetooth_rows': 'Bluetooth rows', 'total_rows': 'Total rows', 'per_protocol_target': 'Per-protocol target'})}

Table 54 shows that the three protocols lived under different perturbation geometries. Bluetooth had {format_cell(realism_summary_rows()[-1]['locked_features'], 'rows')} locked features and only {format_cell(realism_summary_rows()[-1]['mutable_features'], 'rows')} mutable ones, far tighter than WiFi or MQTT. That is a practical explanation for why Bluetooth became both hard to recover and easy to overstate. The attack space itself was narrower.

Table 55 is where data reality becomes impossible to ignore. Bluetooth benign data needed a floor repair, and {format_cell(find_row([row for row in proto_112105_data_cap if row['protocol'] == 'bluetooth' and row['label'] == '0'], 'protocol', 'bluetooth')['rows_moved_to_train'], 'rows')} benign rows were moved back to train to satisfy the floor target. That is not a flaw to hide. It is exactly the kind of detail that determines whether a low-FPR claim is statistically meaningful or only numerically pretty.

Table 56 explains the computational compromises behind the robustness study. The project did not attack all available rows in every phase because that would have made iterative search infeasible. Instead it used large but controlled samples, and the final thesis should state that clearly. The key point is that the sampling was systematic and protocol-aware, not opportunistic.

## 23. Final stability and deployment evidence

### Table 57. Stability envelope of the final candidate groups
{md_table(stability_envelope_rows(proto_112105_stability_check), ['candidate_group', 'all_seed_gate_pass', 'seeds_checked', 'min_clean_fpr', 'max_clean_fpr', 'min_attacked_benign_fpr', 'max_attacked_benign_fpr', 'min_adv_malicious_recall', 'max_adv_malicious_recall', 'primary_fail_reason'], {'candidate_group': 'Candidate group', 'all_seed_gate_pass': 'All-seed gate pass', 'seeds_checked': 'Seeds checked', 'min_clean_fpr': 'Min clean FPR', 'max_clean_fpr': 'Max clean FPR', 'min_attacked_benign_fpr': 'Min attacked benign FPR', 'max_attacked_benign_fpr': 'Max attacked benign FPR', 'min_adv_malicious_recall': 'Min adv. recall', 'max_adv_malicious_recall': 'Max adv. recall', 'primary_fail_reason': 'Primary fail reason'})}

### Table 58. Stable candidate shift between the near-final and final passes
{md_table(stable_candidate_shift_rows(), ['run_id', 'consistent_pass_groups', 'stable_candidates', 'artifact_persistence', 'decision'], {'run_id': 'Run', 'consistent_pass_groups': 'Consistent-pass groups', 'stable_candidates': 'Stable candidates', 'artifact_persistence': 'Artifacts saved', 'decision': 'Decision'})}

### Table 59. Final deployment-readiness facts
{md_table([
    {'dimension': 'Escalation recommendation', 'value': proto_112105_escalation['recommendation'], 'implication': 'The final run judged a stable candidate ready rather than requiring another search cycle.'},
    {'dimension': 'Protocol hard-cap trigger', 'value': 'No' if not proto_112105_escalation['triggered'] else 'Yes', 'implication': 'No emergency protocol cap was needed in the final campaign.'},
    {'dimension': 'Stable all-seed candidate', 'value': ', '.join(row['candidate_group_key'] for row in proto_112105_stability if is_true(row['consistent_gate_pass'])), 'implication': 'Only one candidate survived the full stability filter.'},
    {'dimension': 'External benign augmentation', 'value': proto_112105_external_benign[0]['enabled'], 'implication': 'The final deployment story stays inside the project dataset rather than mixing unmatched external data.'},
], ['dimension', 'value', 'implication'], {'dimension': 'Dimension', 'value': 'Value', 'implication': 'Implication'})}

Table 57 is arguably the most important appendix table in the whole document. It shows the difference between a candidate that can pass once and a candidate that can keep passing. `catboost__E` stayed inside the clean and attacked benign FPR envelope while preserving adversarial malicious recall across all checked seeds. `xgboost__A` and `xgboost__C` failed for the same reason repeatedly: FPR drift. Once the failure mode is this consistent, the final promotion decision stops being subjective.

Table 58 captures the practical meaning of the March 14 rerun. The earlier stable shortlist was analytically interesting but incomplete because the robust artifacts were not saved. The final pass both changed the stable set and closed the artifact gap. That is why the thesis can end with a deployable baseline rather than only a recommendation for future reruns.

Table 59 closes the loop from analysis to operational readiness. The escalation recommendation was `stable_candidate_ready`, no protocol cap was triggered, and the only consistent all-seed candidate was `catboost__E`. In a thesis context, this is the correct level of evidence for claiming that model selection is finished under the current rules.

## 24. Explicit caveats for thesis wording

### Table 60. Caveats that should be stated explicitly in the thesis chapter
{md_table(thesis_caveat_rows(), ['caveat', 'evidence', 'implication'], {'caveat': 'Caveat', 'evidence': 'Evidence', 'implication': 'Implication for thesis wording'})}

This table should not be treated as a weakness list. It is part of the scientific discipline of the chapter. A thesis becomes stronger, not weaker, when it states exactly where the evidence is clean and where it is structurally limited. In this project the biggest non-negotiable caveat is MQTT test benign absence, because it affects which metrics can be estimated honestly.

The Bluetooth caveat is different. Bluetooth is not missing benign negatives, but its benign evidence is concentrated enough that floor repairs and file concentration must be acknowledged. That does not invalidate the result. It means the result should be presented as carefully controlled within the available Bluetooth evidence, not as a universal statement about arbitrary unseen Bluetooth environments.

The final caveat is scope. Because external benign augmentation stayed disabled in the final run, the final baseline should be described as the best in-project deployment candidate rather than as a domain-generalized IDS. That distinction matters, and the artifacts support making it clearly.

## 25. Engineering interventions that preserved scientific validity

### Table 61. Engineering interventions that changed what experiments were feasible
{md_table(engineering_intervention_rows, ['date', 'intervention', 'why', 'effect'], {'date': 'Date', 'intervention': 'Intervention', 'why': 'Why it was needed', 'effect': 'Effect on the scientific campaign'})}

This final table belongs in the document because the project reached a point where software-engineering quality and research validity became inseparable. A stability bug, a nondeterministic seed path, or a broken artifact-saving default would not merely be "implementation issues." They would change which model appears to win.

The cleanest way to summarize the entire journey is therefore this: the final `catboost__E` result is not only the output of model training. It is the output of a progressively tightened research process. The process learned from imbalance, from protocol heterogeneity, from adversarial failure, from runtime constraints, and from stability failures. That is why the final answer is defensible.
""").strip())

OUT_PATH.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
print(f"Wrote {OUT_PATH}")
