Project Context for Codex (READ THIS BEFORE CODING)

One-liner: Build a flow-based intrusion detection model (IDS) for IoMT healthcare networks, trained on tabular flow features (not packet payloads), with attention to class imbalance, operational false positives, adversarial robustness, and explainability.

What we are building (technical deliverables)

A reproducible ML training + evaluation pipeline for binary classification: benign vs malicious using CICIoMT2024 extracted flow features.

Candidate models to implement + compare:

Baselines: Logistic Regression, Random Forest

Main candidates: Gradient Boosting (XGBoost / LightGBM)

Optional: small MLP if it’s clearly beneficial (but boosting is expected to be strong on tabular data).

Evaluation must reflect real IDS tradeoffs:

Prioritize precision/recall tradeoffs, FPR, PR-AUC, F1, confusion matrix.

Include threshold policy selection (example: choose a threshold that targets FPR ≤ 1% and report resulting recall/precision; optionally compare to “maximize F1”).

Robustness testing:

Evaluate degradation under adversarial evasion on flow features (realistic constraints like non-negativity / plausible bounds).

Track attack metrics like attack success rate and ΔF1 / robust recall (exact implementation can evolve).

Explainability:

Provide global and per-instance explanations (e.g., feature importance / SHAP-style outputs) to support interpretability in a healthcare security setting.

Data + constraints (important)

We work at the flow/metadata level (source/dest, ports, protocol, counts/bytes/rates/timing summaries) — not packet payloads.

This is intentional: flow-based monitoring is more scalable and avoids PHI/payload issues (and works even when traffic is encrypted).

CICIoMT2024 includes traffic from multiple protocols (e.g., Wi-Fi, MQTT, BLE) and many attack types; expect heavy class imbalance (malicious ≫ benign).

Definition of “DONE” for the modeling work

There is a single, reproducible pipeline that can:

load/merge the dataset into one consistent table + label,

train multiple candidate models with consistent splits,

output a comparison report (metrics + curves),

select a “best” model + threshold policy,

run adversarial stress tests,

generate explainability artifacts,

save final model + preprocessing objects + a results summary.

Rules for Codex contributions (keep the project consistent)

Always prefer reproducibility:

fixed seeds, deterministic splits, clear train/val/test separation (avoid leakage).

Always log work:

if you modify code/results/decisions, append a Conversation Timeline entry in this file and update Right Now + Recent Changes.

Keep outputs “research-friendly”:

write results to /reports/ (or similar), save figures, save metrics as JSON/CSV, and save trained model artifacts.

If something is ambiguous, make a reasonable default and document it in Decisions.




# Start Here: Project + Conversation Context

Purpose: This is the first file to read in new sessions. It keeps a durable record of current work, decisions, and a timestamped conversation timeline.

How to use:
- Update `Right Now` whenever focus changes.
- Append all major work updates to `Recent Changes`.
- Add one entry per conversation/session in `Conversation Timeline`.
- Keep entries brief and factual.

## Required Logging Rule (All Conversations)
- Every new conversation that performs analysis, file edits, commands, or decisions MUST log its actions in this file.
- At minimum, each conversation MUST append one `Conversation Timeline` entry with: timestamp, user intent, actions taken, outcome, and next step.
- If technical changes were made, the conversation MUST also update: `Right Now`, `Recent Changes`, and `Session Summary (Most Recent)`.
- This file should be treated as the single source of truth for project continuity across sessions.

## Current Goal
- Objective: Build a reproducible flow-based IDS pipeline for CICIoMT2024 (benign vs malicious) with robust evaluation, thresholding, adversarial checks, and explainability artifacts.
- Done when: One deterministic pipeline can load/merge data, train/compare models, choose threshold policy, run robustness and explainability steps, and save final reports/artifacts.

## installing libraries rule
use venv to install libraries without issue

## Modeling Rules
Always use the GPUs available in this server, there should be no reason to use cpu for training.

Interpretation of obtained results is important. There should always be some findings and interpetation derived from reasoning based on the results. This can be used for jey infromation that might be helpfull int the future, why somethings mightve not worked as expected, etc. 

## reasoning and thinking rule
When running a job there is no need to wait until its finished running. You can stop thinking and end the conversation for me to wake you up at a later time to fetch the results mannually. 

This rule to stop thinking is true in general, to stop you from bugging out unless you are actively running scripts *not jobs, scripts* you should stop thinking and reasoning after sending a message


## Important rule
Read capstone quicstart on the capstone15 folder to understand the rules of this enviroment and how to correctly send jobs.
Before starting any modeling/training task, read `/home/capstone15/reports/EDA_FOR_MODELING.md` and follow its data-split/metric/threshold guidance.

## HPC Workflow Defaults (Smooth Operations)
- Always use Slurm jobs for long-running work:
  - `sbatch` for reproducible/background runs (preferred for data extraction/training).
  - `srun` for short interactive debugging only.
- For CPU-heavy preprocessing (like PCAP->CSV), prefer `cpu` partition batch jobs.
- For long GPU training jobs, use `gpu` partition and checkpoint frequently (jobs can be preempted).
- Use `/scratch/$USER` for high-I/O temp work (env builds, archive extraction, intermediates).
- Keep durable assets (code/results/final artifacts) in `/home` or `/data`, not `/scratch`.
- Standard monitoring/ops commands:
  - `squeue -u $USER`, `sacct -u $USER`, `sinfo`, `scancel <JOBID>`.
- Respect Capstone limits from quickstart:
  - Max per job: 64 CPUs, 128 GB RAM, 1 GPU.
  - Max running jobs: 4; max queued jobs: 8.


## Right Now
- Working on: XGBoost interpretability and operator UX are now implemented for protocol-routed models; explainability artifacts were generated from `reports/full_gpu_hpo_models_20260306_195851/` and a Streamlit IDS console is ready for use.
- Branch: main
- Files in focus: `scripts/xgb_protocol_ids_utils.py`, `scripts/generate_xgb_explainability_artifacts.py`, `scripts/generate_xgb_explainability_artifacts.sbatch`, `scripts/ids_xgb_interpretability_ui.py`, `reports/XGBOOST_INTERPRETABILITY_UI.md`, and `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/`.
- Blockers: Local interactive runtime still has `.venv` NumPy `X86_V2` CPU incompatibility; explainability generation is validated via Slurm with a scratch venv bootstrap.

## Recent Changes
- YYYY-MM-DD HH:MM (TZ) - Change:
  - Why:
  - Commands run:
  - Result:
- 2026-03-07 12:00 (Europe/Madrid) - Added XGBoost explainability pipeline + IDS UI and generated artifacts
  - Why: User chose per-protocol XGBoost models and requested local/global explainability plus real-world IDS user experience.
  - Commands run: re-read `/home/capstone15/context if youre an ai.md`, `reports/EDA_FOR_MODELING.md`, and latest run artifacts (`reports/full_gpu_hpo_models_20260306_195851/*`); added `scripts/xgb_protocol_ids_utils.py`, `scripts/generate_xgb_explainability_artifacts.py`, `scripts/ids_xgb_interpretability_ui.py`, `scripts/generate_xgb_explainability_artifacts.sbatch`, and `reports/XGBOOST_INTERPRETABILITY_UI.md`; syntax-checked (`python3 -m py_compile`, `bash -n`); submitted explainability jobs `82`, `83`, `84` via `sbatch`; inspected `squeue`, `sacct`, and `reports/ids-xgb-explain_*.log`; patched threshold parser bug in `load_thresholds_by_protocol` and regenerated artifacts.
  - Result: Job `84` completed successfully and produced explainability bundle at `/home/capstone15/reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/` including protocol global feature rankings, local TP/TN/FP/FN case explanations, reference rows for UI, and summary/manifest outputs aligned to XGBoost thresholds.
- 2026-03-06 19:58 (Europe/Madrid) - Added leakguard trainer + submitted GPU job `81`
  - Why: User requested deep leakage audit after suspiciously strong results and asked for a new script that fixes any inflation risks.
  - Commands run: re-read `/home/capstone15/context if youre an ai.md`; re-read `/home/capstone15/reports/EDA_FOR_MODELING.md`; extracted quickstart text from `/home/capstone15/capstone-quickstart-v1.1 (1).pdf` via `gs`; audited `scripts/train_hpo_gpu_models_fprfix.py` and latest artifacts (`metrics_summary.csv`, `slice_metrics_protocol.csv`, `slice_metrics_attack_family.csv`) plus raw schema/count checks from merged train/test CSVs; created `scripts/train_hpo_gpu_models_leakguard.py` with schema leakage guards, train/test `source_relpath` disjointness assertions, file-level-first validation split fallback diagnostics, validation-only model ranking, and single-class slice flags; created `scripts/train_hpo_gpu_models_leakguard.sbatch`; syntax-checked (`python3 -m py_compile`); submitted `sbatch /home/capstone15/scripts/train_hpo_gpu_models_leakguard.sbatch`; verified startup via `squeue -j 81`, `sacct -j 81`, and `tail /home/capstone15/reports/ids-hpo-leakguard_81.log`.
  - Result: New hardened job `81` is `RUNNING` on `gpu` partition (`haskell`) with dependencies loaded and training starting; selection leakage path is removed and integrity checks are now enforced in-code.
- 2026-03-06 18:10 (Europe/Madrid) - Fixed protocol split collapse and resubmitted FPR-fix GPU job
  - Why: User reported that the completed run produced only WiFi results and requested necessary fixes plus fresh job submission per quickstart workflow.
  - Commands run: re-read `/home/capstone15/context if youre an ai.md`; extracted quickstart instructions from `/home/capstone15/capstone-quickstart-v1.1 (1).pdf` via `gs`; inspected `/home/capstone15/reports/full_gpu_hpo_models_20260306_172441/metrics_summary.json` and `/home/capstone15/reports/ids-hpo-fprfix_79.log`; audited raw protocol/class counts from `metadata_train.csv`; patched `scripts/train_hpo_gpu_models_fprfix.py` to replace `build_val_mask` with protocol+class-aware `build_protocol_class_val_mask` and deterministic fallback split logic; syntax-checked with `python3 -m py_compile`; submitted `sbatch /home/capstone15/scripts/train_hpo_gpu_models_fprfix.sbatch`; validated startup via `squeue -j 80`, `sacct -j 80`, and `tail reports/ids-hpo-fprfix_80.log`.
  - Result: Root cause confirmed (`mqtt val_rows=0`, `bluetooth` val single-class) and fixed in code; replacement job `80` is running on `gpu` partition (`haskell`) with GPU allocated and bootstrap complete.
- 2026-03-06 17:24 (Europe/Madrid) - Submitted FPR-fix HPO GPU job via quickstart-compliant Slurm batch flow
  - Why: User requested submitting `scripts/train_hpo_gpu_models_fprfix.py` as a job according to the capstone quickstart guidance and to read project context first.
  - Commands run: re-read `/home/capstone15/context if youre an ai.md`; extracted quickstart guidance from `/home/capstone15/capstone-quickstart-v1.1 (1).pdf` via `gs`; added launcher `/home/capstone15/scripts/train_hpo_gpu_models_fprfix.sbatch`; submitted `sbatch /home/capstone15/scripts/train_hpo_gpu_models_fprfix.sbatch`; checked status via `squeue -j 79` and `sacct -j 79`; inspected startup log `reports/ids-hpo-fprfix_79.log`.
  - Result: Slurm job `79` accepted and running on `gpu` partition (`haskell`) with `gres=gpu:1`; bootstrap confirms CUDA visibility and training dependencies are active; live log at `/home/capstone15/reports/ids-hpo-fprfix_79.log`.
- 2026-03-06 17:09 (Europe/Madrid) - Added FPR-focused HPO trainer variant with leakage + class-balance fixes
  - Why: User requested a new script implementing four targeted fixes to reduce false positives (especially Bluetooth): file-level validation split, tree-model class weighting, stronger FPR-aware objective, and benign-preserving tuning subsampling.
  - Commands run: re-read `/home/capstone15/context if youre an ai.md`; re-read `/home/capstone15/reports/EDA_FOR_MODELING.md`; re-read quickstart PDF via `gs` text extraction; copied script (`cp scripts/train_hpo_gpu_models.py scripts/train_hpo_gpu_models_fprfix.py`); patched target functions/constructors in the new script; syntax-check (`python3 -m py_compile scripts/train_hpo_gpu_models_fprfix.py`); verified edits via `grep -n`.
  - Result: New script `scripts/train_hpo_gpu_models_fprfix.py` now enforces file-group validation splitting, uses `scale_pos_weight` for XGBoost, `auto_class_weights="Balanced"` for CatBoost, `class_weight="balanced"` for LightGBM, applies a soft in-bound FPR penalty in validation objective, and switches tuning subsampling to preserve benign rows while downsampling attacks.
- 2026-03-06 15:36 (Europe/Madrid) - Submitted updated per-protocol HPO GPU training job
  - Why: User requested launching the train HPO GPU models run after syncing context + quickstart workflow.
  - Commands run: re-read `/home/capstone15/context if youre an ai.md`; re-read `/home/capstone15/reports/EDA_FOR_MODELING.md`; verified launcher settings in `scripts/train_hpo_gpu_models.sbatch`; submitted `sbatch scripts/train_hpo_gpu_models.sbatch`; checked queue state (`squeue -j 77`) and startup log (`reports/ids-hpo-gpu_77.log`).
  - Result: Slurm job `77` is running on `gpu` partition (`haskell`) with log streaming to `/home/capstone15/reports/ids-hpo-gpu_77.log`; environment/bootstrap completed and training launch is in progress.
- 2026-03-06 15:18 (Europe/Madrid) - Per-protocol HPO trainer refactor completed
  - Why: User rejected the last run and requested training an individual model per protocol type instead of one global model.
  - Commands run: re-read `context if youre an ai.md` + `reports/EDA_FOR_MODELING.md`; inspected latest `slice_metrics_protocol.csv`; patched `scripts/train_hpo_gpu_models.py` to (1) train/tune per `protocol_hint`, (2) save per-protocol model artifacts, (3) route test predictions by protocol for global metrics, (4) emit `metrics_summary_per_protocol_models.csv`, (5) aggregate per-protocol HPO outputs; removed CatBoost GPU `rsm` search parameter; syntax-checked via `python3 -m py_compile`; attempted runtime `--help` in `.venv` and reconfirmed NumPy `X86_V2` host incompatibility.
  - Result: Script now supports protocol-specific model training end-to-end and reports both per-protocol and protocol-routed global results, with CatBoost search-space fix included; next action is submitting a new Slurm run with this updated trainer.
- 2026-03-06 14:33 (Europe/Madrid) - Latest HPO job results pulled and interpreted
  - Why: User requested checking the results from the latest job.
  - Commands run: confirmed latest artifacts by timestamp (`ls -lt`), checked Slurm state (`squeue -j 76`, `sacct -j 76`), and read result outputs (`RUN_SUMMARY.txt`, `metrics_summary.csv`, `slice_metrics_protocol.csv`, `slice_metrics_attack_family.csv`, `hpo_trials.csv`, `ensemble_weights.json`) plus `tail` of `reports/ids-hpo-gpu_76.log`.
  - Result: Job `76` completed; best-by-selection is `complex_mlp_tuned` (test F1 `0.99798`, test FPR `0.0601`) while `xgboost_tuned`/ensemble have higher F1 (`0.99805`) but materially higher test FPR (`0.0976`); benign false positives remain concentrated in Bluetooth; CatBoost HPO trials failed due GPU `rsm` constraint.
- 2026-03-06 13:41 (Europe/Madrid) - Iterative GPU HPO pipeline added and launched
  - Why: User requested a major fine-tuning run to aggressively optimize for both low FPR and high F1, with iterative tuning and no waiting for completion.
  - Commands run: implemented `scripts/train_hpo_gpu_models.py` (coarse+refine HPO for XGBoost/CatBoost/LightGBM/MLP, constrained threshold objective, final refit, and ensemble tuning), added launcher `scripts/train_hpo_gpu_models.sbatch`, syntax-checked with `python3 -m py_compile`, submitted `sbatch` job `75`, detected `PartitionTimeLimit`, reduced walltime, canceled `75`, resubmitted as job `76`, and verified `RUNNING` status + bootstrap log creation.
  - Result: New long-running optimization experiment is active as Slurm job `76` with output log at `/home/capstone15/reports/ids-hpo-gpu_76.log`; run is configured to produce tuned model artifacts and full metric/slice reports under a new `reports/full_gpu_hpo_models_*` output directory.
- 2026-03-06 12:12 (Europe/Madrid) - Latest full-data run metrics re-pulled and interpreted
  - Why: User requested the latest modeling run results with interpretation.
  - Commands run: checked latest report directory timestamps (`ls -lt`, `find`), confirmed Slurm completion (`sacct -j 74`), read result artifacts (`RUN_SUMMARY.txt`, `metrics_summary.csv/json`, protocol/family slice CSVs), and reviewed `reports/ids-full-gpu_74.log` warnings.
  - Result: Confirmed latest run is `reports/full_gpu_models_20260306_001638/`; `complex_mlp` leads global F1 while all models exhibit materially higher test FPR than validation-selected `<=1%` target, with most false positives concentrated in Bluetooth benign traffic.
- 2026-03-06 11:59 (Europe/Madrid) - Full context/rules/logs/artifacts resynchronized
  - Why: User requested full continuity recovery from a broken thread before continuing modeling.
  - Commands run: Re-read context + rules (`context if youre an ai.md`, `EDA_FOR_MODELING.md`, quickstart PDF extraction), reviewed referenced scripts/logs/report artifacts, and validated merged dataset schema/header in `metadata_train.csv`/`metadata_test.csv`.
  - Result: Modeling context is fully synchronized with current project state and ready for the next experiment iteration.
- 2026-03-06 00:22 (Europe/Madrid) - Full-data GPU results pulled and verified
  - Why: User requested immediate results retrieval after job completion.
  - Commands run: Slurm status checks (`sacct`) and artifact reads (`metrics_summary.csv/json`, slice metric CSVs, job log tail).
  - Result: Job `74` completed in `00:16:06`; output bundle confirmed at `/home/capstone15/reports/full_gpu_models_20260306_001638/` with global and slice-level metrics.
- 2026-03-06 00:07 (Europe/Madrid) - Submitted full-data GPU modeling job (XGBoost/CatBoost/complex NN)
  - Why: User requested non-subset training on all data with explicit GPU usage and no corner-cutting.
  - Commands run: implemented `scripts/train_full_gpu_models.py` and `scripts/train_full_gpu_models.sbatch`; submitted `sbatch`; verified GPU allocation and startup through `squeue`, `scontrol`, and log checks.
  - Result: Slurm job `74` is running on `gpu` partition with `gres/gpu=1`, bootstrap shows `CUDA_VISIBLE_DEVICES=0`, and model/dependency setup is progressing in `/home/capstone15/reports/ids-full-gpu_74.log`.
- 2026-03-05 23:57 (Europe/Madrid) - Baseline model run executed through Slurm with GPU allocation
  - Why: User requested immediate baseline training across multiple model families and required adherence to quickstart job workflow with GPU usage.
  - Commands run: environment install attempts in `.venv` (`pip install numpy pandas scikit-learn`), Slurm submission/monitoring (`sbatch`, `squeue`, `scontrol`, `sacct`, `scancel`), and baseline trainer execution via `scripts/train_baseline_models_stdlib.py`.
  - Result: Completed Slurm job `73` on `gpu` partition (`gres/gpu=1`) and generated baseline report artifacts at `/home/capstone15/reports/baseline_models_stdlib_20260305_234858/` including global metrics, protocol/family slice metrics, and threshold-selected results.
- 2026-03-05 21:09 (Europe/Madrid) - Deep data inventory + README review
  - Why: User requested complete understanding of repository `data` contents before modeling work.
  - Commands run: file inventory (`find`/`wc`/`stat`), PDF extraction (`gs` text + image render), CSV schema checks (`head` + `python3` metadata scan), archive manifest checks (`unzip -Z -1`), pairing checks (CSV vs PCAP).
  - Result: Confirmed 255 files total (114 CSV, 128 PCAP, 5 PDF, 6 archives, 1 log, 1 md), one shared 45-column CSV schema across all CSVs, WiFi/MQTT attack train/test + profiling mappings, and Bluetooth PCAP-only coverage.
- 2026-03-05 21:39 (Europe/Madrid) - Added Bluetooth PCAP-to-CSV conversion script
  - Why: User requested generating Bluetooth CSVs with the same methodology family used for WiFi/MQTT.
  - Commands run: script implementation, CLI/syntax checks (`python3 -m py_compile`, `--help`), dependency-install attempts for `dpkt`.
  - Result: New converter script created at `scripts/build_bluetooth_csv_from_pcap.py` with default Bluetooth window size 10 and WiFi/MQTT-compatible 45-column output schema; full execution pending local `dpkt` availability.
- 2026-03-05 22:08 (Europe/Madrid) - Executed Bluetooth conversion and validated outputs
  - Why: User requested running the script and checking generated data.
  - Commands run: full conversion run with `.venv` Python and `--overwrite`, plus header/row-count/finiteness checks across generated CSV files.
  - Result: Generated 14 Bluetooth CSV files (`attacks` + `profiling`) totaling 155,076 rows from 1,550,805 packets; headers match WiFi/MQTT schema exactly; sampled rows are numeric/fine.
- 2026-03-05 22:38 (Europe/Madrid) - Metadata-aware full merge into train/test CSV outputs
  - Why: User requested joining all CSV datasets with folder/filename-derived metadata and binary attack labels.
  - Commands run: implemented and executed `scripts/merge_ciciomt_with_metadata.py` over all 128 source CSVs.
  - Result: Created `data/ciciomt2024/merged/metadata_train.csv` and `metadata_test.csv` with metadata columns (protocol/device/scenario/attack parsing + split provenance + labels). Final counts: rows_total=9,320,045, rows_train=7,669,892, rows_test=1,650,153, attack_train=7,067,938, attack_test=1,601,745.
- 2026-03-05 23:14 (Europe/Madrid) - Advanced EDA report with plots/tables/interpretations
  - Why: User requested thesis-oriented advanced EDA deliverable with visualizations, tabulated results, and iterative deeper exploration.
  - Commands run: implemented and executed `scripts/generate_advanced_eda_report.py` over merged train/test data.
  - Result: Produced report package at `reports/eda_advanced_20260305_231027/` with 8 SVG plots, 11 CSV tables, and a narrative report (`REPORT.md`) covering class imbalance, protocol/attack composition, split drift, feature separability, and family-by-protocol deep dive.
- 2026-03-05 23:22 (Europe/Madrid) - Added modeling-focused EDA guidance file + pre-modeling read rule
  - Why: User requested a detailed EDA-to-modeling document and explicit context instruction to read it before modeling.
  - Commands run: created `reports/EDA_FOR_MODELING.md`; updated context rules/sections to enforce pre-modeling review.
  - Result: A modeling playbook now exists with concrete split, metrics, thresholding, slicing, robustness, and explainability guidance derived from EDA.

## Decisions
- Decision:
  - Reason:
  - Alternatives considered:
- Decision: Treat the current modeling-ready tabular base as WiFi+MQTT CSV features; keep Bluetooth as future PCAP-to-feature extraction scope.
  - Reason: No Bluetooth CSV features are present in the current extracted tree.
  - Alternatives considered: Attempt on-the-fly PCAP parsing now (deferred to keep immediate IDS modeling scope focused and reproducible).
- Decision: Implement an in-repo Bluetooth feature extractor script based on documented PCAP+DPKT + fixed-window averaging methodology.
  - Reason: No official conversion script is included in the extracted dataset tree, but methodology details are documented and needed immediately.
  - Alternatives considered: Wait for original authors' exact converter code (not available locally), or proceed with no Bluetooth tabular data.
- Decision: Derive attack labels from folder+filename with conservative rule.
  - Reason: User requirement: any file in `attacks` folder is attack unless explicitly benign in its name; profiling files are benign.
  - Alternatives considered: Filename-only labeling (rejected because folder context is reliable and requested).
- Decision: Assign profiling files without explicit split to train/test via deterministic hash (20% test).
  - Reason: User requested two unified outputs (train/test), while profiling folders lack explicit split labels.
  - Alternatives considered: Put all profiling in train only (rejected for reduced benign coverage in test).
- Decision: Use stdlib baseline trainer for this run instead of sklearn baselines.
  - Reason: Installed `numpy/scikit-learn` wheels on Python 3.13 are incompatible with host CPU (`X86_V2` requirement), blocking library-based training.
  - Alternatives considered: Continue waiting on long pure-python local run (canceled), or build/compile a fully compatible scientific stack before this run (deferred due time).

## Next Steps
1. Use EDA findings to define training protocol: macro + per-protocol metrics, class weighting, and threshold policy at constrained FPR.
2. Build baseline training scripts (LogReg/RF/boosting) on `data/ciciomt2024/merged/metadata_train.csv` with evaluation on `metadata_test.csv`.
3. Add slice-level evaluation (protocol and attack-family confusion/recall) to verify robustness beyond dominant DDoS/WiFi patterns.

## Open Questions / Risks
- Should train/test naming mismatch for TCP_IP attacks be normalized to families before reporting per-attack metrics?
- Do we include profiling CSV captures in training (for benign/background context) or keep them as out-of-distribution evaluation only?
- How closely does the local reconstruction of the feature extraction logic match the original unpublished conversion implementation?
- Should covariance/variance behavior be further calibrated against the original authors' exact unpublished converter (if released)?
- Should profiling split ratio remain 20% test, or be tuned to target specific benign/attack balance?
- Given strong class imbalance and split drift, should threshold calibration be done per-protocol or globally with protocol-aware post-processing?
- Should the highly correlated top features (`Std`, `Radius`, `Max`, `Magnitue`) be regularized/reduced before adversarial robustness analysis?

## Environment / Runbook
- Start:
  - Login: `ssh <user>@10.205.20.10`
  - Activate env: `source /home/capstone15/.venv/bin/activate`
- Test:
  - Quick CPU interactive test: `srun --partition=interactive --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash`
- Build:
  - Submit batch job: `sbatch <job_script>.sh`
- Deploy:
- Env vars:

## References
- Ticket:
- PR:
- Docs:
- Commits:

## Conversation Timeline
- 2026-03-05 00:00 (Europe/Madrid) - Context file initialization
  - User intent: Create a top-level file that future conversations can inspect quickly and include a timeline log of conversations.
  - Actions taken: Created `000_START_HERE_CONTEXT.md` at `/home/capstone15` with project-context sections and timeline format.
  - Outcome: Persistent start-here context file now exists at directory root and is ready for ongoing updates.
  - Next session should start with: Review this file first, update `Right Now`, and append a new timeline entry.
- 2026-03-05 21:09 (Europe/Madrid) - Data directory deep exploration
  - User intent: Read repo context first, then deeply inspect `data` including PDF READMEs, CSV columns, and parquet presence.
  - Actions taken: Read context + `data/README.md`; parsed all 5 PDF READMEs (text extraction and image rendering for image-based PDFs); scanned all CSV headers and sampled values; inventoried PCAP/CSV trees and archive manifests; checked for parquet files.
  - Outcome: Complete structural/data-schema map established; no parquet files found in current tree or zip manifests; WiFi/MQTT CSVs are consistent 45-column flow-feature tables; Bluetooth subset is PCAP-only in this extraction.
  - Next session should start with: Build the dataset loader/labeling layer and lock split policy before model training.
- 2026-03-05 21:39 (Europe/Madrid) - Bluetooth PCAP-to-CSV methodology implementation
  - User intent: Build Bluetooth CSVs using the same methodology family as WiFi/MQTT.
  - Actions taken: Verified local scripts only perform archive extraction; implemented `scripts/build_bluetooth_csv_from_pcap.py` with DPKT-based packet parsing, flow-feature computation, and fixed-window averaging (default window 10 for Bluetooth); validated script syntax/CLI.
  - Outcome: Conversion tooling is now present in-repo; execution is ready once `dpkt` is available locally.
  - Next session should start with: Install `dpkt`, run converter across Bluetooth PCAPs, and inspect output CSV quality.
- 2026-03-05 22:08 (Europe/Madrid) - Bluetooth conversion execution + output QA
  - User intent: Run the converter and verify results.
  - Actions taken: Ran `scripts/build_bluetooth_csv_from_pcap.py` on all Bluetooth PCAP files with window size 10; optimized script to remove O(n^2) per-packet behavior; verified output file set, schema equality, row counts, and numeric finiteness.
  - Outcome: 14 CSV files successfully generated under `Bluetooth/attacks/csv/**` and `Bluetooth/profiling/csv/**`, matching expected naming and schema.
  - Next session should start with: Consume these Bluetooth CSV files in the unified IDS training pipeline.
- 2026-03-05 22:12 (Europe/Madrid) - HPC quickstart review for workflow guidance
  - User intent: Check whether using jobs/Slurm workflows would make future work smoother on this machine.
  - Actions taken: Read `capstone-quickstart-v1.1 (1).pdf` and extracted cluster workflow constraints (partitions, limits, storage tiers, batch vs interactive patterns).
  - Outcome: Confirmed jobs are the recommended path for smoother operation: `sbatch` for long/reproducible runs, `srun` for quick interactive debugging, `/scratch` for heavy I/O temp work.
  - Next session should start with: Convert recurring commands into reusable Slurm job scripts and checkpoint long GPU jobs.
- 2026-03-05 22:15 (Europe/Madrid) - Promoted HPC smooth-run guidance into persistent context
  - User intent: Ensure quickstart-based discoveries are clearly visible for future conversations.
  - Actions taken: Added explicit `HPC Workflow Defaults (Smooth Operations)` and concrete runbook commands in this context file.
  - Outcome: Future sessions now have immediate, prominent Slurm/storage/monitoring guidance without needing to re-open the quickstart PDF.
  - Next session should start with: Default to `sbatch` workflows for long tasks and reserve interactive sessions for short debugging.
- 2026-03-05 22:38 (Europe/Madrid) - Unified metadata merge of all CSV datasets
  - User intent: Join Bluetooth + WiFi/MQTT CSV datasets into train/test outputs while deriving as much metadata as possible from file/folder names and assigning binary attack labels.
  - Actions taken: Added `scripts/merge_ciciomt_with_metadata.py`, encoded folder/filename metadata rules, executed full merge across 128 CSV files, and validated resulting metadata columns/splits/labels.
  - Outcome: Two merged outputs were produced with rich metadata + original features: `data/ciciomt2024/merged/metadata_train.csv` and `metadata_test.csv`.
  - Next session should start with: Train/evaluate models on merged metadata datasets and iterate metadata parsing heuristics if needed.
- 2026-03-05 23:01 (Europe/Madrid) - Random row label validity audit against source counterparts
  - User intent: Cross-check whether row labels in merged outputs are valid by comparing random merged rows with their original source rows.
  - Actions taken: Ran a randomized audit over merged train/test rows (24 total), verified label rule consistency from path/filename metadata, verified split assignment, and compared sampled feature values back to original CSV rows using `source_relpath` + `source_row_index`.
  - Outcome: Audit passed with `24/24` rows valid (`0` failures); no label/split/feature mismatches found in sampled rows.
  - Next session should start with: Optionally run a larger-sample or full-key integrity audit and archive the report artifact.
- 2026-03-05 23:14 (Europe/Madrid) - Advanced EDA deliverable generation
  - User intent: Produce advanced EDA with plots, tabulated results, interpretations, and iterative deep exploration aligned to IDS thesis objective.
  - Actions taken: Built dependency-free EDA pipeline `scripts/generate_advanced_eda_report.py`, ran full metadata scan on merged train/test (9.32M rows), generated report tables/plots, and added deep dive for dominant family by protocol.
  - Outcome: Deliverable available at `reports/eda_advanced_20260305_231027/`; key findings include severe class imbalance (93.02% attack), DDoS dominance (65.42%), protocol concentration (WiFi-heavy), and notable train/test attack-rate drift (+4.91% in test).
  - Next session should start with: Convert EDA findings into model training/evaluation design (class-aware metrics, threshold calibration, per-slice reporting).
- 2026-03-05 23:22 (Europe/Madrid) - EDA-to-modeling playbook added
  - User intent: Create a detailed EDA file intended to guide modeling and enforce reading it before modeling tasks.
  - Actions taken: Authored `/home/capstone15/reports/EDA_FOR_MODELING.md` and added explicit context instruction requiring this document to be read before modeling/training.
  - Outcome: Future modeling sessions now have a concrete operational playbook aligned with EDA findings and thesis priorities.
  - Next session should start with: Use `EDA_FOR_MODELING.md` as the modeling execution checklist and baseline experiment plan.
- 2026-03-05 23:25 (Europe/Madrid) - Context prerequisites re-read before baseline modeling
  - User intent: Ensure all required context files are read before starting baseline IDS models.
  - Actions taken: Re-read `/home/capstone15/context if youre an ai.md`, extracted and reviewed `capstone-quickstart-v1.1 (1).pdf`, and reviewed `/home/capstone15/reports/EDA_FOR_MODELING.md`.
  - Outcome: Modeling constraints are reconfirmed (Slurm workflow limits, split/threshold policy, per-slice evaluation requirements, and artifact/logging expectations).
  - Next session should start with: Implement deterministic baseline training runs (LogReg/RF/boosting) with validation-selected `FPR <= 1%` thresholds and protocol/family slice metrics.
- 2026-03-05 23:57 (Europe/Madrid) - Baseline model matrix executed with Slurm GPU job
  - User intent: Train baseline models from different families for binary attack-vs-benign classification and run through proper quickstart workflow.
  - Actions taken: Implemented baseline scripts (`scripts/train_baseline_models_stdlib.py`, `scripts/train_baseline_models.py`), installed venv packages, detected CPU incompatibility for numpy/sklearn runtime, canceled local run, submitted Slurm jobs (`71`, `72`, `73`) and completed final GPU job.
  - Outcome: Baseline metrics and slice reports produced at `/home/capstone15/reports/baseline_models_stdlib_20260305_234858/`; best global F1 in this run was `logistic_sgd` with threshold selected under validation `FPR <= 1%`.
  - Next session should start with: Replace stdlib baselines with sklearn/XGBoost equivalents once a CPU-compatible Python environment is provisioned, then rerun full baseline matrix on larger sample caps.
- 2026-03-06 00:07 (Europe/Madrid) - Full-data GPU models launched for production baseline pass
  - User intent: Train base XGBoost, CatBoost, and a more complex NN on all rows (no subset), on GPU, and stop active monitoring after stable startup.
  - Actions taken: Added full-data GPU trainer and Slurm launcher, requested 1 GPU/16 CPU/120GB RAM, submitted `sbatch` job `74`, and verified initial runtime health from queue metadata + bootstrap log.
  - Outcome: Job established and running smoothly on `haskell`; log confirms GPU visibility and active environment/model bootstrap.
  - Next session should start with: After user confirms completion, parse `full_gpu_models_*` outputs and summarize global + slice metrics.
- 2026-03-06 00:22 (Europe/Madrid) - Full-data GPU run results extracted
  - User intent: "pull up the results"
  - Actions taken: Confirmed completion state, located latest output directory, and extracted summary/slice metrics from generated CSV/JSON artifacts.
  - Outcome: Best model by global F1 in this run is `complex_mlp`; metrics and slice reports are available for immediate review.
  - Next session should start with: Decide threshold calibration strategy (global vs protocol-specific) due high test FPR drift for boosted models.
- 2026-03-06 11:31 (Europe/Madrid) - Context and log resync after thread crash
  - User intent: Re-read all required context/rules and referenced files, then catch up on latest logs from previous conversations before continuing modeling.
  - Actions taken: Re-read `/home/capstone15/context if youre an ai.md`, `/home/capstone15/reports/EDA_FOR_MODELING.md`, and extracted quickstart guidance from `capstone-quickstart-v1.1 (1).pdf` via `gs`; reviewed `/home/capstone15/reports/ids-full-gpu_74.log`, `/home/capstone15/reports/full_gpu_models_20260306_001638/` metrics/slice CSVs, and local session index/log files.
  - Outcome: Context is fully synchronized; key status remains that `complex_mlp` leads global F1, while boosted models show substantially higher test FPR despite validation-time constrained thresholding.
  - Next session should start with: Execute the next modeling iteration with stricter FPR-control strategy (likely protocol-aware calibration) and robustness/explainability artifacts.
- 2026-03-06 11:59 (Europe/Madrid) - Full context/rules/files re-read for modeling continuation
  - User intent: Resume from a broken thread and require complete re-sync by reading context rules, logs, and all files referenced by the context file before proceeding.
  - Actions taken: Re-read `/home/capstone15/context if youre an ai.md`, `/home/capstone15/reports/EDA_FOR_MODELING.md`, `data/README.md`, and full quickstart PDF text extraction; reviewed referenced scripts (`scripts/train_full_gpu_models.py`, `scripts/train_full_gpu_models.sbatch`, `scripts/train_baseline_models_stdlib.py`, `scripts/train_baseline_models.py`, `scripts/merge_ciciomt_with_metadata.py`, `scripts/build_bluetooth_csv_from_pcap.py`, `scripts/generate_advanced_eda_report.py`), logs (`reports/ids-full-gpu_74.log`, `reports/ids-baselines_71.log`, `reports/ids-baselines_72.log`, `reports/ids-baselines_73.log`, `data/ciciomt2024_extract.log`), and report artifacts under `reports/full_gpu_models_20260306_001638/`, `reports/baseline_models_stdlib_20260305_234858/`, and `reports/eda_advanced_20260305_231027/`; validated merged-data schema/header from `metadata_train.csv` and `metadata_test.csv`.
  - Outcome: Session is fully synchronized with prior modeling state, constraints, and artifacts; no new modeling run started in this resync step.
  - Next session should start with: Implement the next modeling iteration focused on tighter test-time FPR control (likely protocol-aware calibration) plus robustness and explainability outputs.
- 2026-03-06 12:12 (Europe/Madrid) - Latest run results retrieval + interpretation
  - User intent: "pull up the results from the latest run and interpret them."
  - Actions taken: Verified newest report bundle and job status (`sacct -j 74`), re-read full metrics/slice artifacts and job log for run `full_gpu_models_20260306_001638`, and computed protocol-level false-positive concentration summary.
  - Outcome: `complex_mlp` remains best by global F1, but test FPR remains above validation target for all models (largest for XGBoost/CatBoost); false positives are dominated by Bluetooth benign traffic, and minority-family recall is significantly weaker for `complex_mlp` (especially Spoofing/Malformed).
  - Next session should start with: Run protocol-aware threshold calibration and evaluate tradeoffs versus global thresholding before adversarial robustness and explainability passes.
- 2026-03-06 13:00 (Europe/Madrid) - Context file re-read on user request
  - User intent: Read the AI context file in `/home/capstone15` and surface the current project context.
  - Actions taken: Located the requested file (`context if youre an ai.md`), read its full contents including rules, goals, runbook, and timeline sections, and extracted the active status and priorities.
  - Outcome: Context is synchronized for this session; no code/modeling changes were made.
  - Next session should start with: Continue from current `Right Now` focus (FPR-control iteration + robustness/explainability artifacts) unless user redirects.
- 2026-03-06 13:09 (Europe/Madrid) - Model-family expansion decision support
  - User intent: Decide whether to keep the latest three model families or test additional/new families.
  - Actions taken: Reviewed current tabular-ML evidence and mapped it to project constraints (large tabular data, IDS FPR sensitivity, GPU-first workflow); prioritized candidate families by expected ROI and implementation risk.
  - Outcome: Recommendation is to keep current three as strong core, add `LightGBM` as highest-priority missing family, and optionally test a modern tabular DL family (`TabM`) while deprioritizing small-data methods for full-data training.
  - Next session should start with: Implement one additional high-ROI family (`LightGBM`) and compare under the same thresholding/slice protocol before adding lower-priority families.
- 2026-03-06 13:41 (Europe/Madrid) - Iterative HPO job submission for low-FPR/high-F1 push
  - User intent: Launch a major, iterative hyperparameter-tuning job that pushes model classes toward better FPR/F1 tradeoffs and continue asynchronously (no waiting for completion).
  - Actions taken: Added `scripts/train_hpo_gpu_models.py` and `scripts/train_hpo_gpu_models.sbatch`; implemented two-stage HPO (coarse/random then refine), constrained validation objective (`FPR<=target` + F1), final refits, and validation-tuned weighted ensemble; submitted Slurm job `75`, diagnosed `PartitionTimeLimit`, corrected walltime, canceled job `75`, and resubmitted as job `76`.
  - Outcome: Job `76` is running on `gpu` partition (`haskell`), and bootstrap/dependency setup started successfully in `reports/ids-hpo-gpu_76.log`.
  - Next session should start with: Pull artifacts from the newest `reports/full_gpu_hpo_models_*` directory after job completion and compare against prior run `full_gpu_models_20260306_001638`.
- 2026-03-06 14:33 (Europe/Madrid) - HPO job `76` completion review
  - User intent: "check the results of the last job" after pointing to this context file.
  - Actions taken: Re-read context, validated job terminal state (`sacct -j 76`), collected latest metrics and slice artifacts from `reports/full_gpu_hpo_models_20260306_134806/`, and reviewed warning/error traces in `reports/ids-hpo-gpu_76.log`.
  - Outcome: Run completed successfully with `complex_mlp_tuned` selected by constrained objective; however test FPR remains above target (6.01%) and benign Bluetooth contributes most false positives; CatBoost search failed due invalid GPU `rsm` option set, while weighted ensemble collapsed to almost pure XGBoost.
  - Next session should start with: Patch CatBoost GPU HPO space (remove/guard `rsm`), then run protocol-aware threshold calibration to reduce Bluetooth false positives and re-check minority-family recall tradeoffs.
- 2026-03-06 15:18 (Europe/Madrid) - Implemented protocol-specific training pipeline
  - User intent: Replace global-model training with an individual model per protocol type to improve bad last-run results.
  - Actions taken: Re-read required context and modeling guidance (`context if youre an ai.md`, `reports/EDA_FOR_MODELING.md`), inspected `slice_metrics_protocol.csv`, patched `scripts/train_hpo_gpu_models.py` to train/tune by `protocol_hint` with protocol-routed global evaluation and per-protocol summary artifacts, removed CatBoost GPU `rsm` from HPO space, and ran `python3 -m py_compile` for syntax validation.
  - Outcome: Training script now learns separate protocol models and exports both per-protocol metrics plus global routed metrics; runtime import checks in current `.venv` remain blocked by known NumPy `X86_V2` incompatibility on this host.
  - Next session should start with: Submit a fresh Slurm job using the updated `train_hpo_gpu_models.py` and compare new Bluetooth false-positive rates versus run `76`.
- 2026-03-06 15:36 (Europe/Madrid) - Launched per-protocol HPO GPU job `77`
  - User intent: Launch the train HPO GPU models run by following local context and quickstart/job workflow.
  - Actions taken: Re-read context and modeling guidance (`context if youre an ai.md`, `reports/EDA_FOR_MODELING.md`), verified Slurm launcher configuration in `scripts/train_hpo_gpu_models.sbatch`, submitted `sbatch` job `77`, and checked queue/log startup status via `squeue` and `reports/ids-hpo-gpu_77.log`.
  - Outcome: Job `77` is active on `gpu` partition (`RUNNING`) and writing logs to `/home/capstone15/reports/ids-hpo-gpu_77.log`.
  - Next session should start with: After completion, parse the newest `reports/full_gpu_hpo_models_*` bundle and compare Bluetooth benign FPR and per-family recall against run `76`.
- 2026-03-06 17:09 (Europe/Madrid) - New FPR-fix trainer script created on request
  - User intent: Create a new training script version that addresses four structural causes of high false positives in the current HPO pipeline.
  - Actions taken: Re-read required context (`context if youre an ai.md`), modeling guidance (`reports/EDA_FOR_MODELING.md`), and quickstart PDF; copied `scripts/train_hpo_gpu_models.py` to `scripts/train_hpo_gpu_models_fprfix.py`; implemented (1) file-level `build_val_mask` hashing only `source_relpath`, (2) tree-model class balancing (`scale_pos_weight` XGBoost, `auto_class_weights="Balanced"` CatBoost, `class_weight="balanced"` LightGBM), (3) soft in-bound FPR penalty in `objective_from_val`, and (4) benign-preserving tuning subsampling in `stratified_subsample_indices`; validated syntax with `python3 -m py_compile`.
  - Outcome: New script is ready for use and directly encodes the requested four changes while leaving the original trainer unchanged.
  - Next session should start with: Launch a Slurm run using `scripts/train_hpo_gpu_models_fprfix.py` and compare Bluetooth benign FPR/recall tradeoffs against outputs from job `77`.
- 2026-03-06 17:24 (Europe/Madrid) - Submitted FPR-fix trainer as Slurm GPU job `79`
  - User intent: Submit `scripts/train_hpo_gpu_models_fprfix.py` as a proper job per capstone quickstart and read context first.
  - Actions taken: Re-read `/home/capstone15/context if youre an ai.md`; re-extracted quickstart job rules from `/home/capstone15/capstone-quickstart-v1.1 (1).pdf`; created `scripts/train_hpo_gpu_models_fprfix.sbatch` with `gpu` partition, `--gres=gpu:1`, log output, and project `cd`; submitted with `sbatch`; verified status (`squeue -j 79`, `sacct -j 79`) and log startup (`reports/ids-hpo-fprfix_79.log`).
  - Outcome: Job `79` is `RUNNING` on node `haskell` with GPU allocated and bootstrap completed.
  - Next session should start with: Pull the newest `reports/full_gpu_hpo_models_*` artifacts after job completion and compare FPR/recall tradeoffs against prior runs (`76`, `77`).
- 2026-03-06 18:10 (Europe/Madrid) - WiFi-only output root-cause fix + new submission
  - User intent: Investigate why the finished FPR-fix run returned only WiFi protocol results, fix the issue, and submit a corrected job following quickstart guidance.
  - Actions taken: Re-read context and quickstart PDF; inspected run `79` artifacts (`metrics_summary.csv/json`, log) and confirmed protocol-skip causes (`mqtt` had `val_rows=0`, `bluetooth` val split had one class); verified raw train class availability for all protocols from merged CSV; patched `scripts/train_hpo_gpu_models_fprfix.py` to use protocol+class-aware deterministic validation splitting with file-level primary partition and row-level fallback for collapsed strata; syntax-checked; submitted `sbatch` job `80`; verified `RUNNING` state and bootstrap log.
  - Outcome: Corrected trainer is now running as Slurm job `80` on `gpu` partition with valid startup; expected behavior is training/evaluating all available protocols rather than WiFi-only.
  - Next session should start with: After job `80` finishes, check `metrics_summary.json` for non-empty `per_protocol` entries for `wifi`, `mqtt`, and `bluetooth`, then compare FPR/recall deltas vs run `79`.
- 2026-03-06 19:58 (Europe/Madrid) - Leakage/optimism audit + leakguard run submission
  - User intent: Investigate if latest near-perfect results are inflated by leakage, patch the trainer accordingly, and optionally launch a new job.
  - Actions taken: Audited `scripts/train_hpo_gpu_models_fprfix.py` and latest result slices; validated schema positions and protocol/class distributions; verified train/test `source_relpath` disjointness from merged CSVs; implemented `scripts/train_hpo_gpu_models_leakguard.py` with strict schema guards for `feature_cols`, mandatory test `source_relpath` loading + overlap assertion, file-level-first protocol/class validation split with logged fallback events and optional strict row-fallback blocking, validation-only model selection (removed test-based ranking leakage), and explicit single-class slice flags in slice CSV/JSON outputs; added `scripts/train_hpo_gpu_models_leakguard.sbatch`; syntax-checked and submitted `sbatch` job `81`; confirmed running state and bootstrap log.
  - Outcome: New leakguard training job `81` is running on `gpu` (`haskell`) and will emit metrics where model ranking is validation-only and single-class slices are explicitly marked to prevent over-interpretation.
  - Next session should start with: After job `81` completes, inspect `metrics_summary.csv/json` for validation-based ranking consistency, `validation_split.fallback_events`, and `single_class_test_slices` before comparing against run `80`.
- 2026-03-07 12:00 (Europe/Madrid) - XGBoost interpretability + IDS UI implementation
  - User intent: After selecting the per-protocol XGBoost models, add local/global explainability and a practical UI for real-world IDS usage.
  - Actions taken: Re-read context + modeling guidance, inspected latest run outputs in `reports/full_gpu_hpo_models_20260306_195851/`, implemented reusable routing/explainability utilities (`scripts/xgb_protocol_ids_utils.py`), added artifact generator (`scripts/generate_xgb_explainability_artifacts.py`) and Streamlit app (`scripts/ids_xgb_interpretability_ui.py`), added Slurm launcher (`scripts/generate_xgb_explainability_artifacts.sbatch`) and usage guide (`reports/XGBOOST_INTERPRETABILITY_UI.md`), submitted jobs `82`/`83`/`84`, diagnosed dependency and threshold-mapping issues, patched and reran.
  - Outcome: Explainability artifacts are now generated at `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/` with per-protocol global importance tables + local case explanations; Streamlit IDS console supports live inference, batch scoring, global explanation views, and local case exploration.
  - Next session should start with: Launch the Streamlit UI in an environment with `streamlit` installed and optionally wire a live packet/flow ingestion source to feed batch scoring continuously.

## Session Summary (Most Recent)
- Date: 2026-03-07 (Europe/Madrid)
- Summary: Implemented the interpretability and UX layer for the selected protocol-routed XGBoost IDS models. Added shared utilities for model/threshold loading and routed predictions (`scripts/xgb_protocol_ids_utils.py`), a full explainability artifact generator (`scripts/generate_xgb_explainability_artifacts.py`), and an operator-facing Streamlit console (`scripts/ids_xgb_interpretability_ui.py`) with live inference, batch scoring, global feature-importance views, and local case explorer tabs. Added quickstart docs and Slurm launcher; after patching threshold parsing, job `84` completed and produced the artifact bundle under `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/`.
- Immediate next action: Launch the Streamlit app from an environment with `streamlit` available and validate UI flows with `reference_rows.csv` plus a sample uploaded CSV batch.
