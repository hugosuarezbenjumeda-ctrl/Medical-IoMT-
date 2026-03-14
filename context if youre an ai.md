Project Context for Codex (READ THIS BEFORE CODING)

One-liner: Build a flow-based intrusion detection model (IDS) for IoMT healthcare networks, trained on tabular flow features (not packet payloads), with attention to class imbalance, operational false positives, adversarial robustness, and explainability.

What we are building (technical deliverables)

A reproducible ML training + evaluation pipeline for binary classification: benign vs malicious using CICIoMT2024 extracted flow features.

Candidate models to implement + compare:

Baselines: Logistic Regression, Random Forest

Main candidates: Gradient Boosting (XGBoost / LightGBM)

Optional: small MLP if itâ€™s clearly beneficial (but boosting is expected to be strong on tabular data).

Evaluation must reflect real IDS tradeoffs:

Prioritize precision/recall tradeoffs, FPR, PR-AUC, F1, confusion matrix.

Include threshold policy selection (example: choose a threshold that targets FPR â‰¤ 1% and report resulting recall/precision; optionally compare to â€œmaximize F1â€).

Robustness testing:

Evaluate degradation under adversarial evasion on flow features (realistic constraints like non-negativity / plausible bounds).

Track attack metrics like attack success rate and Î”F1 / robust recall (exact implementation can evolve).

Explainability:

Provide global and per-instance explanations (e.g., feature importance / SHAP-style outputs) to support interpretability in a healthcare security setting.

Data + constraints (important)

We work at the flow/metadata level (source/dest, ports, protocol, counts/bytes/rates/timing summaries) â€” not packet payloads.

This is intentional: flow-based monitoring is more scalable and avoids PHI/payload issues (and works even when traffic is encrypted).

CICIoMT2024 includes traffic from multiple protocols (e.g., Wi-Fi, MQTT, BLE) and many attack types; expect heavy class imbalance (malicious â‰« benign).

Definition of â€œDONEâ€ for the modeling work

There is a single, reproducible pipeline that can:

load/merge the dataset into one consistent table + label,

train multiple candidate models with consistent splits,

output a comparison report (metrics + curves),

select a â€œbestâ€ model + threshold policy,

run adversarial stress tests,

generate explainability artifacts,

save final model + preprocessing objects + a results summary.

Rules for Codex contributions (keep the project consistent)

Always prefer reproducibility:

fixed seeds, deterministic splits, clear train/val/test separation (avoid leakage).

Always log work:

if you modify code/results/decisions, append a Conversation Timeline entry in this file and update Right Now + Recent Changes.

Keep outputs â€œresearch-friendlyâ€:

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
- Working on: Thesis-style chapter deliverables are now generated as both source markdown and Word document, with script-verified prose, fewer but more decision-relevant tables, and the final `catboost__E` selection presented as the deployable baseline.
- Branch: main
- Files in focus: `Thesis/THESIS_EXPERIMENT_CHAPTER.docx`, `Thesis/THESIS_EXPERIMENT_CHAPTER.md`, `Thesis/generate_thesis_experiment_chapter.py`, `context if youre an ai.md`, `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/*`.
- Blockers: No blocker for the chapter deliverable; only optional further prose expansion or university-specific formatting remains.
- Next decision pending: Accept the generated `.docx` as the working thesis chapter, or continue with a more formal university-template conversion / additional prose expansion.

## Recent Changes
- YYYY-MM-DD HH:MM (TZ) - Change:
  - Why:
  - Commands run:
  - Result:
- 2026-03-14 18:59 (Europe/Berlin) - Generated thesis-style chapter as markdown plus Word document
  - Why: User asked to turn the long chronology into an actual document, focus more on polished writing than table volume, and verify the narrative against the Python scripts that produced the results.
  - Commands run: created `Thesis/.venv-docs` and installed `python-docx`; reverse-engineered the merge, baseline, HPO, leakguard, robustness, WiFi hardening, rebalance, and protocol-matrix scripts; created `Thesis/generate_thesis_experiment_chapter.py`; rendered `Thesis/THESIS_EXPERIMENT_CHAPTER.md` and `Thesis/THESIS_EXPERIMENT_CHAPTER.docx`; verified generation with `py_compile`.
  - Result: The project now has a script-verified thesis chapter in both markdown and `.docx` form, with fewer but more decision-relevant tables and stronger prose around each experiment pivot and its next-step rationale.
- 2026-03-14 18:19 (Europe/Berlin) - Expanded the chronological thesis document into a fuller chapter with appendices
  - Why: User asked to continue and make the thesis document substantially longer, with more explicit chronology, more tables, and clearer explanation of why each experimental pivot led to the next.
  - Commands run: extended `Thesis/generate_chronological_experiment_journey.py` with extra report loads and appendix builders; regenerated `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md`; verified rendering and reviewed the added sections.
  - Result: The document now includes a dated decision log, extra EDA tables, cross-stage progression tables, robust-matrix campaign anatomy, feasibility/realism constraints, stability-envelope evidence, explicit thesis caveats, and engineering-intervention chronology.
- 2026-03-14 18:02 (Europe/Berlin) - Generated long-form chronological thesis document from project artifacts
  - Why: User requested a very long thesis document that follows the actual experiment timeline, explains each step and next step, and embeds result tables in chronological order.
  - Commands run: inventoried `reports/` directories and `context if youre an ai.md`; extracted EDA, baseline, GPU/HPO, robustness, WiFi hardening, rebalance, data-audit, and final robust-matrix result tables; created `Thesis/generate_chronological_experiment_journey.py`; rendered `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md`.
  - Result: New thesis-ready markdown narrative now exists at `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md` and covers the full path from data preparation to the final `catboost__E` deployment baseline with embedded tables.
- 2026-03-14 17:20 (Europe/Berlin) - Local test inference metrics generated for protocol-routed `catboost__E`
  - Why: User requested prediction-based metrics table on `metadata_test.csv` for the three protocol models with thresholds.
  - Commands run: loaded `wifi/mqtt/bluetooth` coarse `catboost__E` models from run `..._20260314_112105`, applied protocol-specific thresholds from decision tables, executed routed inference over full `data/merged/metadata_test.csv`, and computed `precision, recall, f1, fpr, roc_auc, pr_auc, tp, tn, fp, fn` locally.
  - Result: Saved metrics table at `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/catboost_E_test_predictions_metrics_with_thresholds.csv`; MQTT `fpr`/`roc_auc` are undefined because test MQTT has no benign negatives.
- 2026-03-14 17:07 (Europe/Berlin) - Completed analysis of run `..._20260314_112105` (artifact-persistent stability pass)
  - Why: User confirmed completion of the new robust-matrix run and requested continuation from results.
  - Commands run: inspected run tree and authoritative outputs (`decision_table_global.csv`, `stability_check_global.csv`, `stability_consistency_summary.csv`, `matrix_summary.json`, `escalation_recommendation.json`); verified saved model paths in decision-table `files.saved_model`.
  - Result: Run finished with persisted model artifacts; stability consistency shows only `catboost__E` as all-seed gate-pass (`xgboost__A` and `xgboost__C` fail stability due `fpr_only`).
- 2026-03-14 11:40 (Europe/Berlin) - Clarified robust-training semantics and metric coverage
  - Why: User asked whether models are still trained on general clean data (and not only attacks) and whether reported metrics include clean-data behavior.
  - Commands run: reviewed robust-matrix pipeline stages and run-output metric columns from decision/stability tables.
  - Result: Confirmed pipeline does clean warm-fit plus robust hard-negative augmentation/retrain; reported outputs include both clean metrics (`clean_f1`, `clean_fpr`) and attacked metrics (`attacked_benign_fpr`, `adv_malicious_recall`, `robust_f1`) with known limitation that clean precision/clean recall are not persisted in decision tables.
- 2026-03-14 11:25 (Europe/Berlin) - Interpreted final stabilized run and compared finalists by protocol
  - Why: User requested best-model decision between `xgboost__E` and `catboost__C` with protocol-level (non-aggregated) view.
  - Commands run: parsed `decision_table_global.csv`, `stability_check_global.csv`, `stability_consistency_summary.csv`, `matrix_summary.json`, `escalation_recommendation.json` in `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108`.
  - Result: Two all-seed stable pass groups were confirmed (`xgboost__E`, `catboost__C`); recommendation depends on preference (overall rank vs worst-case stability-side FPR margin).
- 2026-03-14 11:05 (Europe/Berlin) - Enabled robust candidate artifact saving by default
  - Why: User required saving trained robust models for reuse/deployment and not only reporting metrics.
  - Commands run: patched `scripts/train_protocol_multimodel_robust_matrix.py` (`--save-models` default true + `--no-save-models` opt-out), updated both robust-matrix sbatch launchers to pass `--save-models`, validated with `py_compile` and `.venv39` unit tests.
  - Result: New runs now persist robust candidate model artifacts unless explicitly disabled.
- 2026-03-13 12:58 (Europe/Berlin) - Completed data-first audit for robustness planning
  - Why: User requested inspecting data thoroughly before making additional script-level robustness/training changes.
  - Commands run: streamed full-train/full-test audits via `.venv39` on `data/merged/metadata_train.csv` + `metadata_test.csv`; computed protocol/label/family distributions, feature quality, protocol-share drift, top source-file concentration, and attack-stats/query-budget behavior from `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_121948`.
  - Result: New artifact bundle written to `reports/data_audit_20260313_124851` with summary tables/JSON; findings confirm severe imbalance/drift, heavy WiFi dominance, MQTT test benign=0, many near-constant features (especially protocol-specific), and robust-matrix query loops saturating budget (high CPU pressure).
- 2026-03-12 13:28 (Europe/Berlin) - Implemented Targeted V1 recovery changes + launcher/profile updates
  - Why: User requested direct implementation of the Targeted V1 recovery plan and emphasized avoiding CPU bottlenecks/long local runs.
  - Commands run: patched `scripts/train_protocol_multimodel_robust_matrix.py` (new CLI: `--attack-source-mode`, `--family-pack`, `--bluetooth-hardneg-max-fraction`; new attack scorer abstraction + stage-mode resolution; no-pass threshold fallback with `adv_shortfall`; bluetooth hardneg cap; recovery family pack with A/B/C/D/E; matrix summary metadata); patched `scripts/evaluate_xgb_robustness.py` (`run_query_sparse_hillclimb` now supports optional `score_margin_fn` callback); patched `scripts/train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch` (now `stage-mode both` + stability seeds + hybrid source + bluetooth recovery settings); added `scripts/test_train_protocol_multimodel_robust_matrix.py`; validated via `.venv39` `py_compile` and test run; attempted bluetooth smoke run and stopped locally per user request.
  - Result: Targeted V1 feature set is implemented and syntax/test checks pass locally; server-side full run is now the next step.
- 2026-03-12 00:08 (Europe/Berlin) - Fixed fast-run crash from function-signature mismatch
  - Why: Server run `ids-proto-mm-fast-c4_255.log` failed with `TypeError: _run_protocol_model_family_candidate() got an unexpected keyword argument 'query_score_batch_benign'`.
  - Commands run: patched `scripts/train_protocol_multimodel_robust_matrix.py` to add `query_score_batch_benign` and `query_score_batch_mal` to `_run_protocol_model_family_candidate(...)` signature; re-ran `py -3 -m py_compile` for `train_protocol_multimodel_robust_matrix.py` and `evaluate_xgb_robustness.py`.
  - Result: Call-site and function signature are aligned; script is syntax-valid and ready to restage/resubmit.
- 2026-03-11 21:55 (Europe/Berlin) - Implemented attack-loop acceleration for robust matrix runtime
  - Why: User requested concrete fixes for 20h+ runtime and low GPU utilization (CPU-bound query attack loop).
  - Commands run: patched `scripts/evaluate_xgb_robustness.py` to use `booster.inplace_predict` (fallback to `DMatrix`) and added row-batched query scoring in `run_query_sparse_hillclimb` via new `score_batch_rows` parameter; added CLI `--query-score-batch-rows` and wired callsite; patched `scripts/train_protocol_multimodel_robust_matrix.py` to expose/pass `--query-score-batch-rows` and `--val-malicious-score-batch-rows` through all campaign paths, include query profile details in logs/summary, and use `inplace_predict` in XGBoost model inference helper; updated `scripts/train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch` to reduce iterative step loops and increase per-step candidate batch (`steps 12`, `cands 24`, `score-batch-rows 96`) and lowered threshold grid to `120`; validated syntax with `py -3 -m py_compile`.
  - Result: Next run will execute the same coarse all-model matrix with substantially fewer attack-loop iterations and larger batched scoring calls, which should materially cut wall-clock and raise GPU work per scoring call.
- 2026-03-11 10:35 (Europe/Berlin) - Implemented Cycle 3 runtime-cut profile + candidate timing observability
  - Why: User requested a hard runtime reduction from ~20-40h behavior while keeping all 4 model families and all 3 protocols.
  - Commands run: patched `scripts/train_protocol_multimodel_robust_matrix.py` to emit per-candidate phase timings (`warm_fit`, `attack_gen`, `robust_fit`, `threshold_eval`, `total`) and completion log line (`candidate_done[...]`); added new launcher `scripts/train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch` with coarse-only locked profile (all 4 models, reduced samples/budgets, `cpus-per-task=32`, thread env exports, `SKIP_PIP_INSTALL=1` default, `EXTRA_ARGS` override); validated with `py -3 -m py_compile`.
  - Result: New runnable fast path exists without modifying legacy launcher behavior, and logs now surface per-candidate phase duration to diagnose silent stretches.
- 2026-03-10 13:05 (Europe/Berlin) - Cross-source dataset compatibility audit for robustness extension
  - Why: User requested checking whether current CICIoMT-trained models can be evaluated/adapted on external dataset `https://github.com/imfaisalmalik/IoT-Healthcare-Security-Dataset`.
  - Commands run: re-read `context if youre an ai.md`; inspected latest robustness scripts (`train_protocol_multimodel_robust_matrix.py`, `evaluate_xgb_robustness.py`, `xgb_protocol_ids_utils.py`) for required schema; cloned external repo; extracted `Dataset/ICUDatasetProcessed.zip`; inspected CSV schemas (`Attack.csv`, `environmentMonitoring.csv`, `patientMonitoring.csv`); computed required-feature overlap against `reports/full_gpu_hpo_models_20260306_195851/metrics_summary.json`; executed a direct smoke call to `scripts/evaluate_xgb_robustness.py` on external CSV.
  - Result: External dataset has `0/45` overlap with required model features and lacks `protocol_hint`; current robustness pipeline fails at CSV load with `ValueError: Usecols do not match columns ... ['protocol_hint']`. Current trained models are not directly reusable on this source without a feature/domain adapter.
- 2026-03-10 11:30 (Europe/Berlin) - Audited Cycle 3 scripts and fixed stability-consistency + reproducibility defects
  - Why: User requested direct script audit of the Python code being executed on cluster due concern about wasted large-job runtime.
  - Commands run: inspected `scripts/train_protocol_multimodel_robust_matrix.py` and `scripts/consolidate_protocol_multimodel_robust_report.py`; patched candidate stability grouping from `model+family+seed` to `model+family`; added coarse-only acceptance fallback; made protocol seed offsets deterministic (removed use of randomized Python `hash()`); updated consolidator stable-candidate identifier fallback; validated via `py -3 -m py_compile`.
  - Result: Future runs can correctly determine cross-seed stability and have deterministic protocol row-capping behavior; consolidated report now handles `candidate_group_key`.
- 2026-03-10 19:52 (Europe/Berlin) - Applied throughput-focused job config updates (CPU threading + startup cost)
  - Why: User asked how to speed up long robust matrix training and observed low GPU utilization with only 16 CPUs requested.
  - Commands run: patched `scripts/train_protocol_multimodel_robust_matrix.sbatch` to request `--cpus-per-task=64`, export thread env vars (`OMP/MKL/OPENBLAS/NUMEXPR`), print CPU thread settings, and skip per-run pip installs by default (`SKIP_PIP_INSTALL=1` unless explicitly overridden); added `EXTRA_ARGS` override support in sbatch for fast-profile runs without editing script; patched `scripts/train_protocol_multimodel_robust_matrix.py` to set thread counts explicitly for XGBoost (`nthread`), CatBoost (`thread_count`), and LightGBM (`n_jobs`) from Slurm CPU allocation; validated with `py -3 -m py_compile`.
  - Result: Next submissions can use full CPU allocation for CPU-bound/model-threaded phases and avoid repeated dependency install overhead.
- 2026-03-10 11:08 (Europe/Berlin) - Remote submission flow hardened after failed attempt
  - Why: First remote submission had shell paste corruption and `tail` on non-existent pending-job log triggered shell exit under `set -e`.
  - Commands run: canceled job `131`, uploaded updated staged launcher `scripts/train_protocol_multimodel_robust_matrix.sbatch`, resubmitted (`JOB_ID=132`), added safe log-wait loop and GPU/log monitoring commands.
  - Result: Submission path now avoids fragile heredoc pasting and handles pending-log race safely.
- 2026-03-10 11:04 (Europe/Berlin) - Runbook rewritten for Cycle 3 protocol multi-model job
  - Why: User requested complete copy-pasteable Windows PowerShell + server command flow with no placeholder tokens and fail-fast troubleshooting branch.
  - Commands run: rewrote `CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md` with full stage/upload/submit/monitor/fetch/cleanup steps, corrected PowerShell quoting guidance (`ssh rust '...|head...'`), and added corrected resubmit paths for dependency/data/base-run failures.
  - Result: Root runbook now matches current job topology (`/home/capstone15`, staged scripts, fixed data-path discovery, venv path, authoritative outputs).
- 2026-03-09 18:43 (Europe/Berlin) - Added dedicated remote submit guide file for future job launches
  - Why: User requested a standalone top-to-bottom submit guide for staging, submitting, monitoring, fetching outputs, and cleanup for future jobs.
  - Commands run: created `CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md`; patched runbook/context pointers plus recent/timeline/session summary entries in this file.
  - Result: Project root now includes a single reusable remote-submit playbook and context now explicitly points to it.
- 2026-03-09 18:43 (Europe/Berlin) - Added new-thread launch prompt template for remote Slurm workflow
  - Why: User requested a reusable instruction block for future threads to launch jobs from local VS Code terminal using learned constraints.
  - Commands run: patched `context if youre an ai.md` runbook section to add a copy/paste prompt template with path, staging, monitoring, and failure-handling requirements.
  - Result: Context now contains a durable `New-thread prompt template` under `Environment / Runbook` for consistent remote submission guidance.
- 2026-03-09 14:55 (Europe/Berlin) - Added persistent remote-submit runbook (manual vs Codex)
  - Why: User requested future-thread instructions in context file for sending Python jobs from local VS Code terminal, including which actions are manual versus automatable.
  - Commands run: inspected `C:/Users/Hugo/.ssh/config`, `C:/Users/Hugo/.ssh/known_hosts`, validated SSH alias resolution (`ssh -G rust`), reviewed `scripts/train_baselines_cpu.sbatch`, and patched runbook/timeline/session sections in this file.
  - Result: Context now contains a concrete SSH+Slurm workflow using alias `rust`, explicit role split (manual password step vs Codex automation), and command templates for submit/monitor/log checks.
- 2026-03-09 10:07 (Europe/Berlin) - Backfilled missing context state into this file
  - Why: User requested logging missing information before further work because project continuity appeared incomplete.
  - Commands run: inspected this file sections (`Right Now`, `Recent Changes`, `Conversation Timeline`, `Session Summary`) and patched them with completed-run status, interpretation, and unresolved gaps.
  - Result: Context now reflects the current robustness state, key risks, and next hardening decision instead of stale "run cancellation pending" status.
- 2026-03-09 10:02 (Europe/Berlin) - Interpreted realistic full run `xgb_robustness_realistic_full_20260308_212054`
  - Why: User asked to open and interpret the completed server run results.
  - Commands run: reviewed `summary.json`, `robustness_query_metrics_global.csv`, `robustness_query_metrics_protocol.csv`, and `query_trace_summary.json`.
  - Result: Malicious query evasion remained low-impact (global delta F1 about -0.0006 at epsilon 0.1), while benign-side query attack caused major FPR drift (global FPR 0.2329 at epsilon 0.1), concentrated in WiFi (WiFi FPR 0.4658 at epsilon 0.1).
- 2026-03-09 09:35 (Europe/Berlin) - Server submission troubleshooting context gap identified
  - Why: Prior interaction showed repeated confusion around remote project path and where outputs were written/downloaded.
  - Commands run: verified available local artifacts and run directories after sync; checked that interpreted run exists locally under `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_realistic_full_20260308_212054`.
  - Result: Clarified that completed outputs can be interpreted locally once synced; remaining gap is a documented one-command remote submit + fetch workflow for future runs.
- 2026-03-08 19:14 (Europe/Berlin) - Cancellation check for local realistic robustness run
  - Why: User requested canceling the currently running full local realistic robustness execution.
  - Commands run: queried running Python processes for `evaluate_xgb_robustness.py` via `Get-CimInstance Win32_Process` and attempted forced stop when matched.
  - Result: No matching active process was found (`killed=none`), indicating the run had already stopped by the time cancel was requested.
- 2026-03-08 19:08 (Europe/Berlin) - Started full local realistic robustness run (background, CUDA)
  - Why: User requested running the full evaluation directly from local machine without waiting for completion.
  - Commands run: launched background process via `.venv39` Python with full realistic settings (`epsilons=0,0.01,0.02,0.05,0.10`, legacy methods enabled, realistic mode on, query budgets 300/150, realistic sample caps 5000/protocol, chunk size 250000, `--xgb-device cuda`) and timestamped output dir `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_realistic_full_20260308_190831`; logs at `ids-xgb-robust-realistic_full_20260308_190831.out.log/.err.log`.
  - Result: Run started successfully and is actively emitting `[PROGRESS]` lines (train count pass in progress/completed and continuing).
- 2026-03-08 17:12 (Europe/Berlin) - Added GPU-aware inference selection + GPU-default robustness sbatch launcher
  - Why: User requested using GPU whenever beneficial; query-limited robustness repeatedly scores XGBoost models and can leverage GPU inference when available.
  - Commands run: patched `scripts/evaluate_xgb_robustness.py` with `--xgb-device {auto,cpu,cuda}` and runtime device auto-fallback (`configure_boosters_device`); patched `scripts/evaluate_xgb_robustness.sbatch` to `gpu` partition with `--gres=gpu:1` and default `XGB_DEVICE=cuda`; syntax-checked with `.venv39` `py_compile`.
  - Result: Robustness script now explicitly supports GPU inference selection and logs resolved device; sbatch defaults align with GPU-first execution.
- 2026-03-08 17:12 (Europe/Berlin) - Implemented realistic robustness v2 + validated smoke and determinism
  - Why: User requested a realistic robustness mode: query-limited black-box attacks, relation-aware constraints, benign-side perturbation, additional artifacts, and progress visibility.
  - Commands run: extended `scripts/evaluate_xgb_robustness.py` with realistic-mode CLI, realism profile construction, relation constraints, sparse query hillclimb attacks (`query_sparse_hillclimb`, `query_sparse_hillclimb_benign`), query-trace logging, and new outputs (`robustness_query_metrics_global/protocol.csv`, `realism_profile.json`, `query_trace_summary.json`); executed repeated smoke runs to `.../xgb_robustness_realistic_smoke`; executed deterministic rerun to `.../xgb_robustness_realistic_smoke_det`; compared hashes/JSON runs.
  - Result: Realistic mode is working end-to-end with periodic ETA progress, zero reported projection violations in smoke output, epsilon=0 query metrics matching baseline F1, and deterministic query metric artifacts for fixed seed.
- 2026-03-08 16:11 (Europe/Berlin) - Full-budget robustness evaluation executed successfully
  - Why: User requested running a full proper robustness run (not smoke) and required visible progress.
  - Commands run: executed `.venv39` run of `scripts/evaluate_xgb_robustness.py` with full plan settings (`sample-attack-per-protocol=25000`, `sample-benign-per-protocol=25000`, `surrogate-train-per-protocol=200000`, `surrogate-epochs=12`, epsilons `0,0.01,0.02,0.05,0.10`, methods `surrogate_fgsm,surrogate_pgd,heuristic_shap`, `chunk-size=250000`) into `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_full_20260308_1610`.
  - Result: Run completed with full progress logs and generated all robustness artifacts (`robustness_metrics_global/protocol`, `robustness_metrics_by_epsilon.json`, `perturbation_stats.csv`, `constraints_summary.csv`, `attack_config.json`, `summary.json`).
- 2026-03-08 16:00 (Europe/Berlin) - Repaired local Python environment for robustness script execution
  - Why: User hit NumPy C-extension import failure in `.venv` after creating it with Python 3.13 and asked for hands-on fix.
  - Commands run: enumerated installed interpreters (`py -0p`), created fresh venv on Python 3.9 (`py -3.9 -m venv .venv39`), upgraded tooling (`pip/setuptools/wheel`), installed runtime dependencies (`numpy==1.26.4`, `pandas==2.2.3`, `xgboost==2.1.4`), validated imports and script help, then executed smoke run to `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_smoke_py39`.
  - Result: Robustness pipeline now runs successfully end-to-end from `.venv39` with progress logs and output artifacts generated.
- 2026-03-08 15:50 (Europe/Berlin) - Implemented XGBoost robustness evaluator + smoke-validated outputs
  - Why: User requested implementing the agreed robustness plan against the three protocol-routed trained XGBoost models, including constrained attacks and required metrics.
  - Commands run: added `scripts/evaluate_xgb_robustness.py` and `scripts/evaluate_xgb_robustness.sbatch`; syntax-checked with `.venv` `py_compile`; executed smoke robustness runs with sampled settings and epsilons (`0,0.02`) into `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_smoke`; validated output presence and basic invariants/consistency (delta-F1 at epsilon 0, locked-feature stability, unlocked-bound checks).
  - Result: New robustness pipeline now produces `robustness_metrics_global.csv`, `robustness_metrics_protocol.csv`, `robustness_metrics_by_epsilon.json`, `perturbation_stats.csv`, `constraints_summary.csv`, `attack_config.json`, and `summary.json`; console progress/ETA logging is included for long processing phases.
- 2026-03-07 14:04 (Europe/Berlin) - Added simplified Vite React IDS replay UI + simulation-data pipeline
  - Why: User requested replacing the complex Streamlit interface with a simpler React app that always replays `metadata_test.csv`, with sequential alerts and explainability.
  - Commands run: installed Node.js LTS (`winget`); scaffolded `web/ids-react-ui` (Vite React); added `scripts/generate_ids_react_simulation_data.py`; generated `web/ids-react-ui/public/simulation_data.json` using per-protocol XGBoost models and explainability utilities; rewrote React UI (`src/App.jsx`, `src/index.css`); added launch/docs helpers (`start_ids_react_ui.bat`, `build_ids_react_sim_small.bat`, `web/ids-react-ui/README_SIMULATOR.md`); verified with `npm run build` and local dev server health (`http://127.0.0.1:5173` -> HTTP 200).
  - Result: New UI provides fixed-rate replay controls (start/pause/reset), sequential alert stream, click-through local explanations, and always-visible global explanations; current simulation artifact is capped for rapid iteration and full-generation command is documented.
- 2026-03-07 13:23 (Europe/Berlin) - Added sequential replay mode and large-file handling in Streamlit UI
  - Why: User reported that upload behavior should simulate flows over time and that the default 200MB upload cap blocks realistic test files.
  - Commands run: patched `scripts/ids_xgb_interpretability_ui.py` (batch source mode, local CSV path support, one-shot row cap, sequential simulation with tick-based replay/history/recent alerts), added `.streamlit/config.toml` with increased upload/message limits, syntax-checked (`.venv` py_compile), restarted UI and verified `http://localhost:8501` returns `200`.
  - Result: UI now supports realistic sequential flow replay controls and can bypass upload limits via local file path; Streamlit upload limit raised to 2GB for environments where browser/system constraints allow it.
- 2026-03-07 12:58 (Europe/Berlin) - Full explainability artifacts generated + Streamlit startup validated
  - Why: Complete the continuation task end-to-end after dependency setup and smoke validation.
  - Commands run: executed full generator run (`.venv` python) for `reports/full_gpu_hpo_models_20260306_195851`; inspected output files and `summary.json`; launched Streamlit app headless on local ports (`8521`, `8522`) to confirm runtime startup.
  - Result: Full explainability bundle is now present at `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/` and UI server starts successfully using `streamlit run scripts/ids_xgb_interpretability_ui.py`.
- 2026-03-07 12:56 (Europe/Berlin) - Provisioned local venv dependencies + validated explainability smoke run
  - Why: The previous local sync was blocked from runtime validation due missing Python packages.
  - Commands run: created venv (`py -3 -m venv .venv`); installed dependencies with venv python (`numpy`, `pandas`, `xgboost`, `streamlit`); ran smoke artifact generation with reduced caps to `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability_smoke`; re-ran syntax checks and verified Streamlit CLI version.
  - Result: Local runtime is functional for the explainability stack; smoke outputs now include global feature rankings, local case explanations, case/reference rows, and summary/manifest files under `xgb_explainability_smoke`.
- 2026-03-07 12:50 (Europe/Berlin) - Synced local repo with XGBoost explainability + UI stack
  - Why: User requested continuation of global/local explanation and UI setup after re-reading the context file.
  - Commands run: re-read `context if youre an ai.md`; audited local `scripts/` and `reports/` state; added `scripts/xgb_protocol_ids_utils.py`, `scripts/generate_xgb_explainability_artifacts.py`, `scripts/ids_xgb_interpretability_ui.py`, `scripts/generate_xgb_explainability_artifacts.sbatch`, and `reports/XGBOOST_INTERPRETABILITY_UI.md`; syntax-checked via `py -3 -m py_compile`; attempted lightweight artifact-generation smoke run.
  - Result: Missing explainability/UI components are now present in the local repository with protocol-routed prediction, threshold-aware local explanations, global importance outputs, and Streamlit workflows; runtime smoke test is currently blocked by missing local Python packages (`numpy`, `pandas`, `xgboost`, `streamlit`).
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
- Decision: Lock robust-matrix model families to `xgboost` and `catboost` only for current stabilization objective.
  - Reason: User requested removing `mlp`, and stability passes were achieved with these two boosted families under current gates.
  - Alternatives considered: Keeping all families (higher search/runtime cost) or single-family restriction (reduced diversity/risk of overfitting to one family behavior).
- Decision: Use adaptive fraction-based sampling profile (`sampling_policy=adaptive_fraction`) with deterministic seeded selection for coarse and stability stages.
  - Reason: Fixed-count-only sampling underused available data and contributed to unstable seed behavior.
  - Alternatives considered: Fixed-count-only legacy policy (kept as backward-compatible option), or full-pool training every stage (higher runtime/CPU pressure).
- Decision: Keep official hard gates unchanged while hardening threshold search with internal attacked-benign margin.
  - Reason: Goal is all-seed pass without loosening published acceptance criteria.
  - Alternatives considered: Relaxing gate thresholds (rejected), or no margin hardening (weaker stability against attacked-benign drift).
- Decision: Default robust candidate artifact saving to ON.
  - Reason: Finished runs without saved artifacts block deployment and post-hoc model reuse.
  - Alternatives considered: Manual per-run `--save-models` opt-in (error-prone) or metrics-only outputs (insufficient for deployment).
- Decision: Keep attack source mode at `fixed_xgb` for both coarse and stability in this revision.
  - Reason: Reduces attack-source variance while evaluating sampling/threshold/shortlist mechanics.
  - Alternatives considered: `candidate_model` or `hybrid` attack source (more variability; deferred).
- Decision: Keep external benign augmentation disabled in this revision.
  - Reason: Scope control for stabilization and reproducibility of current acceptance objective.
  - Alternatives considered: Enable external benign now (adds domain-shift confounders before core stability objective is finalized).
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
1. Submit a new robust-matrix `stage-mode both` server run using the patched launchers (model-save enabled, adaptive sampling, threshold margin, query fast-projection) and monitor until stability artifacts are complete.
2. Compare `xgboost__E` vs `catboost__C` on protocol-level rows plus all stability seeds from the new artifact-complete run, then lock one final deploy candidate.
3. Patch/export additional persisted clean metrics (precision/recall) in decision outputs if thesis tables require them without recomputation.

## Open Questions / Risks
- Should train/test naming mismatch for TCP_IP attacks be normalized to families before reporting per-attack metrics?
- Do we include profiling CSV captures in training (for benign/background context) or keep them as out-of-distribution evaluation only?
- How closely does the local reconstruction of the feature extraction logic match the original unpublished conversion implementation?
- Should covariance/variance behavior be further calibrated against the original authors' exact unpublished converter (if released)?
- Should profiling split ratio remain 20% test, or be tuned to target specific benign/attack balance?
- Given strong class imbalance and split drift, should threshold calibration be done per-protocol or globally with protocol-aware post-processing?
- Should the highly correlated top features (`Std`, `Radius`, `Max`, `Magnitue`) be regularized/reduced before adversarial robustness analysis?

## Environment / Runbook
- Primary reference for future submissions: `CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md`
- Start:
  - Login (preferred alias): `ssh rust`
  - Login (explicit jump fallback): `ssh capstone15@10.205.20.10 -J capstone15@ssh.iesci.tech`
  - Remote project root: `/home/capstone15`
  - Activate env: `source /home/capstone15/.venv/bin/activate`
- Test:
  - Quick CPU interactive test: `srun --partition=interactive --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash`
- Build:
  - Submit batch job: `sbatch <job_script>.sh`
- Deploy:
  - Follow live job output: `tail -F /home/capstone15/reports/<log_file>.log`
  - Check queue/state: `squeue -u capstone15` and `sacct -j <JOB_ID>`
- Env vars:
  - `PROJECT_DIR`, `STAGE_DIR`, `BASE_RUN_DIR`, `TRAIN_CSV`, `TEST_CSV`, `OUT_ROOT`, `VENV`, `SKIP_PIP_INSTALL`, `EXTRA_ARGS`

### Future-thread remote job workflow (manual vs Codex)
- Account identity:
  - Username: `capstone15`
  - Password handling: Enter manually at SSH prompt; do not store plaintext passwords in this file/repo.
- Manual steps (user must do):
  - Open local VS Code terminal.
  - Run `ssh rust`.
  - Enter password when prompted.
  - Keep the SSH session open while Codex provides remote commands to execute.
- Codex can do itself once session/path are confirmed:
  - Write or patch Python scripts and `.sbatch` launchers.
  - Validate command syntax and resource flags.
  - Provide exact `sbatch`, `squeue`, `sacct`, `tail`, and `scancel` command sequence for the specific job.
  - Parse logs/metrics and summarize outcomes.
- Standard Python job sequence:
  - `cd /home/capstone15`
  - `sbatch /home/capstone15/scripts/<job_name>.sbatch`
  - `squeue -u capstone15`
  - `sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,Elapsed,ExitCode`
  - `tail -f /home/capstone15/reports/<log_prefix>_<JOB_ID>.log`
- Minimal `.sbatch` template:
  - `#!/bin/bash`
  - `#SBATCH --job-name=<name>`
  - `#SBATCH --partition=<cpu|gpu>`
  - `#SBATCH --cpus-per-task=<N>`
  - `#SBATCH --mem=<XG>`
  - `#SBATCH --time=<HH:MM:SS>`
  - `#SBATCH --output=/home/capstone15/reports/<name>_%j.log`
  - `#SBATCH --gres=gpu:1` (only for GPU jobs)
  - `set -euo pipefail`
  - `source /home/capstone15/.venv/bin/activate`
  - `python3 /home/capstone15/scripts/<script>.py <args>`

### New-thread prompt template (copy/paste)
- Use this when starting a fresh Codex chat and you need remote job submission help:
  - `Read context if youre an ai.md first, then help me launch a Slurm job from local VS Code terminal to server alias rust (user capstone15).`
  - `Constraints learned:`
  - `1) Password entry is manual/interactive.`
  - `2) Server project root is /home/capstone15 (not /home/capstone15/Medical-IoMT-).`
  - `3) Auto-discover data paths with find for metadata_train.csv and metadata_test.csv before submit.`
  - `4) Temporary script staging path is /home/capstone15/.jobstage/iomt_<timestamp>/scripts.`
  - `5) Use VENV=/home/capstone15/.venvs/ids-robust-venv for this workflow.`
  - `6) Never give placeholder commands like <JOB_ID> or <timestamp>; always fill concrete values.`
  - `Deliver exact copy-paste commands for: stage -> submit -> monitor (squeue/sacct/tail) -> fetch outputs (scp) -> cleanup stage.`
  - `If failure occurs, immediately tail the job log, identify root cause, and provide corrected resubmit command.`

## References
- Ticket:
- PR:
- Docs:
- Commits:

## Conversation Timeline
- 2026-03-13 12:58 (Europe/Berlin) - Data-first robustness planning audit
  - User intent: Inspect data deeply first to decide how to use all available data effectively before changing robustness scripts.
  - Actions taken: Re-read project context and loaded key run artifacts (`decision_table_protocol_*`, `matrix_summary.json`, `hardening_constraints_summary.csv`, `hardening_realism_profile.json`); ran new streaming dataset audit on full merged train/test CSVs and wrote outputs to `reports/data_audit_20260313_124851` (protocol/label balance, feature quality, family distribution, train-test share shift, top source files, feature drift); aggregated candidate attack stats to quantify query saturation and runtime composition.
  - Outcome: Identified primary data constraints for redesign: strong class/protocol skew, protocol-specific family gaps, no MQTT benign in test, high protocol-conditional feature degeneracy (especially Bluetooth), and query attack loop saturation consistent with CPU-bound runtime pressure.
  - Next session should start with: Implement agreed script changes in order: (1) sampling policy updates informed by audit outputs, (2) attack-loop efficiency changes, (3) rerun coarse+stability on server with updated profile and compare against baseline metrics.
- 2026-03-10 13:05 (Europe/Berlin) - External dataset compatibility check for robustness phase
  - User intent: Determine whether current CICIoMT-trained models and robustness workflow can be evaluated on data from another source (`imfaisalmalik/IoT-Healthcare-Security-Dataset`).
  - Actions taken: Re-read context and latest robustness scripts to extract schema assumptions; cloned and unpacked external dataset; inspected external CSV headers/labels; compared against required `feature_columns` from base run metrics; executed a direct evaluator smoke invocation using external CSV paths to validate runtime compatibility.
  - Outcome: Compatibility is currently blocked: external CSVs are packet/MQTT feature schema (52 cols) with no overlap against required 45 CICIoMT flow features, no `protocol_hint`, and evaluator aborts during `usecols` validation.
  - Next session should start with: Decide adapter strategy (rebuild external data into CICIoMT feature schema + protocol mapping) and then run a controlled cross-source robustness benchmark (at minimum MQTT-routed first pass).
- 2026-03-09 18:43 (Europe/Berlin) - Added standalone remote submit runbook file in project root
  - User intent: Save a full step-0-to-finish guide for sending future jobs from local VS Code terminal to server and retrieving outputs.
  - Actions taken: Created `CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md` with annotated commands for staging scripts, building `.sbatch`, submit/monitor flow, fetching outputs, and cleanup; added runbook pointer in this context file.
  - Outcome: Future sessions now have a single canonical file for operational Slurm submission workflow without rebuilding commands from memory.
  - Next session should start with: Follow `CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md` directly and only adjust script names/arguments.
- 2026-03-09 18:43 (Europe/Berlin) - Added reusable new-thread prompt for remote job launch
  - User intent: Save a copy/paste instruction set so future threads can reliably launch jobs from local VS Code terminal to `rust`.
  - Actions taken: Inserted `New-thread prompt template (copy/paste)` under `Environment / Runbook`, including fixed server-root, manual password step, staged script path, venv path, concrete-command requirement, and fail-fast debug branch.
  - Outcome: Future sessions can be bootstrapped with one prompt that encodes current operational lessons and avoids prior placeholder/path mistakes.
  - Next session should start with: Paste the runbook prompt template and execute the generated submit/monitor/fetch sequence with real IDs/paths.
- 2026-03-09 18:32 (Europe/Berlin) - Reviewed completed full-budget WiFi hardening run report
  - User intent: Confirm and interpret results from the newly completed run in `reports/`.
  - Actions taken: Inspected latest run bundle `reports/full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_180250/`; parsed `hardening_summary.json`, `hard_negative_stats.csv`, and `thresholds_by_protocol.json`; compared key deltas against prior hardening run.
  - Outcome: Full-budget run completed and produced strong hard-negative generation, but selected WiFi threshold dropped sharply (`0.2261 -> 0.0351`) causing large clean and attacked benign FPR increase (~`0.0063 -> 0.0768` and `0.00536 -> 0.07669`) while restoring attacked malicious recall to target (~`0.995`).
  - Next session should start with: Revisit threshold objective/constraints (or decouple malicious-recall target from WiFi benign-FPR objective) before accepting this hardened model for deployment.
- 2026-03-09 16:17 (Europe/Berlin) - Threat-model validation of adaptive/novel robustness coverage
  - User intent: Confirm whether current robustness and hardening setup truly evaluates novel/adaptive adversarial behavior.
  - Actions taken: Audited realistic attack/hardening code paths and latest query-metrics artifacts (`evaluate_xgb_robustness.py`, `train_wifi_robust_hardening.py`, `robustness_query_metrics_global.csv`, `robustness_query_metrics_protocol.csv`, `summary.json` notes).
  - Outcome: Confirmed coverage includes query-limited adaptive black-box attacks with realistic constraints plus legacy transfer-style attacks, while noting remaining gaps (static model assumption, no data-poisoning/temporal multi-flow adaptation, MQTT benign-side evaluation gap).
  - Next session should start with: Expand robustness campaign breadth (multi-seed/budget sweeps and additional black-box optimizers) and add MQTT benign slice before broader hardening decisions.
- 2026-03-09 16:07 (Europe/Berlin) - WiFi hardening mechanism explanation sync
  - User intent: Understand the detailed mechanics/tradeoffs of the current WiFi XGBoost hardening approach.
  - Actions taken: Reviewed `scripts/train_wifi_robust_hardening.py`, `scripts/train_wifi_robust_hardening.sbatch`, threshold-loading utility behavior, and latest hardening output summary fields.
  - Outcome: Session now has a concrete technical mapping of hardening stages (split, constrained benign hard-negative generation, WiFi retrain, threshold recalibration, artifact semantics) and current caveats.
  - Next session should start with: Run/inspect a full-budget hardening pass and then rerun realistic robustness evaluation against the new WiFi model+threshold.
- 2026-03-09 15:58 (Europe/Berlin) - WiFi hardening context/script resync
  - User intent: Re-open latest context and recently written scripts to get fully up to speed on WiFi XGBoost robustness work.
  - Actions taken: Re-read this context file and inspected recent script/run artifacts (`scripts/train_wifi_robust_hardening.py`, `scripts/train_wifi_robust_hardening.sbatch`, `scripts/xgb_protocol_ids_utils.py`, latest `hardening_summary.json` outputs, and robustness query metrics files).
  - Outcome: Session is synchronized; WiFi hardening pipeline is implemented and has produced two local run directories, but latest hardening output is currently small-sample calibration and copied global metrics files remain inherited from base run.
  - Next session should start with: Execute a full-budget WiFi hardening run and then rerun realistic robustness query evaluation against the hardened WiFi model/thresholds.
- 2026-03-09 14:55 (Europe/Berlin) - Added persistent SSH+Slurm runbook for future threads
  - User intent: Document exactly how to submit Python jobs from local terminal and clarify manual actions versus Codex actions.
  - Actions taken: Verified SSH host aliases and known hosts, validated alias expansion for `rust` with `ProxyJump`, reviewed current Slurm launcher, and added a runbook section with role split + command templates.
  - Outcome: Future sessions now have a single reference for remote submit flow (`ssh rust` -> `sbatch` -> monitor/logs) with explicit manual credential entry expectations.
  - Next session should start with: User opens SSH session, then asks Codex to generate or run the specific `.sbatch` workflow for the target script.
- 2026-03-09 10:07 (Europe/Berlin) - Missing-context backfill + robustness interpretation sync
  - User intent: Log missing project state in this context file before further model work.
  - Actions taken: Reviewed and interpreted completed realistic run outputs from `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_realistic_full_20260308_212054` (`summary.json`, query global/protocol metrics, query trace summary); updated `Right Now`, `Recent Changes`, and `Session Summary`.
  - Outcome: Context now records the true latest robustness status and key risk (benign-side FPR drift, mainly WiFi), replacing stale "check if run completed" notes.
  - Next session should start with: Implement WiFi-focused hardening and rerun realistic query evaluation to verify FPR reduction.
- 2026-03-09 10:05 (Europe/Berlin) - Decision checkpoint after realistic run review
  - User intent: Ask whether model fine-tuning is necessary after seeing robustness outputs.
  - Actions taken: Assessed tradeoff between malicious-evasion robustness and benign-side instability from query-limited attack metrics.
  - Outcome: Recommended targeted hardening (adversarial benign hard negatives + threshold recalibration) instead of full architecture overhaul.
  - Next session should start with: Define hardening experiment configuration and acceptance criteria (target FPR bounds at fixed recall).
- 2026-03-08 19:14 (Europe/Berlin) - User-requested cancel of local robustness run
  - User intent: Cancel the locally launched full realistic robustness run.
  - Actions taken: Searched running Windows `python.exe` processes for `evaluate_xgb_robustness.py` command lines and issued force-stop logic for matches.
  - Outcome: No matching process remained at cancellation time, so nothing required termination.
  - Next session should start with: Confirm whether the last run completed or failed by checking the `.out/.err` logs and output directory artifacts.
- 2026-03-08 19:08 (Europe/Berlin) - Local full realistic robustness run launched in background
  - User intent: Run the full realistic robustness evaluation locally and avoid waiting for completion interactively.
  - Actions taken: Started `scripts/evaluate_xgb_robustness.py` with full settings in a background process using `.venv39` and `--xgb-device cuda`; set output to `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_realistic_full_20260308_190831`; validated startup via progress lines in `ids-xgb-robust-realistic_full_20260308_190831.out.log`.
  - Outcome: Job is running locally with visible progress logging and artifacts being written to the new timestamped run directory.
  - Next session should start with: Check completion status, parse generated query/global protocol metrics, and summarize robustness findings.
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
- 2026-03-07 12:50 (Europe/Berlin) - Local continuation: explainability/UI files synced and validated for syntax
  - User intent: Continue the global/local explainability + UI setup, but first re-read the AI context file to stay aligned.
  - Actions taken: Re-read `context if youre an ai.md`; inspected local repo and confirmed missing explainability/UI files in this workspace; added `scripts/xgb_protocol_ids_utils.py`, `scripts/generate_xgb_explainability_artifacts.py`, `scripts/generate_xgb_explainability_artifacts.sbatch`, `scripts/ids_xgb_interpretability_ui.py`, and `reports/XGBOOST_INTERPRETABILITY_UI.md`; ran `py -3 -m py_compile` for syntax validation; attempted a lightweight generation run and checked package availability.
  - Outcome: Local repository now includes protocol-routed XGBoost explainability and Streamlit UX components with runbook docs; runtime execution remains blocked in this shell due missing `numpy/pandas/xgboost/streamlit`.
  - Next session should start with: Install required Python packages in project venv, run `generate_xgb_explainability_artifacts.py`, then launch `streamlit run scripts/ids_xgb_interpretability_ui.py`.
- 2026-03-07 12:56 (Europe/Berlin) - Local dependency install + smoke explainability generation
  - User intent: Continue setup end-to-end and verify that explainability artifact creation works locally.
  - Actions taken: Created `.venv`; installed `numpy`, `pandas`, `xgboost`, and `streamlit`; executed `scripts/generate_xgb_explainability_artifacts.py` with reduced sampling limits to produce a smoke output bundle; confirmed Streamlit installation/version.
  - Outcome: Smoke bundle generated successfully at `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability_smoke/` with all expected files (`global_feature_importance.csv`, local case artifacts, `reference_rows.csv`, `summary.json`, `manifest.json`).
  - Next session should start with: Run full-size artifact generation into `<run-dir>/xgb_explainability` and launch the Streamlit UI for manual operator flow validation.
- 2026-03-07 12:58 (Europe/Berlin) - Full-size explainability generation + UI runtime startup check
  - User intent: Finish the continuation task by completing full output generation and validating UI startup.
  - Actions taken: Ran full `scripts/generate_xgb_explainability_artifacts.py` pass into `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/`; inspected generated files and `summary.json`; launched Streamlit app headless to verify server startup and URL binding.
  - Outcome: Full local explainability bundle is generated and the Streamlit IDS UI starts successfully with the updated scripts and artifacts.
  - Next session should start with: Perform interactive UI validation (single-flow, batch upload, global/case tabs) and, if needed, connect a live flow ingestion source.
- 2026-03-07 13:23 (Europe/Berlin) - UI usability patch for sequential replay + large input handling
  - User intent: Make batch behavior feel like realistic time-sequenced flow processing and address 200MB upload blocking.
  - Actions taken: Updated `scripts/ids_xgb_interpretability_ui.py` to add batch source selection (upload vs local path), one-shot row-limited local scoring, and tick-based sequential replay simulation with configurable rows/tick and flows/sec plus replay metrics/history/recent alerts; added `.streamlit/config.toml` to increase `server.maxUploadSize`/`server.maxMessageSize` to 2048MB; validated syntax and restarted UI.
  - Outcome: Operators can now replay flows in sequential chunks over simulated time and avoid upload-size limitations by pointing directly to local CSV files.
  - Next session should start with: Tune replay parameters against target traffic assumptions and validate alarm cadence with representative capture windows.
- 2026-03-07 14:04 (Europe/Berlin) - Vite React simulator migration for simplified fixed-data replay
  - User intent: Replace complex Streamlit UX with a simpler Vite React simulator that always uses `metadata_test.csv`, replays alerts sequentially at a fixed derived speed, and exposes local/global explanations.
  - Actions taken: Installed Node.js LTS and npm toolchain; scaffolded `web/ids-react-ui`; created `scripts/generate_ids_react_simulation_data.py` to precompute replay windows and alert explanations from protocol-routed XGBoost artifacts; generated a capped simulation bundle (`max_windows=5000`) into `web/ids-react-ui/public/simulation_data.json`; rewrote `src/App.jsx`/`src/index.css` for minimal operator flow (start/pause/reset, live alert feed, local explanation view, global explanation panel); added usage docs and launcher scripts.
  - Outcome: React simulator is running locally at `http://127.0.0.1:5173`, with sequential alert popups and explainability aligned to the selected XGBoost models.
  - Next session should start with: Generate full uncapped simulation bundle (`--max-windows 0`) once UX behavior is confirmed acceptable.
- 2026-03-08 15:50 (Europe/Berlin) - Robustness attack pipeline implementation (local)
  - User intent: Implement a full robustness evaluation workflow for the three trained protocol XGBoost models, including surrogate-transfer FGSM/PGD, constrained SHAP heuristic evasion, strict feature constraints, and required robustness metrics.
  - Actions taken: Added `scripts/evaluate_xgb_robustness.py` with protocol-balanced sampling, per-protocol percentile-bound constraints, semantic/auto locked features, NumPy logistic surrogate training, FGSM/PGD generation in normalized space, SHAP-top constrained attack, global/per-protocol metric reporting (`ASR`, robust `F1/recall`, `Î”F1`, `L0/L2/Lâˆž`), and progress/ETA console logging; added `scripts/evaluate_xgb_robustness.sbatch`; ran smoke executions and validated outputs/invariants.
  - Outcome: Robustness artifacts are generated successfully in `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_smoke/` with the expected file set and baseline consistency checks.
  - Next session should start with: Launch a larger/full-budget robustness run and interpret per-protocol/per-epsilon degradation patterns for thesis reporting.
- 2026-03-08 16:00 (Europe/Berlin) - Python environment remediation for robustness runtime
  - User intent: After receiving full access, fix the NumPy import/runtime issue directly and get the robustness script running.
  - Actions taken: Detected available interpreters (`3.13`, `3.9`), created a clean Python 3.9 environment (`.venv39`), installed pinned runtime packages, validated module imports + `--help`, and executed a full smoke robustness run with generated artifacts.
  - Outcome: Environment is functional using `.venv39`; robustness run completes with live progress output and artifacts under `xgb_robustness_smoke_py39`.
  - Next session should start with: Use `.venv39` for robustness runs, or install Python 3.11/3.12 and recreate canonical `.venv` if strict naming is needed.
- 2026-03-08 16:11 (Europe/Berlin) - Full robustness run completed with progress logging
  - User intent: Run a full proper robustness test (not smoke) and ensure progress visibility during execution.
  - Actions taken: Ran `.venv39` execution of `scripts/evaluate_xgb_robustness.py` with full attack/sample settings and observed progress output across counting/sampling/surrogate epochs/attack loops; generated output bundle in `xgb_robustness_full_20260308_1610`.
  - Outcome: Full run completed successfully with all expected artifact files; progress logging proved usable for long processing stages.
  - Next session should start with: Interpret and summarize per-protocol robustness degradation across epsilon and attack methods for thesis-ready reporting.
- 2026-03-08 17:12 (Europe/Berlin) - Realistic robustness v2 implementation and validation
  - User intent: Implement a more realistic robustness strategy with query-limited black-box attacks, protocol-aware relation constraints, benign-side perturbation effects, and dedicated reporting outputs.
  - Actions taken: Extended `scripts/evaluate_xgb_robustness.py` with realistic mode, realism profiles, relation-constraint projection, sparse coordinate query hillclimb attacks for both malicious and benign targets, and new output artifacts; added GPU inference selector (`--xgb-device`) with auto CUDA fallback logic; updated `scripts/evaluate_xgb_robustness.sbatch` to request GPU and pass device option; ran realistic smoke + deterministic rerun and validated invariants/consistency.
  - Outcome: New realistic pipeline outputs are generated successfully in `xgb_robustness_realistic_smoke*`, including query metrics and trace/profile JSON files; deterministic metrics were confirmed under fixed seed.
  - Next session should start with: Execute a full realistic campaign (balanced pilot or higher budgets) and compare realistic-query degradation against legacy surrogate/SHAP results for thesis interpretation.

## Session Summary (Most Recent)
- Date: 2026-03-14 (Europe/Berlin)
- Summary: Produced a new thesis-style chapter generator, `Thesis/generate_thesis_experiment_chapter.py`, that rewrites the prior chronology into a more formal narrative with fewer but more decision-relevant tables. The deliverables are `Thesis/THESIS_EXPERIMENT_CHAPTER.md` and `Thesis/THESIS_EXPERIMENT_CHAPTER.docx`, both rendered from saved artifacts and checked against the actual merge/training/robustness scripts. A dedicated document-generation venv now exists at `Thesis/.venv-docs` with `python-docx`.
- Immediate next action: Review the generated `.docx` against thesis formatting requirements and decide whether to keep this as the working chapter or run another pass for template-specific formatting and/or further prose expansion.

## Missing Context Backfill (2026-03-09)
- Confirmed canonical realistic run directory: `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_realistic_full_20260308_212054`.
- Confirmed key query metrics files to use for thesis conclusions:
  - `robustness_query_metrics_global.csv`
  - `robustness_query_metrics_protocol.csv`
  - `query_trace_summary.json`
  - `realism_profile.json`
- Confirmed unresolved evaluation gap: `mqtt` has no benign rows in the test split, so benign-side FPR robustness for `mqtt` is `not_available` and must not be inferred.
- Confirmed attack asymmetry finding:
  - Malicious-evasion query attack currently has low success under constraints/budgets.
  - Benign-side query attack can substantially increase false positives (especially WiFi).
- Confirmed operational gap still missing from docs: one stable "submit remotely from Windows + fetch outputs back" runbook with exact repository path discovery and `scp` commands.
## Session Update (2026-03-07, MIoT IDS Prototype refinements)

- React UI title changed to `MIoT IDS Prototype`.
- Local explanations now include:
  - Human-readable feature labels.
  - Feature-level plain-language descriptions.
  - A short textual narrative explaining why an alert was flagged.
- Replay pacing set to 1 flow/s by generating simulation data with:
  - `--window-rows 50 --window-seconds 50`
- Replay/order and alert feed diversity improved by generator defaults:
  - `--replay-order interleave-protocol-source`
  - `--sampling-strategy balanced-protocol`
- UI stats split into:
  - `Alerts Detected` (all model detections)
  - `Alerts Surfaced` (alerts shown in sequential feed)
  This addresses perceived lag between total detections and displayed alerts.
- Current generated `simulation_data.json` now reports mixed sampled protocols early (mqtt/wifi/bluetooth) and flow rate = 1.0 flow/s.

## Session Update (2026-03-08, Realtime Inference Integration)

- Added realtime backend API: `scripts/ids_realtime_api.py`.
  - Runs protocol-routed XGBoost inference live on `metadata_test.csv`.
  - Computes local explanations only for detected alerts.
  - Exposes endpoints:
    - `GET /api/init`
    - `GET /api/state`
    - `POST /api/start`
    - `POST /api/pause`
    - `POST /api/reset`
    - `GET /api/health`
- React UI migrated from static `simulation_data.json` replay to realtime API polling.
  - `web/ids-react-ui/src/App.jsx` now consumes `/api/*`.
  - UI remains titled `MIoT IDS Prototype`.
- Vite proxy added so frontend can call backend locally without CORS friction.
  - `web/ids-react-ui/vite.config.js` proxies `/api` -> `http://127.0.0.1:8000`.
- Added launcher script for realtime API:
  - `start_ids_realtime_api.bat`.
- Updated docs and startup hints:
  - `web/ids-react-ui/README_SIMULATOR.md`
  - `start_ids_react_ui.bat`
- Verified live run:
  - API healthy at `http://127.0.0.1:8000/api/health`.
  - UI dev server healthy at `http://127.0.0.1:5173` and proxy to `/api/health` works.

## Context Refresh (2026-03-08, User-requested update)

### Current System State
- Prototype mode: **Realtime inference** (not precomputed replay).
- Frontend: Vite React app (`web/ids-react-ui`) at `http://127.0.0.1:5173`.
- Backend: Python stdlib HTTP API (`scripts/ids_realtime_api.py`) at `http://127.0.0.1:8000`.
- API is consumed from frontend through Vite proxy (`/api` -> `127.0.0.1:8000`).

### Realtime Behavior
- Three protocol-routed XGBoost models are loaded from run dir and used live per row.
- Simulation default speed: `1 row/flow per second`.
- Local explanations are computed for surfaced alerts in realtime.
- Global explanations are precomputed once at API startup.

### Main Endpoints
- `GET /api/health`
- `GET /api/init`
- `GET /api/state`
- `POST /api/start`
- `POST /api/pause`
- `POST /api/reset`

### Startup Commands
- Backend: `start_ids_realtime_api.bat`
- Frontend: `start_ids_react_ui.bat`

### Notes
- UI title: `MIoT IDS Prototype`.
- Local explanation panel uses human-readable feature labels + plain-language rationale text.
- Alert counters are split into `Alerts Detected` vs `Alerts Surfaced`.


- 2026-03-10 10:54 (Europe/Berlin) - Implemented Cycle 3 protocol multi-model robust matrix scripts (new files only)
  - Why: User requested from-scratch robust retraining matrix across wifi/mqtt/bluetooth with all model families and bounded coarse+stability stages.
  - Actions taken: Added scripts/train_protocol_multimodel_robust_matrix.py (new orchestrator), scripts/train_protocol_multimodel_robust_matrix.sbatch (new launcher), and scripts/consolidate_protocol_multimodel_robust_report.py (new report consolidator); kept legacy scripts untouched; validated with py_compile and local coarse smoke run (wifi+xgboost tiny settings).
  - Result: New pipeline emits protocol decision tables, global decision table, stability check table, escalation recommendation, matrix summary, and consolidation report artifacts.
- 2026-03-10 11:08 (Europe/Berlin) - Cycle 3 remote submit troubleshooting and relaunch flow correction
  - User intent: Submit the full Cycle 3 run to server and recover from failed launch attempts.
  - Actions taken: Diagnosed PowerShell quoting and heredoc paste corruption issues; explained why pending jobs may not have logs yet; switched flow to upload a prebuilt sbatch launcher file and safe wait-for-log monitoring pattern; provided exact cancel/resubmit commands around jobs `131` and `132`.
  - Outcome: Remote workflow became repeatable and less brittle; job submission moved forward with staged launcher usage.
  - Next session should start with: Verify final job outputs and fetch artifacts to local `reports/server_pull_*`.
- 2026-03-10 11:30 (Europe/Berlin) - Code audit of submitted Cycle 3 script found stability consistency defect
  - User intent: "check the python scripts you wrote" to ensure the large cluster job is valid.
  - Actions taken: Performed line-level audit of `scripts/train_protocol_multimodel_robust_matrix.py` and consolidator; found seed-key mismatch in stability consistency calculation; patched to group by `candidate_group_key=model__family`, added coarse-only acceptance fallback, and replaced nondeterministic `hash(proto)` seed offsets with deterministic offsets; re-ran `py_compile`.
  - Outcome: Patched local scripts now produce correct stability consistency logic for future submissions; previously submitted runs with old script should not trust stability consistency/escalation files without recomputation.
  - Next session should start with: Stage patched scripts and resubmit Cycle 3 job (or postprocess finished run to recompute stability consistency).
- 2026-03-11 10:35 (Europe/Berlin) - Runtime-cut implementation for Cycle 3 all-model coarse pass
  - User intent: Stop long 20h+ runs and implement concrete changes to reduce wall-clock while keeping all four model families.
  - Actions taken: Added new launcher `scripts/train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch` with locked coarse-only profile (`wifi,mqtt,bluetooth`; `xgboost,catboost,lightgbm,mlp`; reduced samples/query budgets; `cpus-per-task=32`; thread env exports; `SKIP_PIP_INSTALL=1` default; `EXTRA_ARGS` overrides). Patched `scripts/train_protocol_multimodel_robust_matrix.py` to emit per-candidate phase timings and completion log line (`candidate_done[...]`) covering `warm_fit`, `attack_gen`, `robust_fit`, `threshold_eval`, and total elapsed.
  - Outcome: New fast execution path is available without replacing legacy launchers, and long silent periods are now diagnosable from candidate-level timing telemetry.
  - Next session should start with: Submit the fast launcher profile, verify first completed candidate latency, and apply fallback profile only if first 8 candidates miss runtime target.
- 2026-03-11 21:55 (Europe/Berlin) - Attack-loop throughput patch for coarse all-model run
  - User intent: Cancel the slow run and force implementation of the runtime/GPU-utilization fixes from the start.
  - Actions taken: Patched `scripts/evaluate_xgb_robustness.py` query attack core to use batched scoring (`inplace_predict` with safe fallback) and new `score_batch_rows`; wired new CLI knob in evaluator and in `scripts/train_protocol_multimodel_robust_matrix.py` (`--query-score-batch-rows`, `--val-malicious-score-batch-rows`) across benign/malicious train+val campaigns; updated fast launcher to stronger throughput profile (`steps=12`, `cands=24`, `score_batch_rows=96`, `threshold_grid=120`); syntax-validated modified scripts.
  - Outcome: Code path is now materially less CPU-loop dominated and should produce better GPU work batching with reduced wall-clock for the same coarse matrix structure.
  - Next session should start with: Upload patched scripts/sbatch to stage, resubmit coarse job, and monitor first candidate completion and `candidate_done[...]` phase timings.

- 2026-03-12 12:57 (Europe/Berlin) - Context + latest robust-matrix results review
  - User intent: Re-read project context and inspect latest run outputs in `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260312_121948` to align next robustness/FPR/F1 work.
  - Actions taken: Re-read `context if youre an ai.md`; reviewed authoritative artifacts (`decision_table_protocol_*.csv`, `decision_table_global.csv`, `matrix_summary.json`, `escalation_recommendation.json`, `external_benign_data_summary.csv`, stability CSVs, constraints/realism profiles); extracted gate-pass/failure patterns across protocols/models.
  - Outcome: Global winner is `xgboost__B__42` (only candidate passing all global gates across wifi/mqtt/bluetooth). WiFi and MQTT have many gate-pass candidates, while Bluetooth is the limiting protocol (only `xgboost/B` passes). Stability files are empty in this coarse-only run, and no external benign data was used (`enabled=false`).
  - Next session should start with: Run stability stage (multi-seed for shortlisted candidates) and enable external benign evaluation to validate low-FPR robustness beyond in-distribution benign traffic.
- 2026-03-12 13:28 (Europe/Berlin) - Targeted V1 recovery implementation (code complete, local long-run intentionally stopped)
  - User intent: Implement the full Targeted V1 robustness recovery plan and then avoid long local smoke execution because full training should run on university servers.
  - Actions taken: Patched `train_protocol_multimodel_robust_matrix.py` for attack-source modes (`fixed_xgb`/`candidate_model`/`hybrid`), no-pass threshold fallback (`adv_shortfall` priority), bluetooth recovery family pack (`A/B/C/D/E`), and deterministic bluetooth hardneg cap; patched `evaluate_xgb_robustness.py` query attack loop to accept scorer callbacks; patched fast launcher to `stage-mode both` plus stability seeds and new profile args; added regression tests in `scripts/test_train_protocol_multimodel_robust_matrix.py`; ran `.venv39` `py_compile` and unit tests successfully.
  - Outcome: Implementation is complete and validated with compile/tests; local smoke was manually stopped per user instruction to avoid long local runtime.
  - Next session should start with: Stage these scripts to server and execute the acceptance run profile; evaluate `decision_table_global.csv` for `>=2` gate-pass candidates and confirm stability consistency outputs.
- 2026-03-13 13:45 (Europe/Berlin) - Robust matrix stabilization implementation (XGBoost+CatBoost only)
  - User intent: Implement all-seed stabilization plan without relaxing official gates, with adaptive sampling and threshold hardening.
  - Actions taken: Patched `scripts/train_protocol_multimodel_robust_matrix.py` to enforce allowed models (`xgboost`,`catboost`) in stage resolution, add diversity-preserving stability shortlist construction, finish adaptive sampling wiring (`sampling_policy` + bucket min/max/fraction use), add threshold internal attacked-benign margin selection path metadata, and include margin in matrix summary gates; updated hparam lookups to model-key only; updated launchers `scripts/train_protocol_multimodel_robust_matrix.sbatch` and `scripts/train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch` to fixed_xgb + adaptive balanced coarse/stability profiles + margin `0.8` + no MLP model args; expanded `scripts/test_train_protocol_multimodel_robust_matrix.py` with adaptive sampler, threshold margin fallback/path, model restriction, and shortlist diversity tests.
  - Validation: `py -3 -m py_compile` passed for script+tests; unit tests passed via `.venv39\\Scripts\\python.exe scripts\\test_train_protocol_multimodel_robust_matrix.py` (20 tests, OK).
  - Outcome: Plan is implemented in code and launch profiles; matrix flow now hard-restricts training shortlist/run candidates to XGBoost+CatBoost with adaptive deterministic sampling and threshold hardening.
- 2026-03-13 14:05 (Europe/Berlin) - Added one-command remote submit script for robust matrix jobs
  - User intent: Prepare a practical script to send the updated robust-matrix run as a server job.
  - Actions taken: Added `scripts/submit_protocol_multimodel_robust_matrix_job.ps1` with profile switch (`default`/`fast`) that creates a remote stage under `/home/capstone15/.jobstage`, uploads required scripts/sbatch, auto-discovers remote train/test CSVs, submits via `sbatch --parsable` with exported env vars (`PROJECT_DIR`, `STAGE_DIR`, `BASE_RUN_DIR`, `TRAIN_CSV`, `TEST_CSV`, `OUT_ROOT`, `VENV`, `SKIP_PIP_INSTALL`, `EXTRA_ARGS`), and writes local submit metadata under `reports/remote_submit_meta/`.
  - Validation: PowerShell AST parse check returned OK; remote execution was not run locally in this session.
  - Outcome: User now has a single executable Windows submit script for server job dispatch using the updated XGBoost+CatBoost launchers.
- 2026-03-13 14:35 (Europe/Berlin) - CPU bottleneck reduction pass for query attack loops (GPU-utilization oriented)
  - User intent: Aggressively reduce CPU bottleneck in robustness training attack loops because GPU utilization remained low.
  - Actions taken: Patched `scripts/evaluate_xgb_robustness.py` hillclimb core with fast batched projection (`project_realistic_candidates_fast`) plus top-k full-refine/rescore path (`fast_projection`, `refine_topk`) under strict query-budget accounting; added CLI knobs `--query-fast-projection/--no-query-fast-projection` and `--query-refine-topk`; exposed new summary telemetry (`fast_projection_rows`, `full_projection_rows`, `fast_refine_candidates_scored`, `full_projection_candidates`). Wired same controls through `scripts/train_protocol_multimodel_robust_matrix.py` stage profiles and campaign stats (`query_fast_projection`, `query_refine_topk`, stage overrides for refine-topk). Updated robust matrix sbatch profiles to higher batch/active-row settings and refine-topk (`fast` profile now uses score-batch 4096 and active-row cap 16384).
  - Validation: `py -3 -m py_compile` passed; `.venv39` tests passed for both `scripts/test_evaluate_xgb_robustness.py` and `scripts/test_train_protocol_multimodel_robust_matrix.py`.
  - Outcome: Query attack generation now performs far fewer expensive full-projection operations per row and is configured for larger GPU scoring batches.
- 2026-03-14 11:05 (Europe/Berlin) - Enforced robust-matrix model artifact persistence by default
  - User intent: Ensure trained robust candidate models are actually saved; avoid runs that only emit metrics.
  - Actions taken: Updated `scripts/train_protocol_multimodel_robust_matrix.py` CLI so model saving is default-on (`--save-models` now `default=True`) and added explicit opt-out flag `--no-save-models`; updated both launchers (`train_protocol_multimodel_robust_matrix.sbatch`, `train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch`) to pass `--save-models` explicitly.
  - Validation: `py -3 -m py_compile scripts/train_protocol_multimodel_robust_matrix.py` passed; `.venv39` `scripts/test_train_protocol_multimodel_robust_matrix.py` passed (20 tests).
  - Outcome: New runs will persist robust candidate model artifacts unless explicitly disabled.
- 2026-03-14 11:25 (Europe/Berlin) - Full run-result interpretation + protocol-level finalist comparison captured
  - User intent: Confirm final outcome of run `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108`, pick between `xgboost__E` and `catboost__C`, inspect per-protocol metrics, and clarify model artifact availability.
  - Actions taken:
    - Reviewed authoritative outputs: `decision_table_global.csv`, `stability_check_global.csv`, `stability_consistency_summary.csv`, `matrix_summary.json`, `escalation_recommendation.json`.
    - Confirmed objective status: coarse/global had 10/10 gate-pass candidates; stability consistency had two all-seed-pass groups (`xgboost__E`, `catboost__C`) and one fail (`xgboost__C` with `fpr_only`).
    - Confirmed escalation state: `stable_candidate_ready` with no hard-cap trigger.
    - Produced explicit per-protocol, non-aggregated metric comparison for finalists:
      - `xgboost__E`: wifi/mqtt/bluetooth gate-pass rows all true.
      - `catboost__C`: wifi/mqtt/bluetooth gate-pass rows all true.
      - Compared columns used in decision tables: `clean_f1`, `clean_fpr`, `attacked_benign_fpr`, `adv_malicious_recall`, `robust_f1`.
    - Clarified metric persistence limitation: clean precision/clean recall are not persisted in decision tables for this run.
    - Clarified artifact availability:
      - Base-run models exist under `reports/full_gpu_hpo_models_20260306_195851/models`.
      - Robust candidate model files in run `..._20260314_003108` were not saved (`saved_model` empty), so exact robust `xgboost__E` weights are not recoverable from that completed run.
  - Outcome:
    - Decision recommendation was made contextually:
      - If prioritizing global rank/performance table: `xgboost__E`.
      - If prioritizing worst-case stability-side FPR margin: `catboost__C`.
    - User was informed that new runs after the `--save-models` default patch will persist robust candidate artifacts.
- 2026-03-14 11:40 (Europe/Berlin) - Robust-training semantics and metric-scope clarification
  - User intent: Confirm whether current training is still "general clean training plus robustness" and whether outputs reflect clean-data metrics too.
  - Actions taken: Mapped the robust-matrix stage order (clean warm-fit -> attack/hard-negative generation -> robust refit -> threshold selection) and verified metric columns persisted in protocol/global decision tables.
  - Outcome: Confirmed models are first trained on clean sampled data and then hardened with attack-generated examples; exported decision metrics cover both clean and attacked behavior (`clean_f1`, `clean_fpr`, `attacked_benign_fpr`, `adv_malicious_recall`, `robust_f1`). Noted that clean precision/clean recall are not currently persisted in those decision tables.
  - Next session should start with: Add clean precision/recall persistence to decision outputs if required for reporting, and rerun with artifact-saving enabled for final model export.
- 2026-03-14 17:07 (Europe/Berlin) - New artifact-persistent run reviewed (`..._20260314_112105`)
  - User intent: Continue once the new server run finished and determine practical model outcome.
  - Actions taken: Reviewed authoritative outputs (`decision_table_global.csv`, `stability_check_global.csv`, `stability_consistency_summary.csv`, `matrix_summary.json`, `escalation_recommendation.json`), verified saved model artifacts now exist under candidate/stability directories, and compared coarse vs stability behavior.
  - Outcome: Escalation remains `stable_candidate_ready`; only `catboost__E` is consistent gate-pass across all stability seeds. `xgboost__A` and `xgboost__C` pass coarse global ranking but fail stability due FPR (`fpr_only`), so they are not stable deployment picks under current gates.
  - Next session should start with: Package/promote `catboost__E` artifacts for deployment baseline, or run an explicit xgboost-only recovery pass if xgboost must be retained.
- 2026-03-14 17:20 (Europe/Berlin) - Full local test predictions + requested metric table for `catboost__E`
  - User intent: Run predictions on test data with the three protocol models and return `precision, recall, f1, fpr, roc_auc, pr_auc, tp, tn, fp, fn` plus threshold.
  - Actions taken: Loaded `coarse_*_catboost_E_seed_42` models for wifi/mqtt/bluetooth, applied protocol-specific thresholds from `decision_table_protocol_*.csv`, performed protocol-routed inference over `data/merged/metadata_test.csv`, and computed requested metrics locally (including global routed aggregate).
  - Outcome: Output table saved to `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/catboost_E_test_predictions_metrics_with_thresholds.csv`; MQTT `fpr` and `roc_auc` are undefined due zero benign negatives in MQTT test split.
  - Next session should start with: Use this table for reporting and, if needed, add a small exporter for markdown/LaTeX-ready thesis tables.
- 2026-03-14 18:02 (Europe/Berlin) - Generated chronological thesis journey document with embedded report tables
  - User intent: Produce a very long thesis-style document that follows the real experiment timeline, explains the reasoning behind each next step, and includes result tables from the chronologically ordered `reports/` folders.
  - Actions taken: Re-read `context if youre an ai.md`; inventoried the authoritative report directories and extracted metrics, decision, stability, hardening, robustness, and audit tables; created `Thesis/generate_chronological_experiment_journey.py`; rendered `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md`.
  - Outcome: New document `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md` now narrates the full path from data preparation and EDA through leakguard, realistic robustness, WiFi hardening, robust-matrix stabilization, and final `catboost__E` deployment selection, with embedded tables drawn from the saved artifacts.
  - Next session should start with: Review the markdown against thesis chapter needs and, if necessary, convert the selected tables and narrative into the final LaTeX/Word layout.
- 2026-03-14 18:19 (Europe/Berlin) - Expanded chronological thesis chapter with appendix-level evidence
  - User intent: Continue the thesis-document task and make the chapter significantly longer with more chronology, more tables, and clearer explanation of each pivot.
  - Actions taken: Extended `Thesis/generate_chronological_experiment_journey.py` to load additional EDA, robustness-summary, matrix-summary, stability-check, data-cap, realism-profile, and external-benign artifacts; added generated sections for a dated decision log, extra EDA context, cross-stage metric progressions, robust-matrix campaign evolution, family/gate definitions, realism and feasibility constraints, stability envelope, thesis caveats, and engineering interventions; rerendered and reviewed `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md`.
  - Outcome: `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md` is now an expanded 788-line thesis chapter with materially stronger evidence for the final `catboost__E` deployment decision.
  - Next session should start with: Convert the expanded markdown into the final thesis formatting target or further formalize the prose if the university chapter style requires it.
- 2026-03-14 18:59 (Europe/Berlin) - Produced a thesis-style chapter with a Word deliverable and script-verified prose
  - User intent: Turn the chronology into an actual document, reduce unnecessary tables, focus more on polished writing, and verify the narrative against the Python scripts used in the project.
  - Actions taken: Re-read `Thesis/CHRONOLOGICAL_EXPERIMENT_JOURNEY.md` and the project context; reverse-engineered the core methodology scripts covering merge, baselines, full GPU, HPO, leakguard, explainability, robustness, WiFi hardening, WiFi rebalance, and the protocol-wide robust matrix; created a document-generation venv at `Thesis/.venv-docs`; wrote `Thesis/generate_thesis_experiment_chapter.py`; rendered `Thesis/THESIS_EXPERIMENT_CHAPTER.md` and `Thesis/THESIS_EXPERIMENT_CHAPTER.docx`; validated generation with `py_compile`.
  - Outcome: The project now includes a Word-based thesis chapter and a corresponding markdown source with fewer but more decision-relevant tables, stronger narrative transitions between stages, and direct script-level verification of the described methodology.
  - Next session should start with: Review the `.docx` against the university chapter template and decide whether to keep it as the working chapter or perform another pass for stricter academic style and template-specific formatting.
