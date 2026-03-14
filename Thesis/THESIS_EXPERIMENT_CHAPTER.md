# Experimental Development and Final Selection of the Flow-Based IoMT IDS

Script-verified thesis chapter reconstructed from saved results, chronological context, and reverse-engineered training and evaluation code

# 1. Chapter Aim and Evidence Base

This chapter turns the experiment log into a defensible thesis narrative. The goal is not to repeat every artifact that exists under reports/. The goal is to explain, in chronological order, why each experimental decision was taken, what the saved results revealed at that point, and why those results forced the next step. The evidence base for the chapter is therefore deliberately narrower than the total artifact set: only the tables that changed the direction of the project are retained in the main body, while the finer implementation details are pushed into prose and the script-verification appendix.

Three sources are treated as authoritative. The first is the chronological context log, which records the reasoning and the sequence of pivots. The second is the result folders under reports/, which provide the saved metrics and thresholds used at the time. The third is the Python code itself. I re-read the merge, training, thresholding, robustness, and matrix-selection scripts so that this document reflects what the pipeline actually did rather than what would be convenient to claim afterwards.

**Table 1. Chronological structure of the experimental campaign**

| Date | Phase | Verified script(s) | Authoritative artifact | Decision carried forward |
| --- | --- | --- | --- | --- |
| 2026-03-05 | Dataset merge and advanced EDA | merge_ciciomt_with_metadata.py; generate_advanced_eda_report.py | reports/eda_advanced_20260305_231027 | Establish whether the problem is learnable and whether protocol slices must be treated separately. |
| 2026-03-05 | Deterministic reduced-sample baselines | train_baseline_models_stdlib.py | reports/baseline_models_stdlib_20260305_234858 | Spend GPU budget only after proving the feature space already carries usable signal. |
| 2026-03-06 | Full-data GPU clean modeling | train_full_gpu_models.py | reports/full_gpu_models_20260306_001638 | Measure how much performance survives when the whole dataset and stricter thresholding are used. |
| 2026-03-06 | Constrained GPU HPO | train_hpo_gpu_models.py | reports/full_gpu_hpo_models_20260306_134806 | Optimize under low-FPR intent rather than raw clean F1. |
| 2026-03-06 | Protocol-routed HPO | train_hpo_gpu_models.py | reports/full_gpu_hpo_models_20260306_153556 | Split the problem by protocol after the global model still looked operationally uneven. |
| 2026-03-06 | Split repair and leakguard | train_hpo_gpu_models_fprfix.py; train_hpo_gpu_models_leakguard.py | reports/full_gpu_hpo_models_20260306_172441 -> 195851 | Convert impressive routed metrics into trustworthy metrics by hardening validation and leakage controls. |
| 2026-03-07 to 2026-03-09 | Explainability and realistic robustness | xgb_protocol_ids_utils.py; generate_xgb_explainability_artifacts.py; evaluate_xgb_robustness.py | reports/full_gpu_hpo_models_20260306_195851/xgb_explainability and xgb_robustness_realistic_full_20260308_212054 | Determine whether the clean low-FPR model remains credible under constrained feature-space attack. |
| 2026-03-09 | WiFi hardening and rebalance | train_wifi_robust_hardening.py; train_wifi_robust_rebalance_matrix.py | reports/...wifi_robust_v1_* and ...wifi_rebalance_matrix_v1_20260309_204053 | Repair the benign-side WiFi weakness revealed by robustness evaluation. |
| 2026-03-10 to 2026-03-14 | Protocol-wide robust matrix and stability reruns | train_protocol_multimodel_robust_matrix.py | reports/...protocol_multimodel_robust_matrix_v1_20260314_112105 | Select the deployment baseline by gate passing, stability, and saved-artifact availability rather than by coarse rank alone. |

_Note: Only milestones that materially changed the research direction are kept in the chapter body. Appendix A maps the scripts to the exact methodological consequences in more detail._

The chronology in Table 1 also clarifies the scope of the contribution. This project did not move in a straight line from one clean leaderboard to the next. It began as a learnability question, became a protocol-heterogeneity question, then became a robustness question, and finally became a stability and artifact-selection question. That sequence matters because the final model would have been different if the campaign had stopped after any earlier stage.

# 2. Dataset Construction and EDA Defined the Real Problem

The first non-trivial step was not model training but dataset construction. The script merge_ciciomt_with_metadata.py merged the CICIoMT CSV files into metadata_train.csv and metadata_test.csv while deriving stable metadata fields such as source_relpath, source_modality, protocol_hint, device, scenario, attack_name, attack_family, label, and source_row_index. Attack captures preserved folder-based train/test assignment when it existed. Profiling traffic without explicit split folders was assigned by deterministic hashing. This design mattered because every later claim about protocol-aware training, family-wise slicing, leakage prevention, and routed inference depends on those metadata columns being correct and reproducible.

Once the merged tables existed, the advanced EDA immediately showed that the classification task was not a clean, balanced binary benchmark. The dataset contained 8,669,683 attack rows versus 650,362 benign rows, meaning the positive class represented 93.02% of all observations. The dominant attack mass was concentrated in the DDoS family, followed by DoS and Recon. In other words, a high global F1 score would be easy to inflate unless the analysis paid attention to false positives, minority protocols, and family concentration.

**Table 2. Protocol composition and train-test share after metadata merge**

| Protocol | Rows | Benign rows | Attack rows | Attack share | Train share | Test share |
| --- | --- | --- | --- | --- | --- | --- |
| wifi | 8,640,752 | 422,731 | 8,218,021 | 95.11% | 92.39% | 94.21% |
| mqtt | 524,217 | 197,564 | 326,653 | 62.31% | 6.00% | 3.86% |
| bluetooth | 155,076 | 30,067 | 125,009 | 80.61% | 1.61% | 1.93% |

_Note: The large WiFi share explains why global metrics alone were never sufficient. A protocol-specific failure could remain operationally serious while barely moving the overall score._

Table 2 made the central design pressure visible very early. WiFi dominated the corpus, MQTT was much smaller, and Bluetooth was both the smallest and structurally different slice. That asymmetry implied two things. First, low-FPR thresholding would matter at least as much as raw recall because even a small false positive rate on the dominant benign slice would be costly in practice. Second, protocol-specific analysis was not an optional diagnostic; it was the correct problem formulation. The EDA also confirmed strong feature signal in flow statistics such as flag counts, variance-derived measures, and volume features, so the flow-only approach was worth pursuing without payload inspection.

# 3. Reduced-Sample Baselines Proved Learnability Before GPU Scaling

The project did not jump directly from EDA into expensive GPU experiments. The script train_baseline_models_stdlib.py was written specifically to establish a cheap and deterministic learnability check. It used stable hashing of source_relpath and source_row_index to route a reproducible subset of rows into validation, reservoir sampling to cap train/validation/test volumes by class, and training-time standardization computed only from the sampled training set. That design kept the baseline stage lightweight without relaxing the separation discipline needed later.

**Table 3. Reduced-sample baseline leaderboard**

| Model | Threshold | Precision | Recall | F1 | FPR | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_sgd | 0.396488 | 0.9890 | 0.9866 | 0.9878 | 1.100% | 0.9968 |
| mlp_1hidden | 0.154556 | 0.9787 | 0.9954 | 0.9870 | 2.171% | 0.9988 |
| adaboost_stumps | 0.553930 | 0.9550 | 0.9706 | 0.9627 | 4.571% | 0.9943 |
| tree_stump | 1.000000 | 0.8094 | 0.9447 | 0.8719 | 22.243% | 0.8611 |

_Note: These results came from the reduced deterministic sample, not from the full dataset._

Even on the reduced sample, the signal was already clear. The logistic baseline reached an F1 of 0.9878 with an FPR of 1.100%, while the single stump model collapsed to an F1 of 0.8719 and an FPR of 22.24%. That contrast mattered more than the exact ordering. It showed that the feature space was informative enough that stronger families and full-data training were worth the compute, but it also warned that overly shallow models would not survive protocol heterogeneity.

This stage therefore answered the first research question: yes, the flow features carried real IDS signal. It did not answer the deployment question. The baseline sample was too small, too clean, and too detached from the eventual operating point. The next step was therefore not to tune the baseline harder, but to move to full-data GPU models under an explicit threshold policy.

# 4. Full-Data GPU Modeling Improved Scores but Exposed an Operational Weakness

The next stage was implemented in train_full_gpu_models.py. That script used a deterministic validation split and selected the operating threshold by maximizing validation recall subject to FPR <= 1%. This is an important detail because it means the early GPU runs were already trying to behave like an IDS, not like a generic classifier. The test metrics therefore revealed something scientifically useful: even when the validation threshold was chosen under a one-percent false-positive ceiling, the deployed test-side FPR was still materially higher.

**Table 4. Clean-model progression before protocol routing**

| Stage | Representative model | F1 | FPR | ROC-AUC | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Reduced deterministic sample | logistic_sgd | 0.9878 | 1.100% | 0.9968 | Cheap proof that learnability exists |
| Full-data GPU clean run | complex_mlp | 0.9984 | 3.28% | 0.9995 | Best clean global score, but still too many false positives |
| First constrained GPU HPO | complex_mlp_tuned | 0.9980 | 6.01% | 0.9995 | Optimization improved, deployment credibility did not |
| First constrained GPU HPO | xgboost_tuned | 0.9980 | 9.76% | 0.9998 | Low-FPR objective on validation still did not prevent test-side FPR drift |

_Note: The first two GPU stages improved headline metrics, but neither eliminated the gap between low-FPR intent on validation and operational false positives on test._

Table 4 explains why the thesis could not end at the first strong GPU result. The full-data complex MLP reached an F1 of 0.9984 and an ROC-AUC of 0.9995, but its test FPR was still 3.28%. The first GPU HPO stage, implemented in train_hpo_gpu_models.py, made the objective more sophisticated: it optimized validation F1 plus a PR-AUC bonus and penalized gaps above the target FPR. Yet the best tuned MLP still landed at 6.01% test FPR. These are strong clean classification results, but they are not yet a deployment-grade operating point.

More importantly, the protocol slices showed why the global score was misleading. In the first full-data clean run, the representative MLP produced a Bluetooth FPR of 17.33% while the WiFi slice sat at 1.04%. A single thresholded global score could therefore look excellent while still being operationally uneven. That is the point at which protocol routing stopped being a nice-to-have extension and became the next necessary experiment.

# 5. Protocol Routing Was the First Major Structural Breakthrough

Protocol routing was introduced because the earlier results suggested that WiFi, MQTT, and Bluetooth were not simply three interchangeable subsets of one traffic distribution. The HPO code already supported per-protocol training and thresholding, so the next logical experiment was to let each protocol keep its own model and threshold. When this was first executed in the 20260306_153556 run, the improvement was not marginal; it changed the shape of the problem.

**Table 5. Why protocol routing mattered**

| Stage | Model | Global F1 | Global FPR | Bluetooth FPR | WiFi FPR | What changed |
| --- | --- | --- | --- | --- | --- | --- |
| Full-data clean MLP | complex_mlp | 0.9984 | 3.28% | 17.33% | 1.04% | Global score high, Bluetooth still expensive |
| Initial protocol-routed HPO | xgboost_tuned__protocol_routed | 0.9995 | 0.27% | 0.015% | 0.31% | Routing removes most protocol-specific pain |
| Leakguard routed HPO | xgboost_tuned__protocol_routed | 0.9991 | 0.014% | 0.000% | 0.017% | Routing survives stronger split controls and leakage checks |

_Note: Bluetooth and WiFi are shown because they carried the decisive operational contrast. MQTT was already strong in the clean stages and did not drive the routing decision._

The magnitude of the change is what forced the next methodological reaction. Bluetooth false positives dropped from 17.33% in the clean MLP stage to 0.015% under routed XGBoost. WiFi also improved materially. Those gains were too large to accept uncritically. In IDS work, a difficult slice becoming almost perfect is exactly the point where one has to check whether the validation protocol is flattering the model. The correct scientific response was therefore skepticism, not celebration.

# 6. Split Repair and Leakguard Turned Good Numbers into Trustworthy Numbers

The suspicion raised by the first routed results directly motivated train_hpo_gpu_models_fprfix.py. That script changed the validation construction from a simpler hashed split into a deterministic protocol-and-class-aware split with file-level grouping as the primary behavior. Only when a stratum collapsed to a single source file was row-level fallback allowed. This mattered because it reduced the chance that a tiny or structurally narrow protocol slice would look cleaner than it really was.

The next script, train_hpo_gpu_models_leakguard.py, tightened the pipeline further. It refused to rely on positional feature slicing unless the first twenty metadata columns matched the expected prefix exactly. It explicitly banned metadata and label-like fields from entering the feature set. It also checked that the train and test source_relpath sets were disjoint so that the same capture file could not silently appear on both sides. This stage is the methodological trust anchor of the whole campaign, because after it, strong results could be interpreted as model behavior rather than as schema luck.

The leakguard run left two important signals. First, the routed XGBoost system still achieved an F1 of 0.9991 while pushing global FPR down to 0.014%. Second, the weighted ensemble obtained the highest validation objective but a much higher test FPR of 1.53%. That split in behavior is why the project used the routed XGBoost stack as the base platform for explainability and robustness. It was not simply the clean winner; it was the clean low-FPR model whose behavior was easiest to audit and extend.

# 7. Explainability Was Used as an Audit Tool, Not Cosmetic Decoration

Once the leakguard-routed XGBoost pipeline existed, the next question was whether its decisions could be inspected in a way that supported a healthcare IDS argument. The explainability utilities in xgb_protocol_ids_utils.py and generate_xgb_explainability_artifacts.py did two useful things. They exposed routed predictions with protocol-specific thresholds, and they saved both global and local contribution artifacts for the representative true-positive, true-negative, false-positive, and false-negative cases.

The resulting feature rankings were reassuring. The highest-contribution features were not leaked metadata fields, because leakguard had already forbidden those. Instead, they were plausible flow aggregates such as Tot sum, Header_Length, and Number on Bluetooth, together with analogous traffic-volume and rate features on WiFi and MQTT. This did not prove causal correctness, but it did show that the model was building its decisions from network-flow behavior rather than from accidental identifiers.

More importantly, the explainability stage prepared the project for the next pivot. Because the routed model could now return per-protocol predictions and local contributions, it became feasible to attack the system in a controlled way and then interpret where the model was brittle. The interpretability work was therefore not a presentation layer. It was part of the transition from clean evaluation to adversarial evaluation.

# 8. Realistic Robustness Changed the Definition of Success

The robustness stage was where the thesis changed character. The script evaluate_xgb_robustness.py did not apply unconstrained adversarial noise. It locked semantic features, respected lower and upper bounds, and enforced relation-aware realism constraints before scoring candidate perturbations. It also attacked both malicious and benign traffic. That second choice turned out to be decisive. Malicious-side evasion under the realistic query attack degraded performance only modestly. Benign-side perturbation on WiFi, however, revealed a much more serious operational weakness.

**Table 6. WiFi benign-side degradation under realistic query attack**

| Epsilon | Attacked benign FPR | Precision | F1 | Delta F1 | Mean queries |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.00% | 1.0000 | 0.9991 | 0.0000 | 0.0 |
| 0.01 | 22.54% | 0.8158 | 0.8978 | -0.1013 | 123.2 |
| 0.02 | 30.02% | 0.7688 | 0.8686 | -0.1305 | 114.4 |
| 0.05 | 34.52% | 0.7430 | 0.8519 | -0.1472 | 110.0 |
| 0.10 | 46.58% | 0.6818 | 0.8102 | -0.1889 | 94.0 |

_Note: All rows come from query_sparse_hillclimb_benign on the WiFi protocol slice of the leakguard-routed XGBoost system._

Table 6 is the single most important turning-point table in the project. At epsilon 0.01, attacked benign FPR jumped to 22.54%. By epsilon 0.10, it reached 46.58%, and the corresponding F1 dropped by 0.1889. This result forced a conceptual shift. Clean low-FPR performance was no longer enough. The real deployment question became whether the model could avoid being manipulated into false alarms on benign traffic while still keeping malicious recall high.

That observation explains why the rest of the project is dominated by hardening, rebalance, gates, and stability rather than by conventional clean leaderboard tuning. Once benign-side attack sensitivity became visible, the correct objective was not to maximize one more decimal place of clean F1. It was to define an acceptable operating envelope and select only the candidates that could remain inside it.

# 9. WiFi Hardening and Rebalance Converted a Weakness into a Formal Selection Problem

Because the benign-side failure was concentrated on WiFi, the next experiments became deliberately WiFi-specific. The script train_wifi_robust_hardening.py generated adversarial benign hard negatives, selected a new threshold under clean-recall and optional adversarial-recall constraints, and measured the trade-off between benign protection and malicious detection. These runs were not simply alternative retrains. They were controlled attempts to discover which kind of hardening pressure produced a useful operating point.

**Table 7. WiFi hardening sequence and the rebalance winner**

| Run | Configuration | Selected threshold | Clean FPR | Attacked benign FPR | Adv. malicious recall | Clean F1 | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260309_103936 | Conservative FPR-first hardening | 0.386555 | 0.28% | 0.50% | NA | 0.9975 | Pre-adversarial-recall gate |
| 20260309_135449 | Constraint-aware pilot | 0.122449 | 1.67% | 0.00% | 1.0000 | 0.9978 | Adds adversarial malicious recall control |
| 20260309_180250 | Full-budget hardening | 0.035059 | 7.68% | 7.67% | 0.9952 | 0.9973 | Best malicious-side recall, but too many false alarms |
| 20260309_204053 | WiFi rebalance winner (family A) | 0.308271 | 0.37% | 0.35% | 0.9900 | 0.9969 | Only WiFi candidate that passed all explicit gates |

_Note: The first run focused on false positives only. The second and third runs introduced adversarial-malicious constraints. The rebalance run then reframed the decision as a family-based gate check._

The WiFi sequence is a useful example of why more training budget does not automatically mean a better deployment point. The conservative run pushed attacked benign FPR down to 0.50% while keeping clean FPR at 0.28%, but it was not yet checking malicious-side robustness. The constraint-aware pilot was more balanced and even achieved adversarial malicious recall of 1.0000, yet its clean FPR climbed to 1.67%. The full-budget hardening run looked attractive from the malicious side, but it overshot badly on benign false alarms, landing at roughly 7.68% clean FPR and 7.67% attacked benign FPR.

This is why train_wifi_robust_rebalance_matrix.py mattered. It stopped treating the problem as one threshold on one retrain and instead evaluated model families against explicit gates. Family A passed with clean FPR 0.37%, attacked benign FPR 0.35%, and adversarial malicious recall 0.9900. Families B and C preserved very low FPR but collapsed on adversarial malicious recall. The next decision therefore became obvious: the WiFi gate-based logic had to be promoted to a protocol-wide robust matrix.

# 10. The Protocol-Wide Robust Matrix Reframed Selection Around Gates and Stability

The final stage was implemented in train_protocol_multimodel_robust_matrix.py. This script is important not because it trained more candidates than the earlier code, but because it changed the meaning of model selection. The allowed final families were restricted to XGBoost and CatBoost. Each candidate began with a clean warm-fit, received a base threshold selected from clean FPR, generated benign hard negatives and malicious adversarial examples, refit on the weighted augmented data, and then selected its final threshold using a lexicographic gate search. In other words, passing the robust operating constraints was ranked ahead of squeezing out the last clean-F1 advantage.

**Table 8. Gates enforced by the final protocol-wide robust matrix**

| Gate | Value | Meaning |
| --- | --- | --- |
| Clean FPR max | 0.50% | Upper bound for clean false positives |
| Attacked benign FPR max | 0.50% | Upper bound for benign-side adversarial false positives |
| Adversarial malicious recall min | 99.00% | Lower bound for malicious recall after attack |

_Note: The matrix used an internal attacked-benign safety margin during threshold search, but the three values shown here were the public acceptance criteria for candidate promotion._

The family logic also reflects what the project learned along the way. Families A and B are essentially precision- and balance-oriented references. Families C, D, and E exist because the search needed targeted Bluetooth-recovery behavior without giving back the WiFi false-positive improvements. That is why the final answer eventually came from family E. Family E was not an arbitrary later variant; it was created because the earlier evidence showed that the bottleneck could no longer be solved by a single generic robust retrain.

The matrix configuration in the final authoritative run also makes clear that this was a deliberate search, not an improvised sweep. The attack source mode was fixed to the XGBoost base run, the family pack was the Bluetooth-recovery fallback pack, coarse search used adaptive-fraction sampling with lightweight query budgets, and stability search reused the same split seed while increasing sample support and query budget. The project moved from ordinary tuning to controlled search design because that is what the robustness evidence demanded.

# 11. Data Audit and Feasibility Limits Were Treated as Results, Not Footnotes

The data audit performed on 2026-03-13 is one of the reasons the final chapter can be honest without being weak. It confirmed 7,669,892 training rows, 1,650,153 test rows, and a feature space of 45 flow variables. It also confirmed several structural limits that the later robustness results had already hinted at: WiFi dominance, MQTT test benign absence, Bluetooth benign concentration, and many near-constant protocol-specific features. None of those findings invalidates the final model, but each one changes how the thesis should phrase its claims.

**Table 9. Validation support and caveats by protocol**

| Protocol | Benign rows | Validation benign rows | Largest file share | Split fallback | Why it matters |
| --- | --- | --- | --- | --- | --- |
| wifi | 380,994 | 43,233 | 50.59% | none | Main operational benchmark; strong validation support but still dominant in the corpus. |
| mqtt | 197,564 | 39,513 | 100.00% | row_level_single_file | Validation has benign support, but held-out test has no benign negatives, so FPR and ROC-AUC cannot be estimated there. |
| bluetooth | 23,396 | 15,396 | 92.96% | none | Benign evidence is highly concentrated; floor repair moved 6,488 rows back into training. |

_Note: These rows come from the final authoritative robust-matrix run because those are the statistics that directly governed the final low-FPR gate checks._

The Bluetooth row in Table 9 is especially important. Before the floor repair, the final matrix had only 1,512 Bluetooth benign training rows, which is not enough support for stable low-FPR calibration. The pipeline therefore moved 6,488 rows back into training to reach a floor of 8,000. That should be presented as explicit statistical housekeeping, not as a hidden trick. The MQTT caveat is different: the issue is not missing validation support, but the lack of benign negatives in the held-out test set. That is why the final MQTT FPR and ROC-AUC entries are honestly left undefined.

This section also explains why some artifacts are absent from the final claim. External benign augmentation stayed disabled in the final authoritative run, so the promoted model should be described as the best in-project deployment baseline, not as a domain-generalized IDS across unmatched external datasets. That is a tighter claim, but it is also the claim the evidence actually supports.

# 12. Stability Reruns Determined the Final Winner

The last selection problem was no longer about finding a candidate that could pass the gates once. It was about finding a candidate that kept passing after the search was rerun, the sampling seed moved, and model artifacts were actually saved. This is the stage where a superficial leaderboard reading would have chosen the wrong winner. In the final authoritative 20260314_112105 run, the seed-42 global ranking placed XGBoost families at the top. But the stability tables tell a different story.

**Table 10. Rank on the main run versus stability across reruns**

| Candidate group | Seed-42 rank | All-seed gate pass | Seeds checked | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Primary failure reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__A | 1 | No | 9 | 1.08% | 1.23% | 99.01% | fpr_only |
| xgboost__C | 2 | No | 9 | 2.81% | 2.76% | 99.00% | fpr_only |
| catboost__E | 6 | Yes | 9 | 0.25% | 0.10% | 99.01% | none |

_Note: Seed-42 rank comes from decision_table_global.csv. Stability comes from the nine-seed consistency and stability-check outputs of the final authoritative run._

Table 10 is the most important selection table in the chapter. The rank-1 candidate on the main run was xgboost__A, and xgboost__C was rank 2. Neither survived the stability filter. Both repeatedly failed for the same reason: FPR drift. By contrast, catboost__E was only rank 6 on the seed-42 decision table, yet it was the only candidate that remained gate-pass consistent across all 9 checked seeds. Its attacked benign FPR remained between 0.02% and 0.10%, while adversarial malicious recall stayed between 99.01% and 99.12%.

The March 14 near-final run makes this point even stronger. In the earlier 20260314_003108 pass, the consistent groups were catboost__C, xgboost__E. After the artifact-persistent rerun, the final stable set became catboost__E. That shift is exactly why the project kept going. A candidate that looks stable before artifact persistence is only analytically interesting. A candidate that remains stable after artifact persistence and rerun is a deployment baseline. The escalation output of the final run captured this directly: recommendation=stable_candidate_ready, with no protocol hard cap triggered.

# 13. Final Deployment-Ready Baseline

After stability and artifact persistence were enforced, the final deployment baseline was the protocol-routed catboost__E system saved by the 20260314_112105 run. The final test metrics were computed by applying the saved protocol-specific thresholds to metadata_test.csv and evaluating the routed predictions directly. This matters because it means the closing table is not a validation artifact; it is a held-out test summary for the actual saved model family that survived the full campaign.

**Table 11. Final held-out test metrics for the promoted catboost__E deployment baseline**

| Scope | Threshold | Precision | Recall | F1 | FPR | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WiFi | 0.906667 | 1.0000 | 0.9981 | 0.9991 | 0.03% | 12 | 2,822 |
| MQTT | 0.966667 | 1.0000 | 0.9931 | 0.9966 | NA | 0 | 438 |
| Bluetooth | 0.880000 | 1.0000 | 0.9986 | 0.9993 | 0.00% | 0 | 36 |
| Protocol-routed global | protocol-specific | 1.0000 | 0.9979 | 0.9990 | 0.02% | 12 | 3,296 |

_Note: MQTT FPR and ROC-AUC remain undefined on held-out test because there are no MQTT benign negatives in the test split._

The promoted baseline achieved global precision 1.0000, recall 0.9979, F1 0.9990, and FPR 0.02%. On WiFi, only 12 false positives were produced over 1,554,597 rows. Bluetooth achieved zero false positives in the held-out test slice. MQTT preserved very strong recall and precision, but its benign-side error rates cannot be estimated honestly on the test split because no benign negatives exist there.

The right thesis claim is therefore precise: catboost__E is the best deployment-ready in-project baseline produced by this campaign, not because it was the flashiest coarse winner, but because it is the only saved candidate that survived the complete chain of evidence. That chain included metadata-aware data assembly, protocol-aware evaluation, leakguard validation, realistic robustness, WiFi hardening, protocol-wide robust gates, multi-seed stability, and explicit artifact persistence. The fact that external benign augmentation remained disabled (False) should also be stated clearly, because it keeps the final claim aligned with the dataset actually used.

# 14. Discussion: What the Journey Actually Demonstrated

Several conclusions follow from the chronology. First, data engineering was not a preparatory chore. It shaped the scientific result. The metadata merge enabled protocol-aware and family-aware analysis, and the later leakguard checks ensured that those metadata fields could guide evaluation without leaking into the feature set. Second, protocol heterogeneity was not a minor dataset characteristic. It changed which model families looked acceptable and ultimately justified routed inference as the correct operational design.

Third, clean metrics were necessary but insufficient. The clean GPU and early HPO runs already showed that the classification task was easy to score well on. What separated the final result from the earlier ones was the shift from clean optimization to robust operating-envelope control. That shift only happened after benign-side WiFi attack exposed the real false-positive risk. Fourth, stability was not a luxury add-on. It directly changed the winner. If the project had selected by single-run rank, it would have promoted an XGBoost family that failed under rerun.

The final methodological contribution is therefore broader than a single model artifact. The project built a reproducible selection process for a healthcare IoMT IDS under class imbalance, protocol heterogeneity, low-FPR constraints, and realistic adversarial pressure. That process is what makes the final model defensible. Without the journey, the ending would be a score. With the journey, it becomes a thesis result.

# Appendix A. Script Verification Map

This appendix records how the chapter text was cross-checked against the actual codebase. The point is not to document every function in every file. The point is to show which scripts changed the scientific meaning of the results and what methodological behavior was verified directly in code.

**Table 12. Script-level verification used to write this chapter**

| Script | Phase | Verified behavior | Why it mattered to the narrative |
| --- | --- | --- | --- |
| merge_ciciomt_with_metadata.py | Dataset construction | Derives metadata, protocol hints, attack families, and deterministic profiling splits | Makes later protocol-aware evaluation and leakage checks possible |
| generate_advanced_eda_report.py | EDA | Produces class balance, protocol composition, shift, and separability summaries | Establishes why low-FPR and protocol slicing must drive the modeling agenda |
| train_baseline_models_stdlib.py | Cheap learnability check | Uses stable hash validation and reservoir sampling caps without third-party ML dependencies | Proves the feature space is worth scaling before spending GPU budget |
| train_full_gpu_models.py | First full-data GPU stage | Selects threshold by maximizing recall under validation FPR <= 1% | Shows that low-FPR intent on validation does not guarantee low test FPR |
| train_hpo_gpu_models.py | Constrained HPO | Optimizes validation F1 plus PR-AUC bonus with explicit FPR-gap penalty | Improves optimization discipline but still leaves deployment credibility incomplete |
| train_hpo_gpu_models_fprfix.py | Validation repair | Builds protocol-and-class-aware file-level validation with controlled row fallback | Reduces the chance that routed performance is flattered by split pathology |
| train_hpo_gpu_models_leakguard.py | Leak prevention | Checks metadata prefix, forbids leakage columns, and verifies train/test capture disjointness | Turns strong routed metrics into trustworthy routed metrics |
| xgb_protocol_ids_utils.py | Routed inference utility | Loads per-protocol models and thresholds and exposes routed_predict | Provides the operational bridge from per-protocol training to deployment and explainability |
| evaluate_xgb_robustness.py | Robustness evaluation | Applies realistic query attacks with semantic locks and relation-aware constraints | Reveals benign-side WiFi false positives as the key weakness to harden |
| train_wifi_robust_hardening.py | WiFi hardening | Generates adversarial benign hard negatives and reselects threshold under constraints | Transforms robustness from diagnosis into intervention |
| train_wifi_robust_rebalance_matrix.py | WiFi gate selection | Evaluates family variants under explicit clean-FPR, attacked-benign-FPR, and adv-recall gates | Shows that gate passing, not score chasing, is the right robust selection rule |
| train_protocol_multimodel_robust_matrix.py | Final robust search | Warm-fits clean models, augments with hard negatives and adversarial rows, then selects thresholds lexicographically by gates | Produces the stability-based final winner and the saved deployment artifact |

_Note: The appendix table is intentionally selective. It keeps only the scripts that materially changed the argument of the thesis chapter._

The chapter should therefore be read as a script-verified account of the experimentation process rather than as a retrospective summary written from memory. That is the main reason the final model claim is strong. The narrative has been made to follow the code and the saved artifacts, not the other way around.
