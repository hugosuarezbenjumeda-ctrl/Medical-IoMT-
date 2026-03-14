# Chronological Thesis Narrative of the IoMT IDS Development Journey

This document reconstructs the project as a thesis-style research journey using two authoritative sources: the dated reasoning in `context if youre an ai.md` and the saved results under `reports/`. Each section explains what problem was being addressed at that point, what the saved results showed, and why those results forced the next step.

The point of this document is not only to state the final answer. It is to make the path to the final answer defensible.

## Chronology at a glance

| Date | Phase | Artifact | Next step |
| --- | --- | --- | --- |
| 2026-03-05 | Data merge and advanced EDA | reports/eda_advanced_20260305_231027 | Establish a fast baseline before scaling up. |
| 2026-03-05 | Reduced-sample baseline matrix | reports/baseline_models_stdlib_20260305_234858 | Move to full-data GPU training. |
| 2026-03-06 | Full-data GPU training and first HPO | reports/full_gpu_models_20260306_001638 and reports/full_gpu_hpo_models_20260306_134806 | Make training protocol-aware and repair FPR behavior. |
| 2026-03-06 | Protocol-routed HPO, FPR-fix, split repair, leakguard | reports/full_gpu_hpo_models_20260306_153556 -> ...172441 -> ...180951 -> ...195851 | Use the leakguard base run for explanation and robustness analysis. |
| 2026-03-07 | Explainability and UI | reports/full_gpu_hpo_models_20260306_195851/xgb_explainability | Attack the model under realistic constraints. |
| 2026-03-08 to 2026-03-09 | Realistic robustness and WiFi hardening | reports/...xgb_robustness_realistic_full_20260308_212054 and wifi_robust_v1_* | Rebalance hardening strategies and then move to protocol-wide robust selection. |
| 2026-03-09 | WiFi rebalance matrix | reports/full_gpu_hpo_models_20260306_195851_wifi_rebalance_matrix_v1_20260309_204053 | Build a protocol-wide robust matrix. |
| 2026-03-10 to 2026-03-14 | Protocol-wide robust matrix and stability passes | reports/...20260310_105248 -> ...121948 -> ...180757 -> ...200922 -> ...003108 -> ...112105 | Promote the only stable artifact-persistent deployment baseline. |
| 2026-03-13 | Data-first audit | reports/data_audit_20260313_124851 | Use data reality to justify stability-aware selection and explicit thesis caveats. |

## 1. Advanced EDA defined the real problem before any serious training

The merged metadata table immediately showed that this was not a balanced, homogeneous binary classification problem. The saved EDA tables below explain why the later modeling pipeline became protocol-aware, threshold-aware, and robustness-aware.

### Table 1. Global class balance
| Label | Rows | Share |
| --- | --- | --- |
| benign | 650,362 | 0.0698 |
| attack | 8,669,683 | 0.9302 |

### Table 2. Protocol composition by label
| Protocol | Rows | Benign rows | Attack rows | Benign ratio | Attack ratio |
| --- | --- | --- | --- | --- | --- |
| wifi | 8,640,752 | 422,731 | 8,218,021 | 0.0489 | 0.9511 |
| mqtt | 524,217 | 197,564 | 326,653 | 0.3769 | 0.6231 |
| bluetooth | 155,076 | 30,067 | 125,009 | 0.1939 | 0.8061 |

### Table 3. Attack-family concentration
| Attack family | Rows | Share |
| --- | --- | --- |
| DDoS | 6,097,614 | 0.6542 |
| DoS | 2,290,990 | 0.2458 |
| Benign | 650,362 | 0.0698 |
| Recon | 131,402 | 0.0141 |
| Other | 125,009 | 0.0134 |
| Spoofing | 17,791 | 0.0019 |
| Malformed | 6,877 | 0.0007 |

### Table 4. Train-test protocol shift
| Protocol | Train rows | Test rows | Train share | Test share | Test minus train |
| --- | --- | --- | --- | --- | --- |
| mqtt | 460,502 | 63,715 | 0.0600 | 0.0386 | -0.0214 |
| wifi | 7,086,155 | 1,554,597 | 0.9239 | 0.9421 | 0.0182 |
| bluetooth | 123,235 | 31,841 | 0.0161 | 0.0193 | 0.0032 |

### Table 5. Most separable sampled features
| Feature | Benign mean | Attack mean | Cohen's d | Benign zero rate | Attack zero rate |
| --- | --- | --- | --- | --- | --- |
| psh_flag_number | 0.3562 | 0.0119 | -3.7521 | 0.2152 | 0.9466 |
| Variance | 0.8273 | 0.0703 | -3.7065 | 0.0910 | 0.7923 |
| rst_count | 2,310.65 | 7.3843 | -3.2502 | 0.1546 | 0.8215 |
| ack_flag_number | 0.7539 | 0.0742 | -2.9486 | 0.1546 | 0.8417 |
| Max | 417.0126 | 68.0446 | -2.0599 | 0.0000 | 0.0000 |
| Magnitue | 17.4404 | 10.2028 | -1.9295 | 0.0000 | 0.0000 |
| Std | 98.8050 | 4.7860 | -1.8043 | 0.0450 | 0.7731 |
| Radius | 139.3289 | 6.6782 | -1.7996 | 0.0455 | 0.7772 |

The EDA results established three facts that remained true for the rest of the project. First, the dataset was attack-heavy, so low FPR thresholding had to matter at least as much as raw F1. Second, WiFi dominated the corpus, so protocol slices were necessary to keep the dominant traffic from hiding minority-protocol failure modes. Third, the feature space clearly contained usable signal, so a flow-based IDS was worth pursuing without payload inspection.

The next step followed logically: before committing full GPU budget, confirm quickly that the problem is learnable with a small baseline matrix.

## 2. Reduced-sample baselines showed that the feature space already carried real IDS signal

### Table 6. Reduced-sample global baselines
| Model | Threshold | Precision | Recall | F1 | FPR | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_sgd | 0.3965 | 0.9890 | 0.9866 | 0.9878 | 0.0110 | 0.9968 |
| mlp_1hidden | 0.1546 | 0.9787 | 0.9954 | 0.9870 | 0.0217 | 0.9988 |
| adaboost_stumps | 0.5539 | 0.9550 | 0.9706 | 0.9627 | 0.0457 | 0.9943 |
| tree_stump | 1.0000 | 0.8094 | 0.9447 | 0.8719 | 0.2224 | 0.8611 |

### Table 7. Reduced-sample protocol slices
| Model | Protocol | Rows | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_sgd | bluetooth | 1,029 | 0.9829 | 1.0000 | 0.9914 | 0.0022 |
| logistic_sgd | mqtt | 261 | 1.0000 | 0.7586 | 0.8627 | 0.0000 |
| logistic_sgd | wifi | 12,710 | 0.9888 | 0.9953 | 0.9920 | 0.0123 |
| tree_stump | bluetooth | 1,029 | 0.1118 | 1.0000 | 0.2010 | 1.0000 |
| tree_stump | mqtt | 261 | 1.0000 | 0.0651 | 0.1223 | 0.0000 |
| tree_stump | wifi | 12,710 | 0.9097 | 0.9784 | 0.9428 | 0.1057 |
| adaboost_stumps | bluetooth | 1,029 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaboost_stumps | mqtt | 261 | 1.0000 | 0.7701 | 0.8701 | 0.0000 |
| adaboost_stumps | wifi | 12,710 | 0.9537 | 0.9953 | 0.9741 | 0.0526 |
| mlp_1hidden | bluetooth | 1,029 | 0.5111 | 1.0000 | 0.6765 | 0.1204 |
| mlp_1hidden | mqtt | 261 | 1.0000 | 0.9732 | 0.9864 | 0.0000 |
| mlp_1hidden | wifi | 12,710 | 0.9937 | 0.9962 | 0.9949 | 0.0069 |

### Table 8. Family-wise behavior of the reduced-sample winner
| Attack family | Rows | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- |
| Benign | 7,000 | 0.0000 | 0.0000 | 0.0000 | 0.0110 |
| DDoS | 4,874 | 1.0000 | 0.9928 | 0.9964 | 0.0000 |
| DoS | 1,887 | 1.0000 | 0.9809 | 0.9904 | 0.0000 |
| Malformed | 9 | 1.0000 | 0.3333 | 0.5000 | 0.0000 |
| Other | 115 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Recon | 108 | 1.0000 | 0.9074 | 0.9515 | 0.0000 |
| Spoofing | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`logistic_sgd` reached an F1 of 0.9878 with FPR 0.0110. That did not mean the thesis was solved. It meant the feature space was already good enough that stronger families and full-data training were worth the cost. The next step was therefore to scale the training pipeline rather than spend more effort squeezing tiny gains out of shallow baselines.

## 3. Full-data GPU training exposed the real operational weakness: false positives on specific protocols

### Table 9. First full-data GPU global metrics
| Model | Threshold | Precision | Recall | F1 | FPR | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| complex_mlp | 0.0511 | 0.9990 | 0.9977 | 0.9984 | 0.0328 | 0.9995 |
| catboost | 0.2117 | 0.9965 | 0.9993 | 0.9979 | 0.1153 | 0.9998 |
| xgboost | 0.2062 | 0.9962 | 0.9992 | 0.9977 | 0.1269 | 0.9998 |

### Table 10. First full-data GPU protocol slices
| Model | Protocol | Rows | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | bluetooth | 31,841 | 0.8219 | 1.0000 | 0.9023 | 0.8174 |
| xgboost | mqtt | 63,715 | 1.0000 | 0.9979 | 0.9989 | 0.0000 |
| xgboost | wifi | 1,554,597 | 0.9995 | 0.9993 | 0.9994 | 0.0166 |
| catboost | bluetooth | 31,841 | 0.8411 | 1.0000 | 0.9137 | 0.7129 |
| catboost | mqtt | 63,715 | 1.0000 | 0.9976 | 0.9988 | 0.0000 |
| catboost | wifi | 1,554,597 | 0.9995 | 0.9993 | 0.9994 | 0.0197 |
| complex_mlp | bluetooth | 31,841 | 0.9561 | 1.0000 | 0.9775 | 0.1733 |
| complex_mlp | mqtt | 63,715 | 1.0000 | 0.9880 | 0.9940 | 0.0000 |
| complex_mlp | wifi | 1,554,597 | 0.9997 | 0.9981 | 0.9989 | 0.0104 |

The global results were outstanding. The complex MLP reached F1 0.9984. But the protocol slices showed why global scores were insufficient for an IDS thesis. Bluetooth behaved very differently from WiFi and MQTT, and false-positive behavior remained much less comfortable there than the global row suggested.

That is why the project did not stop at "full-data training works." The next move was to tune model families under an explicit low-FPR objective.

## 4. First HPO pass improved optimization sophistication, not deployment credibility

### Table 11. First constrained HPO global metrics
| Model | Selection score | Threshold | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| complex_mlp_tuned | 0.7975 | 0.0679 | 0.9982 | 0.9978 | 0.9980 | 0.0601 |
| xgboost_tuned | 0.6475 | 0.4096 | 0.9971 | 0.9990 | 0.9980 | 0.0976 |
| weighted_ensemble | 0.6475 | 0.4095 | 0.9971 | 0.9990 | 0.9980 | 0.0976 |

### Table 12. First constrained HPO protocol slices
| Model | Protocol | Rows | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost_tuned | bluetooth | 31,841 | 0.8484 | 1.0000 | 0.9180 | 0.6744 |
| xgboost_tuned | mqtt | 63,715 | 1.0000 | 0.9969 | 0.9984 | 0.0000 |
| xgboost_tuned | wifi | 1,554,597 | 0.9998 | 0.9991 | 0.9995 | 0.0055 |
| complex_mlp_tuned | bluetooth | 31,841 | 0.9089 | 1.0000 | 0.9523 | 0.3782 |
| complex_mlp_tuned | mqtt | 63,715 | 1.0000 | 0.9886 | 0.9943 | 0.0000 |
| complex_mlp_tuned | wifi | 1,554,597 | 0.9997 | 0.9981 | 0.9989 | 0.0093 |
| weighted_ensemble | bluetooth | 31,841 | 0.8484 | 1.0000 | 0.9180 | 0.6744 |
| weighted_ensemble | mqtt | 63,715 | 1.0000 | 0.9969 | 0.9984 | 0.0000 |
| weighted_ensemble | wifi | 1,554,597 | 0.9998 | 0.9991 | 0.9995 | 0.0055 |

The HPO machinery became better, but the deployment story did not become good enough. The selected tuned MLP still had FPR 0.0601, which was too high for the low-FPR thesis target. The context log also recorded that CatBoost search had configuration issues and the ensemble mostly mirrored XGBoost.

That forced the first major structural decision of the project: stop assuming a single global model family should solve all protocol regimes in the same way. The next stage became protocol-routed training.

## 5. Protocol-routed HPO was the first big structural breakthrough

### Table 13. Initial protocol-routed global metrics
| Model | Selection score | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| xgboost_tuned__protocol_routed | 0.9995 | 0.9999 | 0.9991 | 0.9995 |
| lightgbm_tuned__protocol_routed | 0.9995 | 0.9999 | 0.9991 | 0.9995 |
| catboost_tuned__protocol_routed | 0.9993 | 0.9998 | 0.9988 | 0.9993 |
| weighted_ensemble__protocol_routed | 0.9930 | 0.9999 | 0.9862 | 0.9930 |
| complex_mlp_tuned__protocol_routed | 0.9924 | 0.9999 | 0.9850 | 0.9924 |

### Table 14. Initial protocol-routed per-protocol model table
| Protocol | Model | Selection score | Threshold | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bluetooth | xgboost_tuned__bluetooth | 1.0000 | 0.3251 | 1.0000 | 1.0000 | 1.0000 | 0.0001 |
| bluetooth | catboost_tuned__bluetooth | 1.0000 | 0.4821 | 1.0000 | 1.0000 | 1.0000 | 0.0001 |
| bluetooth | lightgbm_tuned__bluetooth | 1.0000 | 0.5763 | 0.9999 | 1.0000 | 1.0000 | 0.0003 |
| bluetooth | complex_mlp_tuned__bluetooth | 0.3005 | 0.5074 | 0.9998 | 0.1768 | 0.3005 | 0.0001 |
| bluetooth | weighted_ensemble__bluetooth | 0.3005 | 0.6183 | 0.9998 | 0.1768 | 0.3005 | 0.0001 |
| mqtt | xgboost_tuned__mqtt | 0.9987 | 0.6961 | 1.0000 | 0.9974 | 0.9987 | 0.0000 |
| mqtt | catboost_tuned__mqtt | 0.9978 | 0.5623 | 1.0000 | 0.9957 | 0.9978 | 0.0000 |
| mqtt | lightgbm_tuned__mqtt | 0.9987 | 0.6987 | 1.0000 | 0.9975 | 0.9987 | 0.0000 |
| mqtt | complex_mlp_tuned__mqtt | 0.9973 | 0.6675 | 1.0000 | 0.9946 | 0.9973 | 0.0000 |
| mqtt | weighted_ensemble__mqtt | 0.9983 | 0.4734 | 1.0000 | 0.9966 | 0.9983 | 0.0000 |
| wifi | xgboost_tuned__wifi | 0.9995 | 0.4594 | 0.9999 | 0.9992 | 0.9995 | 0.0031 |
| wifi | catboost_tuned__wifi | 0.9993 | 0.4025 | 0.9998 | 0.9989 | 0.9993 | 0.0075 |
| wifi | lightgbm_tuned__wifi | 0.9995 | 0.4195 | 0.9999 | 0.9991 | 0.9995 | 0.0051 |
| wifi | complex_mlp_tuned__wifi | 0.9990 | 0.0498 | 0.9999 | 0.9981 | 0.9990 | 0.0052 |
| wifi | weighted_ensemble__wifi | 0.9996 | 0.4383 | 0.9999 | 0.9992 | 0.9996 | 0.0033 |

Routing by protocol changed the picture dramatically. The routed XGBoost model reached F1 0.9995, and the Bluetooth slice became almost clean enough to look solved. That was exactly why the next step had to be skeptical rather than celebratory. When a difficult slice suddenly becomes near-perfect, the correct response is to audit the validation logic and leakage controls.

## 6. FPR-fix, split repair, and leakguard turned impressive metrics into trustworthy metrics

### Table 15. WiFi-only collapse in the first FPR-fix run
| Protocol | Model | Selection score | Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| wifi | xgboost_tuned__wifi | 0.9992 | 0.2261 | 1.0000 | 0.9985 | 0.9992 |
| wifi | catboost_tuned__wifi | 0.9993 | 0.3153 | 1.0000 | 0.9986 | 0.9993 |
| wifi | complex_mlp_tuned__wifi | 0.9987 | 0.1076 | 1.0000 | 0.9973 | 0.9987 |
| wifi | weighted_ensemble__wifi | 0.9993 | 0.2122 | 1.0000 | 0.9986 | 0.9993 |

### Table 16. Split-repaired protocol/class-aware run
| Protocol | Model | Selection score | Threshold | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bluetooth | xgboost_tuned__bluetooth | 0.9914 | 0.9983 | 1.0000 | 0.9830 | 0.9914 | 0.0000 |
| bluetooth | catboost_tuned__bluetooth | 0.9900 | 0.5102 | 1.0000 | 0.9802 | 0.9900 | 0.0000 |
| bluetooth | lightgbm_tuned__bluetooth | 0.0000 | inf | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bluetooth | complex_mlp_tuned__bluetooth | 1.0000 | 0.1640 | 1.0000 | 1.0000 | 1.0000 | 0.0001 |
| bluetooth | weighted_ensemble__bluetooth | 0.5855 | 0.4347 | 0.9717 | 1.0000 | 0.9856 | 0.1100 |
| mqtt | xgboost_tuned__mqtt | 0.9985 | 0.4879 | 1.0000 | 0.9970 | 0.9985 | 0.0000 |
| mqtt | catboost_tuned__mqtt | 0.9971 | 0.6447 | 1.0000 | 0.9942 | 0.9971 | 0.0000 |
| mqtt | lightgbm_tuned__mqtt | 0.9988 | 0.4241 | 1.0000 | 0.9977 | 0.9988 | 0.0000 |
| mqtt | complex_mlp_tuned__mqtt | 0.9956 | 0.5013 | 1.0000 | 0.9913 | 0.9956 | 0.0000 |
| mqtt | weighted_ensemble__mqtt | 0.9978 | 0.3337 | 1.0000 | 0.9955 | 0.9978 | 0.0000 |
| wifi | xgboost_tuned__wifi | 0.9992 | 0.2261 | 1.0000 | 0.9985 | 0.9992 | 0.0002 |
| wifi | catboost_tuned__wifi | 0.9993 | 0.3153 | 1.0000 | 0.9986 | 0.9993 | 0.0011 |
| wifi | complex_mlp_tuned__wifi | 0.9987 | 0.1076 | 1.0000 | 0.9973 | 0.9987 | 0.0008 |
| wifi | weighted_ensemble__wifi | 0.9993 | 0.2122 | 1.0000 | 0.9986 | 0.9993 | 0.0002 |

### Table 17. Leakguard global metrics
| Model | Selection score | Validation objective | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| weighted_ensemble__protocol_routed | 1.0377 | 1.0377 | 0.9995 | 0.9985 | 0.9990 | 0.0153 |
| xgboost_tuned__protocol_routed | 1.0363 | 1.0363 | 1.0000 | 0.9981 | 0.9991 | 0.0001 |
| complex_mlp_tuned__protocol_routed | 1.0256 | 1.0256 | 1.0000 | 0.9973 | 0.9986 | 0.0007 |
| catboost_tuned__protocol_routed | 1.0253 | 1.0253 | 1.0000 | 0.9977 | 0.9988 | 0.0010 |
| lightgbm_tuned__protocol_routed | 0.7085 | 0.7085 | 1.0000 | 0.7139 | 0.8331 | 0.0000 |

### Table 18. Best leakguard protocol-specific models
| Protocol | Model | Selection score | Threshold | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bluetooth | catboost_tuned__bluetooth | 1.0500 | 0.5102 | 1.0000 | 0.9802 | 0.9900 | 0.0000 |
| mqtt | xgboost_tuned__mqtt | 1.0500 | 0.8608 | 1.0000 | 0.9959 | 0.9979 | 0.0000 |
| wifi | weighted_ensemble__wifi | 1.0368 | 0.2122 | 1.0000 | 0.9986 | 0.9993 | 0.0002 |

Table 15 is a necessary failure in the story. The first FPR-fix run collapsed to WiFi-only output, which exposed a split-construction defect. Table 16 shows the repair: once the validation logic became protocol- and class-aware with deterministic fallback, all protocols returned again. Table 17 then becomes the true anchor for the rest of the project because leakguard enforced file-level overlap checks and validation-only ranking. The project chose methodological trust over cleaner-looking but weaker guarantees.

That is why the next phase did not add yet another training family. The next phase moved into explainability and adversarial robustness on top of the leakguard base run.

## 7. Explainability made the routed model intelligible enough for thesis use

### Table 19. Top explainability features by protocol
| Protocol | Top feature 1 | Top feature 2 | Top feature 3 | Top feature 4 | Top feature 5 |
| --- | --- | --- | --- | --- | --- |
| bluetooth | Tot sum | Header_Length | Number | IAT | Max |
| mqtt | Rate | syn_count | ack_count | IAT | rst_count |
| wifi | IAT | Magnitue | Tot size | Number | rst_count |

The explainability artifacts confirmed that the protocol-routed architecture was not only operationally useful but also interpretable. Different protocols were driven by different feature groups, which supported the earlier decision to route the traffic rather than force a single global decision function.

The next step was to stop asking "why is the model making these decisions?" and start asking "how does the model fail under adaptive pressure?"

## 8. Realistic robustness changed the project from a performance study into a deployment study

### Table 20. Realistic query-attack global metrics
| Attack method | Epsilon | Objective | Budget | Targeted rows | Targeted success rate | Precision | Recall | F1 | FPR | Delta F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.0000 | none | 0.0000 | 0 | 0.0000 | 1.0000 | 0.9929 | 0.9965 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | 0.0000 | malicious_evasion | 300.0000 | 15,000 | 0.0071 | 1.0000 | 0.9929 | 0.9965 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | 0.0100 | malicious_evasion | 300.0000 | 15,000 | 0.0080 | 1.0000 | 0.9920 | 0.9960 | 0.0000 | -0.0005 |
| query_sparse_hillclimb | 0.0200 | malicious_evasion | 300.0000 | 15,000 | 0.0079 | 1.0000 | 0.9921 | 0.9960 | 0.0000 | -0.0004 |
| query_sparse_hillclimb | 0.0500 | malicious_evasion | 300.0000 | 15,000 | 0.0081 | 1.0000 | 0.9919 | 0.9960 | 0.0000 | -0.0005 |
| query_sparse_hillclimb | 0.1000 | malicious_evasion | 300.0000 | 15,000 | 0.0083 | 1.0000 | 0.9917 | 0.9958 | 0.0000 | -0.0006 |
| query_sparse_hillclimb_benign | 0.0000 | benign_side_effect | 150.0000 | 10,000 | 0.0000 | 1.0000 | 0.9929 | 0.9965 | 0.0000 | 0.0000 |
| query_sparse_hillclimb_benign | 0.0100 | benign_side_effect | 150.0000 | 10,000 | 0.1127 | 0.9297 | 0.9929 | 0.9603 | 0.1127 | -0.0362 |
| query_sparse_hillclimb_benign | 0.0200 | benign_side_effect | 150.0000 | 10,000 | 0.1501 | 0.9084 | 0.9929 | 0.9488 | 0.1501 | -0.0476 |
| query_sparse_hillclimb_benign | 0.0500 | benign_side_effect | 150.0000 | 10,000 | 0.1726 | 0.8961 | 0.9929 | 0.9421 | 0.1726 | -0.0544 |
| query_sparse_hillclimb_benign | 0.1000 | benign_side_effect | 150.0000 | 10,000 | 0.2329 | 0.8648 | 0.9929 | 0.9244 | 0.2329 | -0.0720 |

### Table 21. Protocol query metrics at baseline and epsilon 0.1
| Attack method | Protocol | Epsilon | Objective | Targeted success rate | Precision | Recall | F1 | FPR | Delta F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | bluetooth | 0.0000 | none | 0.0000 | 1.0000 | 0.9838 | 0.9918 | 0.0000 | 0.0000 |
| baseline | mqtt | 0.0000 | none | 0.0000 | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |
| baseline | wifi | 0.0000 | none | 0.0000 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | bluetooth | 0.0000 | malicious_evasion | 0.0162 | 1.0000 | 0.9838 | 0.9918 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | mqtt | 0.0000 | malicious_evasion | 0.0032 | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | wifi | 0.0000 | malicious_evasion | 0.0018 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | bluetooth | 0.1000 | malicious_evasion | 0.0162 | 1.0000 | 0.9838 | 0.9918 | 0.0000 | 0.0000 |
| query_sparse_hillclimb | mqtt | 0.1000 | malicious_evasion | 0.0060 | 1.0000 | 0.9940 | 0.9970 | 0.0000 | -0.0014 |
| query_sparse_hillclimb | wifi | 0.1000 | malicious_evasion | 0.0026 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | -0.0004 |
| query_sparse_hillclimb_benign | bluetooth | 0.0000 | benign_side_effect | 0.0000 | 1.0000 | 0.9838 | 0.9918 | 0.0000 | 0.0000 |
| query_sparse_hillclimb_benign | mqtt | 0.0000 | benign_side_effect |  | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |
| query_sparse_hillclimb_benign | wifi | 0.0000 | benign_side_effect | 0.0000 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |
| query_sparse_hillclimb_benign | bluetooth | 0.1000 | benign_side_effect | 0.0000 | 1.0000 | 0.9838 | 0.9918 | 0.0000 | 0.0000 |
| query_sparse_hillclimb_benign | mqtt | 0.1000 | benign_side_effect |  | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |
| query_sparse_hillclimb_benign | wifi | 0.1000 | benign_side_effect | 0.4658 | 0.6818 | 0.9982 | 0.8102 | 0.4658 | -0.1889 |

The robust metrics changed the research question. Malicious evasion stayed comparatively controlled, with targeted success rate 0.0083 at epsilon 0.1. Benign-side drift was the real deployment problem. At epsilon 0.1 the global benign-side query attack pushed FPR to 0.2329, and the protocol table showed that WiFi was the main driver. MQTT also lacked benign test rows entirely, which became an explicit limitation.

That result forced the next phase: targeted WiFi hardening rather than generic retraining.

## 9. WiFi hardening taught the project how fragile the FPR/recall trade-off really was

### Table 22. Comparison of the three WiFi hardening runs
| Run | Intent | Train flip rows | Train selected rows | Base threshold | New threshold | Clean FPR | Attacked benign FPR | Attacked malicious recall | Delta clean F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260309_103936 | Conservative FPR-first hardening | 247 | 247 | 0.2261 | 0.3866 | 0.0028 | 0.0050 |  | -0.0001 |
| 20260309_135449 | Constraint-aware pilot | 1 | 6 | 0.2261 | 0.1224 | 0.0167 | 0.0000 | 1.0000 | 0.0001 |
| 20260309_180250 | Full-budget hardening | 28,132 | 35,632 | 0.2261 | 0.0351 | 0.0768 | 0.0767 | 0.9952 | 0.0011 |

### Table 23. Full-budget hard-negative generation summary
| Split | Epsilon | Targeted rows | Flip rows | Flip rate | Selected rows | Queries mean | Queries p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.0300 | 30,000 | 7,683 | 0.2561 | 10,183 | 172.0320 | 220.0000 |
| train | 0.0500 | 30,000 | 8,535 | 0.2845 | 11,035 | 167.6713 | 220.0000 |
| train | 0.1000 | 30,000 | 11,914 | 0.3971 | 14,414 | 145.4462 | 220.0000 |
| val | 0.0300 | 12,000 | 397 | 0.0331 | 2,897 | 213.3995 | 220.0000 |
| val | 0.0500 | 12,000 | 447 | 0.0372 | 2,947 | 212.8328 | 220.0000 |
| val | 0.1000 | 12,000 | 689 | 0.0574 | 3,189 | 208.9661 | 220.0000 |
| val | 0.1000 | 12,000 |  |  |  | 206.6819 | 220.0000 |

The conservative run looked good. The small-sample constrained run looked even better. But the full-budget run exposed the cost of trying to recover adversarial malicious recall aggressively: the threshold fell to 0.0351, clean FPR rose to 0.0768, and attacked benign FPR rose to 0.0767. That is the moment where the project learned that the WiFi problem could not be solved by a single hardening recipe.

The next step therefore became a family rebalance matrix with explicit gate checking.

## 10. The WiFi rebalance matrix found a temporary answer, then invalidated it through stability

### Table 24. WiFi rebalance decision table
| Family | Family name | Threshold | Clean FPR | Attacked benign FPR | Adv. malicious recall |
| --- | --- | --- | --- | --- | --- |
| A | fpr_priority | 0.3083 | 0.0037 | 0.0035 | 0.9900 |
| C | recall_priority_control | 0.9875 | 0.0000 | 0.0000 | 0.5975 |
| B | balanced | 0.9950 | 0.0000 | 0.0000 | 0.3850 |

### Table 25. WiFi rebalance stability check
| Seed | Family | Family name | Gate pass | Clean FPR | Attacked benign FPR | Adv. malicious recall | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | A | fpr_priority | True | 0.0037 | 0.0035 | 0.9900 | matrix_primary |
| 43 | A | fpr_priority | False | 0.0000 | 0.0000 | 0.4470 | stability_rerun |

Family A finally passed the WiFi gate on the primary seed. Then it failed stability, with adversarial malicious recall collapsing to 0.4470 on the rerun. That single table explains why the project moved away from WiFi-only remediation and into a protocol-wide robust matrix. A thesis that ignored this stability failure would be overstating its defense.

## 11. The first protocol-wide robust matrix run proved the original formulation was infeasible

### Table 26. First protocol-wide robust matrix global table
| Candidate | Model | Family | Failure category | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Mean clean F1 | Mean robust F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__A__123 | xgboost | A | adv_recall_only | 0.0000 | 0.0000 | 0.9688 | 0.9959 | 0.9841 |
| xgboost__B__123 | xgboost | B | adv_recall_only | 0.0000 | 0.0000 | 0.9688 | 0.9929 | 0.9841 |
| xgboost__C__123 | xgboost | C | adv_recall_only | 0.0000 | 0.0000 | 0.9688 | 0.9929 | 0.9841 |

Every candidate in the first protocol-wide matrix failed, and the run only covered WiFi with XGBoost families. That failure was useful because it justified the heavy runtime and search-profile redesign work that immediately followed.

## 12. The first successful all-model coarse protocol matrix proved the idea could work

### Table 27. All-model coarse protocol matrix global ranking
| Candidate | Model | Family | Gate pass | Failure category | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Mean clean F1 | Mean robust F1 | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__B__42 | xgboost | B | True | none | 0.0001 | 0.0000 | 0.9900 | 0.9969 | 0.9971 | 1 |
| lightgbm__B__42 | lightgbm | B | False | adv_recall_only | 0.0000 | 0.0000 | 0.0527 | 0.6715 | 0.6976 | 2 |
| lightgbm__C__42 | lightgbm | C | False | adv_recall_only | 0.0000 | 0.0000 | 0.0500 | 0.6713 | 0.6958 | 3 |
| xgboost__C__42 | xgboost | C | False | adv_recall_only | 0.0001 | 0.0000 | 0.0027 | 0.6664 | 0.6674 | 4 |
| mlp__C__42 | mlp | C | False | adv_recall_only | 0.0001 | 0.0000 | 0.0000 | 0.5157 | 0.5201 | 5 |
| lightgbm__A__42 | lightgbm | A | False | adv_recall_only | 0.0000 | 0.0007 | 0.0460 | 0.6710 | 0.6940 | 6 |
| catboost__A__42 | catboost | A | False | adv_recall_only | 0.0001 | 0.0007 | 0.0013 | 0.6639 | 0.6650 | 7 |
| catboost__B__42 | catboost | B | False | adv_recall_only | 0.0001 | 0.0007 | 0.7927 | 0.9651 | 0.9594 | 8 |
| catboost__C__42 | catboost | C | False | adv_recall_only | 0.0001 | 0.0013 | 0.5913 | 0.9680 | 0.9119 | 9 |
| mlp__A__42 | mlp | A | False | adv_recall_only | 0.0023 | 0.0020 | 0.0000 | 0.5123 | 0.5170 | 10 |
| mlp__B__42 | mlp | B | False | adv_recall_only | 0.0020 | 0.0027 | 0.0000 | 0.5179 | 0.5212 | 11 |
| xgboost__A__42 | xgboost | A | False | both | 0.1935 | 0.1727 | 0.9773 | 0.9642 | 0.9683 | 12 |

### Table 28. Protocol bottleneck summary from the coarse pass
| Protocol | Gate-pass candidates | Best candidate | Clean F1 | Clean FPR | Adv. malicious recall | Robust F1 | Gate pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wifi | 11 | catboost__C | 0.9971 | 0.0000 | 0.9967 | 0.9983 | True |
| mqtt | 9 | lightgbm__B | 0.9969 | 0.0000 | 0.9907 | 0.9953 | True |
| bluetooth | 1 | xgboost__B | 0.9970 | 0.0000 | 0.9900 | 0.9950 | True |

This run mattered because `xgboost__B__42` became the first all-protocol gate-pass candidate. But Bluetooth still acted as the limiting protocol, and there was no stable final answer yet. That is why targeted recovery and multi-seed stability still had to happen.

## 13. Targeted recovery and early stabilization showed coarse winners were still not enough

### Table 29. Targeted V1 recovery run
| Candidate | Model | Family | Gate pass | Failure category | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Mean clean F1 | Mean robust F1 | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost__B__42 | catboost | B | True | none | 0.0015 | 0.0010 | 0.9900 | 0.9986 | 0.9972 | 1 |
| xgboost__E__42 | xgboost | E | False | fpr_only | 0.0118 | 0.0170 | 0.9900 | 0.9977 | 0.9938 | 2 |
| xgboost__C__42 | xgboost | C | False | fpr_only | 0.4740 | 0.4540 | 0.9900 | 0.9289 | 0.9309 | 3 |
| catboost__C__42 | catboost | C | False | fpr_only | 0.5334 | 0.5350 | 0.9900 | 0.9232 | 0.9266 | 4 |
| xgboost__D__42 | xgboost | D | False | fpr_only | 0.5566 | 0.5370 | 0.9900 | 0.9201 | 0.9216 | 5 |
| catboost__E__42 | catboost | E | False | fpr_only | 0.5466 | 0.5470 | 0.9900 | 0.9220 | 0.9256 | 6 |
| catboost__A__42 | catboost | A | False | fpr_only | 0.7691 | 0.7760 | 0.9900 | 0.8991 | 0.9015 | 7 |
| catboost__D__42 | catboost | D | False | fpr_only | 0.7826 | 0.7830 | 0.9910 | 0.8987 | 0.9036 | 8 |
| mlp__B__42 | mlp | B | False | fpr_only | 0.8613 | 0.8700 | 0.9900 | 0.8785 | 0.8580 | 9 |
| mlp__E__42 | mlp | E | False | fpr_only | 0.8715 | 0.8830 | 0.9900 | 0.8846 | 0.8695 | 10 |
| mlp__C__42 | mlp | C | False | fpr_only | 0.9106 | 0.9170 | 0.9900 | 0.8810 | 0.8645 | 11 |
| mlp__A__42 | mlp | A | False | fpr_only | 0.9334 | 0.9340 | 0.9900 | 0.8748 | 0.8572 | 12 |
| mlp__D__42 | mlp | D | False | fpr_only | 0.9442 | 0.9490 | 0.9900 | 0.8631 | 0.8429 | 13 |
| xgboost__B__42 | xgboost | B | False | fpr_only | 0.9741 | 0.9700 | 0.9900 | 0.8827 | 0.8848 | 14 |
| xgboost__A__42 | xgboost | A | False | fpr_only | 0.9785 | 0.9750 | 0.9900 | 0.8826 | 0.8843 | 15 |

### Table 30. Stability summary for the targeted recovery run
| Candidate group | Model | Family | Consistent pass | Seeds checked | Any seed passed |
| --- | --- | --- | --- | --- | --- |
| catboost__B | catboost | B | False | 5 | True |
| xgboost__E | xgboost | E | False | 5 | False |

### Table 31. Longer stabilization sweep
| Candidate | Model | Family | Gate pass | Failure category | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Mean clean F1 | Mean robust F1 | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost__C__42 | catboost | C | True | none | 0.0015 | 0.0000 | 0.9910 | 0.9989 | 0.9980 | 1 |
| catboost__B__42 | catboost | B | True | none | 0.0015 | 0.0000 | 0.9910 | 0.9987 | 0.9977 | 2 |
| catboost__E__42 | catboost | E | True | none | 0.0015 | 0.0000 | 0.9900 | 0.9989 | 0.9975 | 3 |
| catboost__D__42 | catboost | D | True | none | 0.0010 | 0.0010 | 0.9900 | 0.9985 | 0.9972 | 4 |
| catboost__A__42 | catboost | A | False | fpr_only | 0.0059 | 0.0090 | 0.9900 | 0.9982 | 0.9957 | 5 |
| xgboost__E__42 | xgboost | E | False | fpr_only | 0.0118 | 0.0170 | 0.9900 | 0.9989 | 0.9953 | 6 |
| xgboost__B__42 | xgboost | B | False | fpr_only | 0.0163 | 0.0240 | 0.9900 | 0.9987 | 0.9937 | 7 |
| xgboost__C__42 | xgboost | C | False | fpr_only | 0.0163 | 0.0250 | 0.9900 | 0.9989 | 0.9937 | 8 |
| xgboost__A__42 | xgboost | A | False | fpr_only | 0.0163 | 0.0280 | 0.9900 | 0.9989 | 0.9936 | 9 |
| xgboost__D__42 | xgboost | D | False | fpr_only | 0.0197 | 0.0280 | 0.9900 | 0.9990 | 0.9936 | 10 |
| mlp__C__42 | mlp | C | False | fpr_only | 0.0731 | 0.1110 | 0.9900 | 0.9877 | 0.9728 | 11 |
| mlp__A__42 | mlp | A | False | fpr_only | 0.1168 | 0.1180 | 0.9910 | 0.9895 | 0.9649 | 12 |
| mlp__E__42 | mlp | E | False | fpr_only | 0.1557 | 0.1550 | 0.9900 | 0.9926 | 0.9636 | 13 |
| mlp__B__42 | mlp | B | False | fpr_only | 0.1838 | 0.1910 | 0.9900 | 0.9800 | 0.9459 | 14 |
| mlp__D__42 | mlp | D | False | fpr_only | 0.1966 | 0.1940 | 0.9910 | 0.9856 | 0.9504 | 15 |

### Table 32. Stability summary for the longer sweep
| Candidate group | Model | Family | Consistent pass | Seeds checked | Any seed passed |
| --- | --- | --- | --- | --- | --- |
| catboost__B | catboost | B | False | 9 | True |
| catboost__C | catboost | C | False | 9 | True |
| catboost__E | catboost | E | False | 9 | True |

These tables explain why the project did not promote an early CatBoost winner. Coarse candidates existed, but none survived the stability check. This is the exact gap that the later final stabilized run had to close.

## 14. The data audit justified more disciplined stability-aware selection

### Table 33. Train/test protocol balance from the audit
| Split | Protocol | Attack | Benign | Total | Benign ratio | Attack ratio |
| --- | --- | --- | --- | --- | --- | --- |
| train | bluetooth | 99,839 | 23,396 | 123,235 | 0.1898 | 0.8102 |
| train | mqtt | 262,938 | 197,564 | 460,502 | 0.4290 | 0.5710 |
| train | wifi | 6,705,161 | 380,994 | 7,086,155 | 0.0538 | 0.9462 |
| test | bluetooth | 25,170 | 6,671 | 31,841 | 0.2095 | 0.7905 |
| test | mqtt | 63,715 | 0.0000 | 63,715 | 0.0000 | 1.0000 |
| test | wifi | 1,512,860 | 41,737 | 1,554,597 | 0.0268 | 0.9732 |

### Table 34. Protocol share shift from the audit
| Protocol | Train rows | Test rows | Train share | Test share | Test minus train |
| --- | --- | --- | --- | --- | --- |
| wifi | 7,086,155 | 1,554,597 | 0.9239 | 0.9421 | 0.0182 |
| mqtt | 460,502 | 63,715 | 0.0600 | 0.0386 | -0.0214 |
| bluetooth | 123,235 | 31,841 | 0.0161 | 0.0193 | 0.0032 |

### Table 35. Largest audited feature shifts
| Feature | Train mean | Test mean | Train std | Test std | Absolute mean shift | SMD-like |
| --- | --- | --- | --- | --- | --- | --- |
| Variance | 0.1292 | 0.0877 | 0.2869 | 0.2354 | 0.0415 | 0.1445 |
| rst_count | 188.4019 | 54.4232 | 978.6324 | 449.1082 | 133.9787 | 0.1369 |
| Max | 96.6673 | 71.0882 | 201.9684 | 134.6479 | 25.5791 | 0.1266 |
| psh_flag_number | 0.0379 | 0.0217 | 0.1306 | 0.0967 | 0.0162 | 0.1240 |
| Std | 12.6449 | 5.4316 | 61.0755 | 36.8353 | 7.2133 | 0.1181 |
| Radius | 17.7763 | 7.5793 | 86.3672 | 52.0674 | 10.1970 | 0.1181 |
| Magnitue | 10.7909 | 10.3387 | 4.3411 | 3.3806 | 0.4522 | 0.1042 |
| Drate | 0.8401 | 2.0951 | 12.8664 | 20.5036 | 1.2550 | 0.0975 |
| HTTPS | 0.0125 | 0.0041 | 0.0943 | 0.0528 | 0.0084 | 0.0894 |
| Rate | 14,228.01 | 17,627.60 | 38,397.40 | 41,696.57 | 3,399.58 | 0.0885 |

The audit confirmed that WiFi dominance, MQTT test benign absence, and measurable feature drift were real structural pressures. That made stability-aware model selection more important, not less.

## 15. The first near-final answer: stable finalists without saved robust artifacts

### Table 36. Stabilized finalist global decision table
| Candidate | Model | Family | Gate pass | Failure category | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Mean clean F1 | Mean robust F1 | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__E__42 | xgboost | E | True | none | 0.0000 | 0.0004 | 0.9912 | 0.9982 | 0.9984 | 1 |
| xgboost__C__42 | xgboost | C | True | none | 0.0000 | 0.0007 | 0.9915 | 0.9982 | 0.9982 | 2 |
| catboost__C__42 | catboost | C | True | none | 0.0000 | 0.0007 | 0.9912 | 0.9970 | 0.9971 | 3 |
| xgboost__A__42 | xgboost | A | True | none | 0.0000 | 0.0008 | 0.9920 | 0.9983 | 0.9983 | 4 |
| catboost__D__42 | catboost | D | True | none | 0.0000 | 0.0008 | 0.9910 | 0.9971 | 0.9975 | 5 |
| catboost__E__42 | catboost | E | True | none | 0.0000 | 0.0008 | 0.9908 | 0.9967 | 0.9967 | 6 |
| xgboost__B__42 | xgboost | B | True | none | 0.0000 | 0.0013 | 0.9915 | 0.9983 | 0.9981 | 7 |
| catboost__B__42 | catboost | B | True | none | 0.0001 | 0.0013 | 0.9912 | 0.9970 | 0.9969 | 8 |
| catboost__A__42 | catboost | A | True | none | 0.0001 | 0.0021 | 0.9910 | 0.9968 | 0.9964 | 9 |
| xgboost__D__42 | xgboost | D | True | none | 0.0000 | 0.0025 | 0.9912 | 0.9983 | 0.9976 | 10 |

### Table 37. Stability consistency for the finalist run
| Candidate group | Model | Family | Consistent pass | Seeds checked | Any seed passed |
| --- | --- | --- | --- | --- | --- |
| catboost__C | catboost | C | True | 9 | True |
| xgboost__E | xgboost | E | True | 9 | True |
| xgboost__C | xgboost | C | False | 9 | True |

### Table 38. Protocol comparison of the two stable finalists
| Protocol | Candidate | Clean F1 | Clean FPR | Attacked benign FPR | Adv. malicious recall | Robust F1 | Gate pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wifi | xgboost__E | 0.9948 | 0.0000 | 0.0000 | 0.9912 | 0.9956 | True |
| wifi | catboost__C | 0.9949 | 0.0000 | 0.0000 | 0.9912 | 0.9956 | True |
| mqtt | xgboost__E | 0.9999 | 0.0000 | 0.0004 | 1.0000 | 0.9996 | True |
| mqtt | catboost__C | 0.9961 | 0.0000 | 0.0007 | 0.9932 | 0.9959 | True |
| bluetooth | xgboost__E | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | True |
| bluetooth | catboost__C | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | True |

This run gave the first stable finalists: `xgboost__E` and `catboost__C`. But the robust artifacts were not saved. That meant the best analytical answer still was not a deployable answer. The only honest next step was to rerun the final pass with artifact persistence enabled.

## 16. The artifact-persistent final run promoted `catboost__E` as the deployment baseline

### Table 39. Final artifact-persistent global decision table
| Candidate | Model | Family | Gate pass | Failure category | Worst clean FPR | Worst attacked benign FPR | Worst adv. recall | Mean clean F1 | Mean robust F1 | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__A__42 | xgboost | A | True | none | 0.0000 | 0.0000 | 0.9918 | 0.9980 | 0.9985 | 1 |
| xgboost__C__42 | xgboost | C | True | none | 0.0000 | 0.0000 | 0.9915 | 0.9978 | 0.9980 | 2 |
| xgboost__B__42 | xgboost | B | True | none | 0.0000 | 0.0000 | 0.9915 | 0.9972 | 0.9972 | 3 |
| xgboost__D__42 | xgboost | D | True | none | 0.0000 | 0.0000 | 0.9912 | 0.9980 | 0.9983 | 4 |
| xgboost__E__42 | xgboost | E | True | none | 0.0000 | 0.0000 | 0.9912 | 0.9974 | 0.9978 | 5 |
| catboost__E__42 | catboost | E | True | none | 0.0001 | 0.0000 | 0.9923 | 0.9969 | 0.9975 | 6 |
| catboost__D__42 | catboost | D | True | none | 0.0001 | 0.0000 | 0.9918 | 0.9970 | 0.9976 | 7 |
| catboost__B__42 | catboost | B | True | none | 0.0001 | 0.0000 | 0.9912 | 0.9971 | 0.9975 | 8 |
| catboost__C__42 | catboost | C | True | none | 0.0001 | 0.0000 | 0.9903 | 0.9965 | 0.9971 | 9 |
| catboost__A__42 | catboost | A | True | none | 0.0001 | 0.0001 | 0.9925 | 0.9972 | 0.9979 | 10 |

### Table 40. Final stability consistency summary
| Candidate group | Model | Family | Consistent pass | Seeds checked | Any seed passed |
| --- | --- | --- | --- | --- | --- |
| catboost__E | catboost | E | True | 9 | True |
| xgboost__A | xgboost | A | False | 9 | True |
| xgboost__C | xgboost | C | False | 9 | True |

### Table 41. Final protocol-level robust metrics for `catboost__E`
| Protocol | Selected threshold | Clean F1 | Clean FPR | Attacked benign FPR | Adv. malicious recall | Robust F1 | Gate pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wifi | 0.9067 | 0.9954 | 0.0000 | 0.0000 | 0.9923 | 0.9961 | True |
| mqtt | 0.9667 | 0.9953 | 0.0001 | 0.0000 | 0.9929 | 0.9964 | True |
| bluetooth | 0.8800 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | True |

### Table 42. Final full-test routed prediction metrics
| Scope | Model | Threshold | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | TP | TN | FP | FN | Rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wifi | catboost__E | 0.9067 | 1.0000 | 0.9981 | 0.9991 | 0.0003 | 0.9999 | 1.0000 | 1,510,038 | 41,725 | 12 | 2,822 | 1,554,597 |
| mqtt | catboost__E | 0.9667 | 1.0000 | 0.9931 | 0.9966 |  |  | 1.0000 | 63,277 | 0 | 0 | 438 | 63,715 |
| bluetooth | catboost__E | 0.8800 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 1.0000 | 1.0000 | 25,134 | 6,671 | 0 | 36 | 31,841 |
| global_protocol_routed | catboost__E | protocol_specific | 1.0000 | 0.9979 | 0.9990 | 0.0002 | 0.9999 | 1.0000 | 1,598,449 | 48,396 | 12 | 3,296 | 1,650,153 |

This run is the final answer because it combines all the requirements that earlier candidates failed to combine: protocol coverage, gate pass, stability, and saved artifacts. The final stability summary shows that only `catboost__E` remained an all-seed consistent pass. The coarse-ranked XGBoost families no longer mattered more than that stability fact.

The final routed full-test metrics show why that decision is defensible. The global routed `catboost__E` row retained near-perfect precision and recall, with extremely low false-positive behavior. The one mandatory caveat remains MQTT: because the MQTT test split contains no benign negatives, MQTT FPR and ROC-AUC are not estimable on that split and must be stated as unavailable rather than silently implied.

## 17. Core synthesis

The complete journey supports a stronger thesis claim than any single run could support. The project did not simply find a high-performing IoMT IDS. It built a reproducible decision process that started from data reality, treated protocol heterogeneity as first-order information, replaced clean-score optimism with leakguard and split controls, discovered that benign-side adversarial false positives were the real operational danger, and used multi-seed stability to prevent premature promotion of coarse winners.

### Table 43. Milestone comparison from first baseline to final deployment baseline
| Phase | Run | Model | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Reduced-sample baseline winner | baseline_models_stdlib_20260305_234858 | logistic_sgd | 0.9890 | 0.9866 | 0.9878 | 0.0110 |
| First full-data GPU winner | full_gpu_models_20260306_001638 | complex_mlp | 0.9990 | 0.9977 | 0.9984 | 0.0328 |
| First HPO winner | full_gpu_hpo_models_20260306_134806 | complex_mlp_tuned | 0.9982 | 0.9978 | 0.9980 | 0.0601 |
| Initial protocol-routed winner | full_gpu_hpo_models_20260306_153556 | xgboost_tuned__protocol_routed | 0.9999 | 0.9991 | 0.9995 | 0.0027 |
| Leakguard base run | full_gpu_hpo_models_20260306_195851 | weighted_ensemble__protocol_routed | 0.9995 | 0.9985 | 0.9990 | 0.0153 |
| Final deployment baseline | full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105 | catboost__E | 1.0000 | 0.9979 | 0.9990 | 0.0002 |

The most important conclusion is that the final promoted model was not chosen because it had the single flashiest coarse score. It was chosen because it was the only candidate that survived the full chain of evidence: data realism, protocol-aware evaluation, adversarial stress, stability, and artifact persistence.

The remaining sections deepen that claim with appendix-level support. They are included because a thesis defense rarely fails on the headline metric; it fails on whether the intermediate pivots, constraints, and caveats were documented carefully enough.

## 18. Detailed decision log extracted from the project timeline

### Table 44. Decision log from observation to next move
| Date | Observed result | Decision taken | Why that forced the next step |
| --- | --- | --- | --- |
| 2026-03-05 | EDA showed 93% attack share and overwhelming WiFi dominance. | Use protocol slices and FPR-aware thresholding from the start. | Global accuracy would be misleading. |
| 2026-03-05 | Small stdlib baselines already achieved strong F1. | Scale to full-data GPU models instead of polishing shallow baselines. | Learnability was confirmed quickly and cheaply. |
| 2026-03-06 | Full-data global metrics were excellent, but Bluetooth FPR remained uncomfortable. | Move to constrained HPO under lower-FPR objectives. | Operational benign alarms mattered more than another decimal in global F1. |
| 2026-03-06 | First HPO still left the tuned MLP above the desired FPR regime. | Abandon the single-global-model assumption and route by protocol. | Different traffic regimes were behaving as different problems. |
| 2026-03-06 | Protocol routing made difficult slices look almost perfect. | Audit split logic, leakage, and fallback behavior before trusting the result. | A suspiciously easy win is a warning, not a conclusion. |
| 2026-03-06 | First FPR-fix run collapsed to WiFi-only output. | Repair split construction with protocol- and class-aware handling. | Evaluation integrity had become the main blocker. |
| 2026-03-06 | Leakguard produced the first trusted routed baseline. | Use that run as the base for explainability and robustness. | The project needed trustworthy failure analysis, not another clean-only score. |
| 2026-03-08 | Realistic query attacks barely hurt malicious recall but exploded WiFi benign FPR. | Target WiFi hardening directly. | Benign-side robustness became the operational bottleneck. |
| 2026-03-09 | WiFi rebalance found a seed-42 answer that failed stability. | Generalize the search into a protocol-wide robust matrix. | A local WiFi-only fix was not thesis-grade evidence. |
| 2026-03-10 to 2026-03-14 | Coarse winners kept failing stability or artifact persistence. | Keep rerunning until one candidate satisfied gates, stability, and saved-model requirements simultaneously. | Deployment claims needed more than a single-seed leaderboard. |

This table matters because it shows that the project did not move randomly from one experiment to the next. Each pivot responded to a specific failure mode. The chronology begins with a data-imbalance problem, moves into a protocol-heterogeneity problem, then into a robustness problem, and finally into a stability-and-artifact problem. That sequence is exactly why the final thesis claim is stronger than a standard "best test F1" story.

Another way to read Table 44 is as a record of increasing methodological strictness. Early in the timeline the main question was whether the feature space could separate benign from malicious traffic. Midway through the timeline the question became whether the separation still held after leakage controls, protocol-aware routing, and adversarial perturbations. By the end, the question was whether any candidate still looked good when all of those filters were applied simultaneously and repeatedly across seeds.

## 19. Additional EDA details that shaped all later modeling choices

### Table 45. Split-level label balance
| Split | Rows | Benign rows | Attack rows | Benign ratio | Attack ratio |
| --- | --- | --- | --- | --- | --- |
| train | 7,669,892 | 601,954 | 7,067,938 | 0.0785 | 0.9215 |
| test | 1,650,153 | 48,408 | 1,601,745 | 0.0293 | 0.9707 |

### Table 46. Most frequent concrete attack names
| Attack name | Rows |
| --- | --- |
| TCP_IP-DDoS-UDP1 | 411,824 |
| TCP_IP-DDoS-ICMP2 | 390,510 |
| TCP_IP-DDoS-UDP2 | 363,711 |
| TCP_IP-DDoS-ICMP1 | 348,945 |
| MQTT-DDoS-Connect_Flood | 214,952 |
| TCP_IP-DDoS-UDP3 | 206,604 |
| TCP_IP-DDoS-UDP4 | 206,343 |
| TCP_IP-DDoS-UDP5 | 205,507 |
| TCP_IP-DDoS-UDP8 | 204,105 |
| TCP_IP-DDoS-TCP3 | 204,075 |

### Table 47. Attack-family hotspots by protocol
| Attack family | Protocol | Rows |
| --- | --- | --- |
| DDoS | wifi | 5,846,623 |
| DDoS | mqtt | 250,991 |
| DDoS | bluetooth | 0 |
| DoS | wifi | 2,222,205 |
| DoS | mqtt | 68,785 |
| DoS | bluetooth | 0 |
| Benign | wifi | 422,731 |
| Benign | mqtt | 197,564 |
| Benign | bluetooth | 30,067 |
| Recon | wifi | 131,402 |
| Recon | mqtt | 0 |
| Recon | bluetooth | 0 |

Table 45 explains why threshold calibration could not be treated as a one-time generic choice. The benign ratio fell from 0.0785 in train to 0.0293 in test. In other words, the project was validating in one operating mix and deploying into another. That is exactly the kind of shift that inflates apparent performance when thresholds are selected carelessly.

Table 46 shows that "attack" was not a smooth, homogeneous class. It was dominated by specific DDoS attack names, several of them in the same family cluster. The implication for the thesis was straightforward: a model could look globally excellent while partly learning the signatures of a few dominant scenarios. That is why the project kept returning to per-family and per-protocol slices whenever a result looked too clean.

Table 47 adds the key protocol insight. WiFi contained the massive DDoS and DoS concentration, MQTT mixed benign with several flood-oriented attacks, and Bluetooth had a much smaller but structurally different distribution. That distributional asymmetry is the direct reason why protocol-aware routing became more than a convenience. It became the correct problem formulation.

## 20. Cross-stage metric progressions make the bottlenecks visible

### Table 48. Early protocol progression from baseline to first HPO
| Protocol | Baseline best model | Baseline F1 | Baseline FPR | Full GPU best model | Full GPU F1 | Full GPU FPR | First HPO best model | First HPO F1 | First HPO FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wifi | mlp_1hidden | 0.9949 | 0.0069 | xgboost | 0.9994 | 0.0166 | xgboost_tuned | 0.9995 | 0.0055 |
| mqtt | mlp_1hidden | 0.9864 | 0.0000 | xgboost | 0.9989 | 0.0000 | xgboost_tuned | 0.9984 | 0.0000 |
| bluetooth | logistic_sgd | 0.9914 | 0.0022 | complex_mlp | 0.9775 | 0.1733 | complex_mlp_tuned | 0.9523 | 0.3782 |

### Table 49. Bluetooth from bottleneck to controlled slice
| Stage | Representative model | Threshold | F1 | FPR | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Full-data GPU | complex_mlp | 0.0511 | 0.9775 | 0.1733 | Best early clean slice, but still too many benign alarms for deployment. |
| First global HPO | complex_mlp_tuned | 0.0679 | 0.9523 | 0.3782 | Optimization improved little for the real Bluetooth problem. |
| Initial protocol routing | xgboost_tuned__bluetooth | 0.3251 | 1.0000 | 0.0001 | Routing made Bluetooth nearly look solved, which triggered skepticism. |
| Split repair | xgboost_tuned__bluetooth | 0.9983 | 0.9914 | 0.0000 | After repairing split logic, Bluetooth remained strong with defensible validation. |
| Leakguard base | xgboost_tuned__bluetooth | 0.9983 | 0.9914 | 0.0000 | Leakguard preserved the improvement under stricter overlap controls. |
| Final robust matrix | catboost__E | 0.8800 | 1.0000 | 0.0000 | Final robust deployment candidate kept Bluetooth clean and stable. |

### Table 50. WiFi robustness trade-off across the hardening sequence
| Stage | Representative candidate | Clean FPR | Attacked benign FPR | Malicious-side recall | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Realistic baseline | xgboost leakguard base | 0.0001 | 0.0000 | 0.9988 | Clean WiFi behavior looked excellent before benign-side attack pressure. |
| Benign query attack eps=0.1 | query_sparse_hillclimb_benign | 0.0001 | 0.4658 | 0.9974 | Operational problem emerged as benign drift, not malicious evasion. |
| Conservative WiFi hardening | wifi_robust_v1_103936 | 0.0028 | 0.0050 | 0.9951 | First hardening pass improved robustness without wrecking the clean slice. |
| Full-budget WiFi hardening | wifi_robust_v1_180250 | 0.0768 | 0.0767 | 0.9952 | More aggressive recovery made the clean FPR trade-off unacceptable. |
| WiFi rebalance seed-42 winner | family A | 0.0037 | 0.0035 | 0.9900 | This was the temporary answer that failed stability on rerun. |
| Final robust matrix | catboost__E | 0.0000 | 0.0000 | 0.9923 | The final deployment baseline restored a viable WiFi trade-off under stability. |

Table 48 shows that the three protocols were not improving in the same way. MQTT was strong almost immediately, WiFi improved in already-small margins, and Bluetooth remained the place where model family choice visibly changed the operational risk profile. That asymmetry is the strongest compact argument for why a single global threshold or family comparison would have been incomplete.

Table 49 makes the Bluetooth story concrete. The slice began with acceptable but still operationally expensive false-positive behavior under the best full-data clean model, then became nearly perfect under routing, and finally stayed controlled under leakguard and the final robust matrix. The important point is not that Bluetooth eventually reached FPR 0.0 in multiple tables. The important point is that it only became believable after the suspiciously easy version survived split repair and leak controls.

Table 50 captures the WiFi lesson that changed the entire second half of the thesis. Clean WiFi FPR was never the issue. The issue was benign-side attack sensitivity. Once the epsilon 0.1 benign query attack pushed attacked benign FPR to 0.4658, the project had to stop thinking in terms of clean-score optimization and start thinking in terms of controlled operating envelopes. That single transition is why the later campaign was framed around gate passing, not only ranking.

## 21. The robust-matrix campaign became a study in search design, not just model ranking

### Table 51. Evolution of the protocol-wide robust matrix campaign
| Run | Campaign focus | Candidate rows | Gate-pass candidates | Stable groups | Rank-1 candidate | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 20260310_105248 | Feasibility pass | 3 | 0 |  | xgboost__A__123 | Original formulation was too narrow and infeasible. |
| 20260312_121948 | All-model coarse pass | 12 | 1 |  | xgboost__B | First all-protocol gate-pass candidates appeared. |
| 20260312_180757 | Targeted recovery | 15 | 1 | 0 | catboost__B | Recovery families widened the candidate set but not stable promotion. |
| 20260312_200922 | Longer stabilization sweep | 15 | 4 | 0 | catboost__C | Any-pass groups existed, but none were consistent. |
| 20260314_003108 | Near-final stabilized pass | 10 | 10 | 2 | xgboost__E | Two stable finalists existed, but robust artifacts were missing. |
| 20260314_112105 | Artifact-persistent final pass | 10 | 10 | 1 | xgboost__A | Only one fully stable and saved deployment candidate remained. |

### Table 52. Final robust family definitions
| Family | Name | Description | Hard-negative weight | Malicious-adv weight | Extra top-k per epsilon |
| --- | --- | --- | --- | --- | --- |
| A | fpr_priority | strict_flip_only_low_weight | 1.5000 | 1.5000 | 0.0000 |
| B | balanced | strict_flips_limited_near_miss_moderate_weight | 2.0000 | 2.0000 | 500.0000 |
| C | recall_priority_control | bluetooth_recovery_fallback_tuned | 1.9000 | 2.1000 | 500.0000 |
| D | bluetooth_recovery_d | bluetooth_recovery_conservative_plus | 1.7500 | 2.2500 | 250.0000 |
| E | bluetooth_recovery_e | bluetooth_recovery_aggressive_recall | 2.2500 | 2.7500 | 500.0000 |

### Table 53. Final gate definitions
| Gate | Value | Meaning |
| --- | --- | --- |
| clean_fpr_max | 0.0050 | Upper bound for clean benign false-positive rate. |
| attacked_benign_fpr_max | 0.0050 | Upper bound for benign-side FPR after attack. |
| adv_malicious_recall_min | 0.9900 | Minimum acceptable malicious recall under attack. |
| threshold_gate_attacked_benign_margin | 0.8000 | Internal threshold margin used during gate-preserving selection. |
| strict_fpr_feasibility_check | 1.0000 | Rejects claims when FPR cannot be resolved precisely enough. |
| min_val_benign_for_fpr_gate | 200.0000 | Minimum benign validation rows required to enforce FPR gates. |

Table 51 clarifies why the project kept running new matrix passes instead of choosing the earliest rank-1 candidate. The number of gate-pass candidates was not the problem after the search widened. The problem was that gate passing on a single seed was cheap, while stable gate passing across seeds with saved artifacts was rare. That is why the later runs look repetitive in directory structure but not in scientific meaning.

Table 52 makes the family logic explicit. Families A and B were basically precision- and balance-oriented references. Families C, D, and E were not arbitrary extra variants; they were targeted attempts to recover Bluetooth without breaking the global robust constraints. The final answer coming from family E is therefore consistent with the project history. The family itself was created because the bottleneck demanded it.

Table 53 is important thesis evidence because it states the acceptance criteria directly. A candidate had to stay under 0.0050 clean FPR, under 0.0050 attacked benign FPR, and above 0.9900 adversarial malicious recall. Once these constraints are written plainly, the later decision between XGBoost and CatBoost families becomes a constrained-selection question rather than a beauty contest.

## 22. Feasibility and realism constraints are part of the result, not just implementation detail

### Table 54. Final realism profile by protocol
| Protocol | Rows used | Locked features | Mutable features | Tot/num ratio low | Tot/num ratio high | Rate ratio low | Rate ratio high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wifi | 300,000 | 24 | 20 | 0.9713 | 1.1774 | 1.0000 | 1.0000 |
| mqtt | 300,000 | 24 | 21 | 0.9220 | 1.2286 | 1.0000 | 1.0000 |
| bluetooth | 87,871 | 32 | 13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Table 55. Final data-cap and FPR-feasibility summary
| Protocol | Label | Rows | Unique files | Largest file share | Fallback type | Validation rows | FPR resolution | Floor applied | Rows moved to train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bluetooth | benign | 23,396 | 10 | 0.9296 | none | 15,396 | 0.0001 | True | 6,488 |
| bluetooth | attack | 99,839 | 1 | 1.0000 | row_level_single_file | 19,968 | 0.0001 | False | 0 |
| mqtt | benign | 197,564 | 1 | 1.0000 | row_level_single_file | 39,513 | 0.0000 | False | 0 |
| mqtt | attack | 262,938 | 5 | 0.6581 | file_level_rebalance | 44,376 | 0.0000 | False | 0 |
| wifi | benign | 380,994 | 37 | 0.5059 | none | 43,233 | 0.0000 | False | 0 |
| wifi | attack | 6,705,161 | 45 | 0.0309 | none | 1,660,984 | 0.0000 | False | 0 |

### Table 56. Sampling profile of the realistic robustness campaign
| Stage | WiFi rows | MQTT rows | Bluetooth rows | Total rows | Per-protocol target |
| --- | --- | --- | --- | --- | --- |
| Surrogate training | 200,000 | 200,000 | 123,235 | 523,235 | 200,000 |
| Evaluation | 50,000 | 25,000 | 31,671 | 106,671 | attack<=25000, benign<=25000 |
| Realistic evaluation | 10,000 | 5,000 | 10,000 | 25,000 | attack<=5000, benign<=5000 |

Table 54 shows that the three protocols lived under different perturbation geometries. Bluetooth had 32 locked features and only 13 mutable ones, far tighter than WiFi or MQTT. That is a practical explanation for why Bluetooth became both hard to recover and easy to overstate. The attack space itself was narrower.

Table 55 is where data reality becomes impossible to ignore. Bluetooth benign data needed a floor repair, and 6,488 benign rows were moved back to train to satisfy the floor target. That is not a flaw to hide. It is exactly the kind of detail that determines whether a low-FPR claim is statistically meaningful or only numerically pretty.

Table 56 explains the computational compromises behind the robustness study. The project did not attack all available rows in every phase because that would have made iterative search infeasible. Instead it used large but controlled samples, and the final thesis should state that clearly. The key point is that the sampling was systematic and protocol-aware, not opportunistic.

## 23. Final stability and deployment evidence

### Table 57. Stability envelope of the final candidate groups
| Candidate group | All-seed gate pass | Seeds checked | Min clean FPR | Max clean FPR | Min attacked benign FPR | Max attacked benign FPR | Min adv. recall | Max adv. recall | Primary fail reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost__E | True | 8 | 0.0002 | 0.0025 | 0.0002 | 0.0010 | 0.9901 | 0.9912 | none |
| xgboost__A | False | 8 | 0.0074 | 0.0108 | 0.0074 | 0.0123 | 0.9901 | 0.9904 | fpr_only |
| xgboost__C | False | 8 | 0.0138 | 0.0281 | 0.0133 | 0.0276 | 0.9900 | 0.9901 | fpr_only |

### Table 58. Stable candidate shift between the near-final and final passes
| Run | Consistent-pass groups | Stable candidates | Artifacts saved | Decision |
| --- | --- | --- | --- | --- |
| 20260314_003108 | 2 | catboost__C, xgboost__E | No | Useful analytical shortlist, but not deployable. |
| 20260314_112105 | 1 | catboost__E | Yes | Single stable deployment baseline remained. |

### Table 59. Final deployment-readiness facts
| Dimension | Value | Implication |
| --- | --- | --- |
| Escalation recommendation | stable_candidate_ready | The final run judged a stable candidate ready rather than requiring another search cycle. |
| Protocol hard-cap trigger | No | No emergency protocol cap was needed in the final campaign. |
| Stable all-seed candidate | catboost__E | Only one candidate survived the full stability filter. |
| External benign augmentation | False | The final deployment story stays inside the project dataset rather than mixing unmatched external data. |

Table 57 is arguably the most important appendix table in the whole document. It shows the difference between a candidate that can pass once and a candidate that can keep passing. `catboost__E` stayed inside the clean and attacked benign FPR envelope while preserving adversarial malicious recall across all checked seeds. `xgboost__A` and `xgboost__C` failed for the same reason repeatedly: FPR drift. Once the failure mode is this consistent, the final promotion decision stops being subjective.

Table 58 captures the practical meaning of the March 14 rerun. The earlier stable shortlist was analytically interesting but incomplete because the robust artifacts were not saved. The final pass both changed the stable set and closed the artifact gap. That is why the thesis can end with a deployable baseline rather than only a recommendation for future reruns.

Table 59 closes the loop from analysis to operational readiness. The escalation recommendation was `stable_candidate_ready`, no protocol cap was triggered, and the only consistent all-seed candidate was `catboost__E`. In a thesis context, this is the correct level of evidence for claiming that model selection is finished under the current rules.

## 24. Explicit caveats for thesis wording

### Table 60. Caveats that should be stated explicitly in the thesis chapter
| Caveat | Evidence | Implication for thesis wording |
| --- | --- | --- |
| Attack-heavy corpus | attack share=0.9302 | Accuracy would overstate performance; FPR-constrained evaluation is mandatory. |
| WiFi dominates the merged dataset | wifi rows=8,640,752, share=0.9271 | Protocol slicing is needed to stop WiFi from masking minority-protocol failures. |
| MQTT lacks benign negatives in test | test mqtt benign=0 | MQTT FPR and ROC-AUC on the final test split are unavailable, not zero. |
| Bluetooth benign data is highly concentrated | largest benign file share=0.9296, benign floor applied=True | Bluetooth claims must be interpreted with file concentration and floor repairs in mind. |
| No external benign augmentation in final run | external benign enabled=False | Final deployment claims remain in-distribution rather than cross-source calibrated. |

This table should not be treated as a weakness list. It is part of the scientific discipline of the chapter. A thesis becomes stronger, not weaker, when it states exactly where the evidence is clean and where it is structurally limited. In this project the biggest non-negotiable caveat is MQTT test benign absence, because it affects which metrics can be estimated honestly.

The Bluetooth caveat is different. Bluetooth is not missing benign negatives, but its benign evidence is concentrated enough that floor repairs and file concentration must be acknowledged. That does not invalidate the result. It means the result should be presented as carefully controlled within the available Bluetooth evidence, not as a universal statement about arbitrary unseen Bluetooth environments.

The final caveat is scope. Because external benign augmentation stayed disabled in the final run, the final baseline should be described as the best in-project deployment candidate rather than as a domain-generalized IDS. That distinction matters, and the artifacts support making it clearly.

## 25. Engineering interventions that preserved scientific validity

### Table 61. Engineering interventions that changed what experiments were feasible
| Date | Intervention | Why it was needed | Effect on the scientific campaign |
| --- | --- | --- | --- |
| 2026-03-10 | Stability grouping and deterministic seed logic were corrected. | Cross-seed consistency was being misread by earlier campaign code. | Stability results became scientifically interpretable. |
| 2026-03-10 | CPU threading and startup overhead were tuned for the Slurm jobs. | Robust matrix runs were too slow and underutilizing the machine. | Later campaigns could evaluate more candidates within feasible wall-clock. |
| 2026-03-11 | Per-candidate timing and batched query scoring were added. | The attack loop was CPU-bound and opaque during long runs. | Runtime became diagnosable and significantly more tractable. |
| 2026-03-12 | Recovery family pack and Bluetooth-specific controls were implemented. | Bluetooth remained the protocol that determined whether any candidate could pass globally. | Families C, D, and E became plausible finalists instead of speculative ideas. |
| 2026-03-12 | A fast-run crash from a function-signature mismatch was repaired. | Large scheduled runs were failing before producing usable results. | The next matrix submission became execution-ready instead of wasting queue time. |
| 2026-03-14 | Robust candidate artifact saving was enabled by default. | A stable analytical winner without saved models was not deployable. | The final rerun could promote an actual deployment baseline rather than only a report row. |

This final table belongs in the document because the project reached a point where software-engineering quality and research validity became inseparable. A stability bug, a nondeterministic seed path, or a broken artifact-saving default would not merely be "implementation issues." They would change which model appears to win.

The cleanest way to summarize the entire journey is therefore this: the final `catboost__E` result is not only the output of model training. It is the output of a progressively tightened research process. The process learned from imbalance, from protocol heterogeneity, from adversarial failure, from runtime constraints, and from stability failures. That is why the final answer is defensible.
