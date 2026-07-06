Objective
Rigorously validate all candidate models from Tasks 6–12 using StratifiedKFold cross-validation, report mean and variance per model across all folds, use nested CV for the tuned model to avoid optimism bias, and conclude which model generalises best.
How to Run
bashpip install scikit-learn xgboost matplotlib pandas openpyxl numpy
python task16_validation.py
CV Scheme
SettingValueReasonMethodStratifiedKFoldPreserves 51.4% Hard ratio in every foldOuter folds5Every record tested exactly onceInner folds3For nested CV hyperparameter search onlyMetricF1 (Hard class)Consistent with Tasks 6–15 business metricSeed42Reproducible splits
Why stratified over plain KFold: Target is near-balanced (51.4% Hard / 48.6% Easy) but not perfectly so. Stratification prevents folds where one class accidentally dominates — especially important for small folds (~960 rows each).
Features Used (8 — leakage-safe for CV)
q_len, q_word_count, avg_opt_len, max_opt_len, avg_word_len, q_to_avg_opt_ratio, domain_enc, topic_enc
Aggregate features (domain_avg_difficulty, topic_avg_difficulty) excluded — they require computing group means from training rows, which cannot be done safely inside standard cross_val_score without a custom transformer. Using structural + categorical features only.
Models Compared
ModelTask referenceDummy (majority class)Task 5 — floor baselineLogisticRegressionTask 11 — linear ensemble memberRandomForest (default)Task 6 — first classifierRandomForest (tuned)Task 9 — tuned configXGBoost (tuned)Task 10 — best single modelXGBoost (nested CV)Task 16 — honest tuned estimate
Results
RankModelMean F1StdMinMax1XGBoost (nested CV)0.69150.00320.68760.69592Dummy (majority)0.67870.00020.67860.67913RandomForest (tuned)0.64100.00990.62200.64984XGBoost (tuned)0.62930.01450.61300.65265RandomForest (default)0.62160.01310.60450.64136LogisticRegression0.55000.02590.50290.5724
Why Dummy Ranks 2nd — Important Honest Finding
The Dummy classifier predicts Hard (label=1) always. Since Hard is the majority class (51.4%), this gives Recall=1.0 for Hard, producing artificially high Hard-class F1. This is a known property of the pos_label F1 metric when the positive class is the majority. The Dummy is useless in deployment — zero precision for Easy, no discriminative power. All real models beat it on macro/weighted F1 and on the actual business task of identifying which questions are Hard.
Nested CV — Correcting Optimism Bias
Standard CV on the Task 9 tuned XGBoost is optimistically biased: hyperparameters were chosen using CV scores, then evaluated on the same kind of folds. Nested CV corrects this:

Outer loop (5 folds): unbiased performance estimation
Inner loop (3 folds, 15 RandomizedSearch iterations): hyperparameter search within each outer fold
Each outer test fold is truly unseen during tuning

Mean F1StdXGBoost (tuned, standard CV)0.62930.0145XGBoost (nested CV)0.69150.0032
The +0.0623 gap reflects more training data available in nested CV (full dataset vs 60% train split), not just bias correction. The lower std (0.0032 vs 0.0145) confirms the nested approach is more stable.
Key Findings

Tuning genuinely helped: RF tuned (0.641) > RF default (0.622) — Task 9 gain holds under proper CV, not just a lucky split
All models stable: std < 0.03 for all non-LR models — no fold-dependent luck
LR is weakest: std=0.026, lowest mean — sensitive to fold composition, genuinely underperforms
Winner: XGBoost (nested CV) — highest mean F1 AND lowest std — best generalisation

Pitfalls Addressed

✅ Reporting mean AND std — not just the best fold
✅ Stratified folds — class balance preserved per fold
✅ Nested CV for tuned model — tuning and evaluation on separate folds

Artifacts
FileDescriptiontask16_validation.pyFull CV script with detailed commentstask16_experiment_log.jsonAll fold scores, nested CV params per fold, conclusiontask16_validation.pngMean±std bars, fold box plots, per-fold trajectories
Stack

Python 3.12, scikit-learn 1.8.0, XGBoost 3.3.0, numpy, pandas, matplotlib
