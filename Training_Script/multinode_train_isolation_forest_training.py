# ============================================================
# ISOLATION FOREST  —  TRAINING DATASETS
# ============================================================
#
# Input  : result_training/ (EKF fused, 200-point sampled)
# Output : result_training/isolation_forest_training.pkl
#          result_training/if_training_scaler.pkl
#          result_training/if_confusion_matrix.png
#          result_training/if_score_distribution.png
#          result_training/if_pr_curve.png
#
# Training data : Dataset 2 ONLY (clean walking, normal behaviour)
#   Dataset 1 excluded — it is the anomaly class (running)
#   Dataset 3 excluded — fused_speed = 0 entirely due to
#                        epoch-0 timestamp issue at sparse sampling
#
# Rolling window = 20  (suitable for 199-row dataset)
# Contamination  = 0.05
#
# Ground truth labels for evaluation:
#   Dataset 1 (running)  -> anomaly  (-1)
#   Dataset 2 (walking)  -> normal   ( 1)
#   Dataset 3 (walking)  -> normal   ( 1)
#
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

ROLLING_WINDOW = 50   # ~8% of 599 rows per dataset

FEATURES = [
    'fused_speed',
    'acceleration_magnitude',
    'fused_vx',
    'fused_vy',
    'movement_direction',
    'speed_change',
    'direction_change',
    'rolling_mean_speed',
    'rolling_std_speed',
    'rolling_mean_accel'
]

def engineer_features(df):
    df = df.copy().reset_index(drop=True)
    df['speed_change']       = df['fused_speed'].diff().fillna(0)
    df['direction_change']   = df['movement_direction'].diff().fillna(0)
    df['rolling_mean_speed'] = df['fused_speed'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['rolling_std_speed']  = df['fused_speed'].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0)
    df['rolling_mean_accel'] = df['acceleration_magnitude'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return df

# ============================================================
# LOAD TRAINING DATA  (Dataset 2 only)
# ============================================================

df2_train = engineer_features(
    pd.read_csv('result_training/standardized_dataset_2_EKF_FUSED.csv')
)

print("=" * 50)
print("TRAINING DATA  (Dataset 2 — Walking)")
print("=" * 50)
print(f"Rows           : {len(df2_train)}")
print(f"Fused speed    : mean={df2_train['fused_speed'].mean():.4f}  max={df2_train['fused_speed'].max():.4f}")
print(f"Accel mag      : mean={df2_train['acceleration_magnitude'].mean():.4f}")

X_train = df2_train[FEATURES].dropna()

print(f"Training rows  : {len(X_train)}")

# ============================================================
# SCALE AND FIT
# ============================================================

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

print("\nIsolation Forest training complete.")

# ============================================================
# LOAD EVALUATION DATA  (all 3 datasets)
# ============================================================

df1 = engineer_features(pd.read_csv('result_training/standardized_dataset_1_EKF_FUSED.csv'))
df2 = engineer_features(pd.read_csv('result_training/standardized_dataset_2_EKF_FUSED.csv'))
df3 = engineer_features(pd.read_csv('result_training/standardized_dataset_3_EKF_FUSED.csv'))

df1['true_label'] = -1
df2['true_label'] =  1
df3['true_label'] =  1

df1['scenario'] = 'Dataset 1 (Running)'
df2['scenario'] = 'Dataset 2 (Walking)'
df3['scenario'] = 'Dataset 3 (Walking - EKF zero speed)'

combined = pd.concat([df1, df2, df3], ignore_index=True)

print("\n" + "=" * 50)
print("EVALUATION DATASET SUMMARY")
print("=" * 50)
print(f"Dataset 1 (Running)  : {len(df1)} rows")
print(f"Dataset 2 (Walking)  : {len(df2)} rows")
print(f"Dataset 3 (Walking)  : {len(df3)} rows")
print(f"Total                : {len(combined)} rows")

# ============================================================
# PREDICT
# ============================================================

X_eval      = combined[FEATURES].dropna()
combined    = combined.loc[X_eval.index]
X_eval_sc   = scaler.transform(X_eval)

predictions = model.predict(X_eval_sc)
scores      = model.decision_function(X_eval_sc)

combined['predicted_label'] = predictions
combined['anomaly_score']   = scores

# ============================================================
# OPTIMAL THRESHOLD
# ============================================================

y_true        = combined['true_label']
y_true_bin    = (y_true == -1).astype(int)

precisions, recalls, thr_pr = precision_recall_curve(y_true_bin, -scores)
f1s           = np.where(
    (precisions + recalls) == 0, 0,
    2 * precisions * recalls / (precisions + recalls)
)
best_idx      = f1s.argmax()
best_thr      = -thr_pr[best_idx]

y_pred_opt    = np.where(scores < best_thr, -1, 1)
combined['predicted_optimal'] = y_pred_opt

print(f"\nDefault  threshold : 0.0")
print(f"Optimal  threshold : {best_thr:.6f}")

# ============================================================
# METRICS HELPER
# ============================================================

def print_metrics(y_true, y_pred, label=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    print(f"\n{'=' * 50}")
    print(f"METRICS  {label}")
    print(f"{'=' * 50}")
    print(f"Accuracy  : {acc  * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec  * 100:.2f}%")
    print(f"F1-Score  : {f1   * 100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Anomaly (-1)', 'Normal (1)'],
        labels=[-1, 1],
        zero_division=0
    ))

print_metrics(y_true, predictions,  "[Default Threshold]")
print_metrics(y_true, y_pred_opt,   "[Optimal Threshold]")

# ============================================================
# PER-DATASET BREAKDOWN
# ============================================================

print("=" * 50)
print("PER-DATASET BREAKDOWN  (Optimal Threshold)")
print("=" * 50)

for scenario, group in combined.groupby('scenario'):
    total   = len(group)
    flagged = (group['predicted_optimal'] == -1).sum()
    normal  = (group['predicted_optimal'] ==  1).sum()
    print(f"\n{scenario}")
    print(f"  Total rows   : {total}")
    print(f"  Flagged (-1) : {flagged}  ({flagged/total*100:.1f}%)")
    print(f"  Normal  ( 1) : {normal}  ({normal/total*100:.1f}%)")

# ============================================================
# CONFUSION MATRIX PLOT  (Optimal Threshold)
# ============================================================

cm = confusion_matrix(y_true, y_pred_opt, labels=[-1, 1])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Anomaly', 'Predicted Normal'],
    yticklabels=['True Anomaly (Running)', 'True Normal (Walking)']
)
plt.title('Isolation Forest — Confusion Matrix (Training Dataset, Optimal Threshold)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('result_training/if_confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix saved: result_training/if_confusion_matrix.png")

# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, color='steelblue', linewidth=2)
plt.scatter(
    recalls[best_idx], precisions[best_idx],
    color='red', s=80, zorder=5,
    label=f"Optimal  F1={f1s[best_idx]*100:.1f}%"
)
plt.title('Isolation Forest — Precision-Recall Curve (Training Dataset)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('result_training/if_pr_curve.png', dpi=150)
plt.show()
print("PR curve saved: result_training/if_pr_curve.png")

# ============================================================
# ANOMALY SCORE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
for scenario, group in combined.groupby('scenario'):
    ax.hist(group['anomaly_score'], bins=40, alpha=0.6, label=scenario)
ax.axvline(x=0,        color='gray', linestyle='--', linewidth=1.2, label='Default boundary')
ax.axvline(x=best_thr, color='red',  linestyle='--', linewidth=1.5,
           label=f'Optimal boundary ({best_thr:.4f})')
ax.set_title('Anomaly Score Distribution — Training Dataset')
ax.set_xlabel('Anomaly Score  (lower = more anomalous)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('result_training/if_score_distribution.png', dpi=150)
plt.show()
print("Score distribution saved: result_training/if_score_distribution.png")

# ============================================================
# SAVE MODEL AND SCALER
# ============================================================

joblib.dump(model,  'result_training/isolation_forest_training.pkl')
joblib.dump(scaler, 'result_training/if_training_scaler.pkl')

print("\n" + "=" * 50)
print("MODEL SAVED")
print("=" * 50)
print("Model  : result_training/isolation_forest_training.pkl")
print("Scaler : result_training/if_training_scaler.pkl")

# ============================================================
# COMPLETE
# ============================================================

print("\nTRAINING PIPELINE COMPLETE")
