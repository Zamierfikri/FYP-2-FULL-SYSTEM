# ============================================================
# RANDOM FOREST GEOFENCING  —  TRAINING DATASETS
# ============================================================
#
# Input  : result_training/*_EKF_FUSED_GEOFENCE_LABELED.csv
# Output : result_training/random_forest_training.pkl
#          result_training/rf_confusion_matrix.png
#          result_training/rf_feature_importance.png
#
# Train  : Dataset 1 + Dataset 2  (398 rows)
# Test   : Dataset 3              (198 rows)
#
# Same model parameters as multinode_train_random_forest.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# LOAD DATASETS
# ============================================================

df1 = pd.read_csv('result_training/standardized_dataset_1_EKF_FUSED_GEOFENCE_LABELED.csv')
df2 = pd.read_csv('result_training/standardized_dataset_2_EKF_FUSED_GEOFENCE_LABELED.csv')
df3 = pd.read_csv('result_training/standardized_dataset_3_EKF_FUSED_GEOFENCE_LABELED.csv')

print("=" * 45)
print("DATASET INFORMATION")
print("=" * 45)
print(f"Dataset 1 shape : {df1.shape}")
print(f"Dataset 2 shape : {df2.shape}")
print(f"Dataset 3 shape : {df3.shape}")

# ============================================================
# FEATURES AND TARGET  (same as multinode_train_random_forest.py)
# ============================================================

FEATURES = [
    'fused_x',
    'fused_y',
    'fused_vx',
    'fused_vy',
    'fused_speed',
    'acceleration_magnitude',
    'movement_direction'
]

TARGET = 'geofence_label'

# ============================================================
# TRAIN / TEST SPLIT BY DATASET HOLDOUT
# Train: Dataset 1 + 2  |  Test: Dataset 3
# ============================================================

train_df = pd.concat([df1, df2], ignore_index=True).dropna(subset=FEATURES + [TARGET])
test_df  = df3.dropna(subset=FEATURES + [TARGET])

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

print("\n" + "=" * 45)
print("TRAIN / TEST SPLIT")
print("=" * 45)
print(f"Training  (DS1 + DS2) : {X_train.shape[0]} rows")
print(f"Testing   (DS3)       : {X_test.shape[0]} rows")

print("\nTraining label distribution:")
print(y_train.value_counts().to_string())

print("\nTesting label distribution:")
print(y_test.value_counts().to_string())

# ============================================================
# RANDOM FOREST MODEL  (same parameters as main script)
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("\n" + "=" * 45)
print("TRAINING MODEL")
print("=" * 45)

model.fit(X_train, y_train)

print("Training complete.")

# ============================================================
# PREDICTION
# ============================================================

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================
# METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "=" * 45)
print("EVALUATION RESULTS")
print("=" * 45)
print(f"Accuracy  : {accuracy  * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall    * 100:.2f}%")
print(f"F1-Score  : {f1        * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# CONFUSION MATRIX
# ============================================================

classes = sorted(model.classes_)
cm      = confusion_matrix(y_test, y_pred, labels=classes)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)
plt.title('Random Forest — Confusion Matrix (Training Dataset)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('result_training/rf_confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix saved: result_training/rf_confusion_matrix.png")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\n" + "=" * 45)
print("FEATURE IMPORTANCE")
print("=" * 45)
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 5))
colors = ['steelblue' if i == 0 else 'lightsteelblue' for i in range(len(importance_df))]
plt.bar(importance_df['Feature'], importance_df['Importance'], color=colors)
plt.xticks(rotation=30, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest — Feature Importance (Training Dataset)')
plt.tight_layout()
plt.savefig('result_training/rf_feature_importance.png', dpi=150)
plt.show()
print("Feature importance saved: result_training/rf_feature_importance.png")

# ============================================================
# PER-CLASS METRICS TABLE
# ============================================================

print("\n" + "=" * 45)
print("PER-CLASS METRICS")
print("=" * 45)

for cls in classes:
    mask  = y_test == cls
    total = mask.sum()
    correct = ((y_test == cls) & (y_pred == cls)).sum()
    p = precision_score(y_test, y_pred, labels=[cls], average='macro', zero_division=0)
    r = recall_score(y_test, y_pred, labels=[cls], average='macro', zero_division=0)
    f = f1_score(y_test, y_pred, labels=[cls], average='macro', zero_division=0)
    print(f"\n  {cls}")
    print(f"    Support   : {total}")
    print(f"    Correct   : {correct}")
    print(f"    Precision : {p*100:.2f}%")
    print(f"    Recall    : {r*100:.2f}%")
    print(f"    F1-Score  : {f*100:.2f}%")

# ============================================================
# SAVE MODEL AND FEATURES
# ============================================================

joblib.dump(model,    'result_training/random_forest_training.pkl')
joblib.dump(FEATURES, 'result_training/rf_training_features.pkl')

print("\n" + "=" * 45)
print("MODEL SAVED")
print("=" * 45)
print("Model   : result_training/random_forest_training.pkl")
print("Features: result_training/rf_training_features.pkl")

# ============================================================
# REAL-TIME TEST EXAMPLE
# ============================================================

print("\n" + "=" * 45)
print("REAL-TIME TEST EXAMPLE")
print("=" * 45)

sample_row = X_test.iloc[[0]]
pred       = model.predict(sample_row)[0]
proba      = model.predict_proba(sample_row)[0]

print(f"Input features:\n{sample_row.to_string()}")
print(f"\nPrediction  : {pred}")
print(f"Probability : {dict(zip(model.classes_, np.round(proba, 4)))}")

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 45)
print("TRAINING PIPELINE COMPLETE")
print("=" * 45)
