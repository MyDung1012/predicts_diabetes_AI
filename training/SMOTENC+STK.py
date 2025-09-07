import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import joblib

# =========================
# Reproducibility
# =========================
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# =========================
# Load & prepare the dataset
# =========================
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

# =========================
# Preprocessing helper
# =========================
scaler = None
pca = None

def data_preprocessing(X, y, fit):
    # Log1p
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    X = log_transformer.transform(X)
    X = pd.DataFrame(X)
    X.fillna(0, inplace=True)

    # Scale
    global scaler
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # PCA 99%
    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y

# =========================
# Train/Test split
# =========================
X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================
# SMOTENC (oversampling, giữ 'gender' là categorical)
# =========================
from imblearn.over_sampling import SMOTENC
sm = SMOTENC(categorical_features=[X.columns.get_loc('gender')], random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("Train size after SMOTENC:", X_train.shape[0])

# =========================
# Preprocess
# =========================
X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test,  y_test  = data_preprocessing(X_test,  y_test,  fit=False)

# =========================
# Build Stacking model
# =========================
def build_stacking_model(random_state=42):
    base_estimators = [
        ('xgb', XGBClassifier(random_state=random_state)),
        ('ada', AdaBoostClassifier(random_state=random_state))
    ]
    final_estimator = LogisticRegression(random_state=random_state, max_iter=1000)

    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        passthrough=True
    )
    return stacking_model

stack_model = build_stacking_model(random_state=42)
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

# =========================
# Metrics
# =========================
print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.5f}")
print(f"Recall: {recall_score(y_test, y_pred):.5f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.5f}")
print(f"Accuracy: {stack_model.score(X_test, y_test):.5f}")

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.5f}")

# =========================
# Confusion Matrix — print & save (counts + normalized)
# =========================
os.makedirs("reports", exist_ok=True)

# Counts
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Counts):")
print(cm)

# Lưu CSV (counts)
cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cm_df.to_csv("reports/confusion_matrix_counts_SMOTENC_stacking.csv", index=True)

# Vẽ & lưu PNG (counts)
fig_c, ax_c = plt.subplots(figsize=(6, 5))
disp_c = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetes', 'Diabetes'])
disp_c.plot(ax=ax_c, values_format='d', colorbar=False)
ax_c.set_title("SConfusion Matrix (Counts)")
plt.tight_layout()
fig_c.savefig("reports/confusion_matrix_counts_SMOTENC_stacking.png", dpi=300)
plt.close(fig_c)

# Normalized (row-wise)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
print("\nConfusion Matrix (Row-normalized):")
print(np.round(cm_norm, 3))

# Lưu CSV (normalized)
cmn_df = pd.DataFrame(cm_norm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cmn_df.to_csv("reports/confusion_matrix_normalized_SMOTENC_stacking.csv", index=True)

# Vẽ & lưu PNG (normalized)
fig_n, ax_n = plt.subplots(figsize=(6, 5))
disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Non-Diabetes', 'Diabetes'])
disp_n.plot(ax=ax_n, values_format='.2f', colorbar=False)
ax_n.set_title("SMOTENC + Stacking — Confusion Matrix (Row-normalized)")
plt.tight_layout()
fig_n.savefig("reports/confusion_matrix_normalized_SMOTENC_stacking.png", dpi=300)
plt.close(fig_n)

print("\n✅ Confusion matrices saved to 'reports/':")
print(" - reports/confusion_matrix_counts_SMOTENC_stacking.csv")
print(" - reports/confusion_matrix_counts_SMOTENC_stacking.png")
print(" - reports/confusion_matrix_normalized_SMOTENC_stacking.csv")
print(" - reports/confusion_matrix_normalized_SMOTENC_stacking.png")

# =========================
# Save model
# =========================
joblib.dump(stack_model, 'SMOTENC_stacking_model.joblib')
print("\n✅ Saved model: SMOTENC_stacking_model.joblib")
