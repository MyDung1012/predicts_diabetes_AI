import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load and prepare the dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

def data_preprocessing(X, y, fit):
    # 🔁 Scale tập dữ liệu
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # Áp dụng vào X_train
    X = log_transformer.transform(X)
    X = pd.DataFrame(X)  # Chuyển lại thành DataFrame nếu cần dùng .fillna()
    X.fillna(0, inplace=True)
    global scaler
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # 🔎 Kiểm tra và xử lý giá trị âm trước khi áp dụng log-transform

    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y

# Prepare inputs for the GAN
X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# PCA for visualization (optional)
from imblearn.over_sampling import SMOTENC

sm = SMOTENC(categorical_features=[X.columns.get_loc('gender')], random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train)
print(X_train.shape[0])

X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test, y_test = data_preprocessing(X_test, y_test, fit=False)


# --------------- 5. Xây dựng mô hình Stacking ---------------
def build_stacking_model(random_state=42):
    base_estimators = [
        ('xgb', XGBClassifier(random_state=random_state)),
        ('ada', AdaBoostClassifier(random_state=random_state))
    ]
    final_estimator = LogisticRegression(random_state=random_state)

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

print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.5f}")
print(f"Recall: {recall_score(y_test, y_pred):.5f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.5f}")
print(f"Accuracy: {stack_model.score(X_test, y_test):.5f}")

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Tính Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.5f}")

# Vẽ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=['Non-Diabetes', 'Diabetes']).plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.title('SMOTENC and MLP model')
plt.ylabel('True Label')
plt.grid(False)
plt.show()

import joblib

# Lưu mô hình
joblib.dump(stack_model, 'SMOTENC_stacking_model.joblib')

