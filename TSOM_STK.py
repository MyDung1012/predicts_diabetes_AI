# %% Import thư viện cần thiết
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (classification_report, precision_score, recall_score,
                             f1_score, roc_auc_score)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
# %% Import thêm thư viện cần thiết cho phương pháp Targeted SMOTE Oversampling and Matching
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
# %% Thiết lập seed để tái lập kết quả
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# %% Load dữ liệu
data = pd.read_csv('diabetes_prediction_dataset.csv')

# %% Chia dữ liệu 80% train, 20% test
X = data.drop(columns='diabetes')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% Tiền xử lý dữ liệu

# %% Tiền xử lý dữ liệu
def data_preprocessing(X, y=None, fit=False):


    X = X.copy()

    # Chuyển đổi categorical thành numeric
    X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})
    X = pd.get_dummies(X, columns=['smoking_history'], drop_first=True)

    # Loại bỏ NaN sau khi xử lý
    X.fillna(0, inplace=True)
    # 🔁 Scale tập dữ liệu

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
    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return (X, y) if y is not None else X


def smote_oversample(X_minority, N=1, k=5):
    neigh = NearestNeighbors(n_neighbors=k).fit(X_minority)
    synthetic_samples = []
    for sample in X_minority:
        neighbors = neigh.kneighbors([sample], return_distance=False)[0]
        for _ in range(N):
            neighbor = X_minority[np.random.choice(neighbors)]
            diff = neighbor - sample
            synthetic_sample = sample + np.random.rand() * diff
            synthetic_samples.append(synthetic_sample)
    return np.array(synthetic_samples)

def psm_matching(X, treatment, caliper=0.2):
    log_reg = LogisticRegression().fit(X, treatment)
    pscores = log_reg.predict_proba(X)[:, 1]

    treatment_indices = np.where(treatment == 1)[0]
    control_indices = np.where(treatment == 0)[0]

    matched_indices = []
    for idx in treatment_indices:
        distances = np.abs(pscores[idx] - pscores[control_indices])
        min_dist = np.min(distances)
        if min_dist <= caliper:
            control_idx = control_indices[np.argmin(distances)]
            matched_indices.append((idx, control_idx))
            control_indices = control_indices[control_indices != control_idx]

    return matched_indices

def targeted_smote_psm(X, y, treatment, N=1, k=5, caliper=0.5):
    X_treated_minority = X[(treatment == 1) & (y == 1)]

    X_syn = smote_oversample(X_treated_minority, N=N, k=k)

    X_augmented = np.vstack([X, X_syn])
    y_augmented = np.concatenate([y, np.ones(X_syn.shape[0])])
    treatment_augmented = np.concatenate([treatment, np.ones(X_syn.shape[0])])

    matches = psm_matching(X_augmented, treatment_augmented, caliper)

    matched_X, matched_y, matched_treatment = [], [], []

    for treat_idx, control_idx in matches:
        matched_X.append(X_augmented[treat_idx])
        matched_X.append(X_augmented[control_idx])

        matched_y.append(y_augmented[treat_idx])
        matched_y.append(y_augmented[control_idx])

        matched_treatment.extend([1, 0])

    return np.array(matched_X), np.array(matched_y), np.array(matched_treatment)

# Thêm biến giả định về treatment (giả sử: diabetes = 1 là nhóm treatment)
treatment = y.copy()  # Hoặc bạn có thể thay thế bằng treatment thật sự của mình

k = 5
N = 10
gamma = 0.5
# Chạy thuật toán Targeted SMOTE + PSM
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
plt.ylabel('True Label')
plt.grid(False)
plt.show()
