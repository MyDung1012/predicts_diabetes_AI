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
X_train, y_train, treatment_resampled = targeted_smote_psm(X_train, y_train, y_train, N, k, gamma)

y_train = pd.Series(y_train)

counts = y_train.value_counts().sort_index()  # Sắp xếp theo thứ tự (0, 1)
print(counts)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # (samples, timesteps=1, features)
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))      # (samples, timesteps=1, features)

# Xây dựng mô hình LSTM
lstm_model = Sequential([
    LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
    Dropout(0.4),
    LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
optimizer = Adam(learning_rate=0.0001)
lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Thiết lập Early Stopping và ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)


# Huấn luyện mô hình LSTM với class weights và callbacks
history = lstm_model.fit(X_train, y_train,
               epochs=200,
               batch_size=32,
               validation_split=0.2,
               verbose=1,
               callbacks=[early_stopping, reduce_lr],
               )

# Đánh giá mô hình trên tập test
loss, accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Dự đoán và tính toán các chỉ số đánh giá
y_pred_prob = lstm_model.predict(X_test)


y_pred_adjusted = (y_pred_prob > 0.74).astype("int32")
print("Classification Report:")
print(classification_report(y_test, y_pred_adjusted))

precision = precision_score(y_test, y_pred_adjusted)
recall = recall_score(y_test, y_pred_adjusted)
f1 = f1_score(y_test, y_pred_adjusted)
auc_score = roc_auc_score(y_test, y_pred_prob)

print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1 Score: {f1:.5f}")
print(f"ROC-AUC Score: {auc_score:.5f}")
print(f"Accuracy: {accuracy:.5f}")

import matplotlib.pyplot as plt

# Trích xuất dữ liệu từ lịch sử huấn luyện
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Training Loss', color='royalblue')
plt.plot(epochs, val_loss, label='Validation Loss', color='darkorange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title('TSOM and LSTM model')

plt.show()

