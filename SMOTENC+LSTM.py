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
# Huấn luyện mô hình LSTM với class weights và callbacks
history = lstm_model.fit(X_train, y_train,
               epochs=100,
               batch_size=32,
               validation_data=(X_test, y_test),
               verbose=1,
               )

# Lưu lại lịch sử huấn luyện vào DataFrame
history_df = pd.DataFrame(history.history)
history_df.index += 1  # để epoch bắt đầu từ 1 thay vì 0
history_df.reset_index(inplace=True)
history_df.rename(columns={"index": "Epoch"}, inplace=True)

# Xuất ra file Excel
history_df.to_excel("SMOTENC_LSTM_history.xlsx", index=False)
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

import joblib
import pickle

# Lưu mô hình Keras sang file .h5
lstm_model.save('SMOTENC+LSTM_model.h5')

# Lưu scaler dùng trong preprocessing
joblib.dump(scaler, 'scaler.pkl')

# Lưu lịch sử huấn luyện (nếu muốn dùng lại)
with open('SMOTENC+LSTM.pkl', 'wb') as f:
    pickle.dump(history.history, f)
