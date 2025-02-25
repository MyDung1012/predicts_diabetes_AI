import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization, Attention, LeakyReLU, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# --------------- Thiết lập Seed ---------------
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------- 1. Load dữ liệu ---------------
data = pd.read_csv('diabetes_prediction_dataset.csv')

# --------------- 2. Xử lý dữ liệu ---------------
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']

data[numeric_cols] = data[numeric_cols].replace(0, np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean(numeric_only=True))

for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

if data.isnull().sum().sum() > 0:
    data.fillna(data.median(numeric_only=True), inplace=True)

# --------------- 3. Chọn đặc trưng dựa trên Mutual Information ---------------
X = data.drop('diabetes', axis=1)
y = data['diabetes']

if X.isnull().sum().sum() > 0:
    X.fillna(X.median(numeric_only=True), inplace=True)

mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)
selected_features = mi_scores[mi_scores > 0.01].index
X = X[selected_features]

# --------------- 4. Cân bằng dữ liệu sử dụng SMOTE ---------------
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# --------------- 5. Chuẩn hóa dữ liệu ---------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# --------------- 6. Chia tập train và test ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --------------- Chuẩn bị dữ liệu cho LSTM ---------------
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# --------------- 8. Xây dựng mô hình LSTM với Attention sử dụng Functional API ---------------
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM Layers
x = LSTM(96, activation='relu', return_sequences=True)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = LSTM(64, activation='relu', return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Attention Layer
attention = Attention()([x, x])
# Sau Attention, ta cần flatten để đưa vào Dense layers
x_flat = Flatten()(attention)

# Các lớp Dense
x_dense = Dense(16, activation='relu')(x_flat)
output = Dense(1, activation='sigmoid')(x_dense)

# Định nghĩa mô hình
model = Model(inputs=inputs, outputs=output)

# --------------- 9. Biên dịch mô hình ---------------
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --------------- 10. Thiết lập Early Stopping ---------------
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --------------- 11. Huấn luyện mô hình ---------------
history = model.fit(X_train, y_train, 
                    epochs=500, 
                    batch_size=64, 
                    validation_split=0.2, 
                    verbose=1, 
                    callbacks=[early_stopping])

# --------------- 12. Đánh giá mô hình trên tập test ---------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# --------------- 13. Dự đoán và tính toán các chỉ số đánh giá ---------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("Classification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
