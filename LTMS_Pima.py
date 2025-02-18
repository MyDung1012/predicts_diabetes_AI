import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import random
import tensorflow as tf
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)
# 1. Load dữ liệu
data = pd.read_csv('diabetes.csv')

# 2. Xử lý dữ liệu: Thay thế giá trị 0 bằng NaN đối với một số cột, sau đó thay bằng giá trị trung bình
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

# 3. Chọn các feature dựa trên Mutual Information (MI)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)
selected_features = mi_scores[mi_scores > 0.01].index  # Giữ lại các feature có MI > 0.01
X = X[selected_features]

# 4. Cân bằng dữ liệu sử dụng SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# 5. Chuẩn hóa dữ liệu với StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 6. Reshape dữ liệu cho LSTM: (samples, timesteps, features)
# Với dữ liệu tabular, ta xem mỗi mẫu là một chuỗi với 1 timestep.
X = X.reshape(X.shape[0], 1, X.shape[1])

# 7. Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Xây dựng mô hình LSTM cải tiến
model = Sequential()

# Lớp LSTM 1: tăng số units, thêm kernel regularization và BatchNormalization
model.add(LSTM(128, 
               return_sequences=True, 
               input_shape=(1, X.shape[2]), 
               activation='tanh',
               kernel_regularizer='l2',
               recurrent_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Lớp LSTM 2
model.add(LSTM(64, 
               return_sequences=True, 
               activation='tanh',
               kernel_regularizer='l2',
               recurrent_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Lớp LSTM 3 (không trả về chuỗi)
model.add(LSTM(32, 
               activation='tanh',
               kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Lớp Dense đầu ra
model.add(Dense(1, activation='sigmoid'))

# 9. Biên dịch mô hình với Adam optimizer và learning rate tùy chỉnh
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 10. Thiết lập các callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# 11. Huấn luyện mô hình
history = model.fit(
    X_train, 
    y_train, 
    epochs=300, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# 12. Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 13. Tính toán và in ra các chỉ số Precision, Recall, F1-score
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
print(f"accuracy: {accuracy:.2f}")
