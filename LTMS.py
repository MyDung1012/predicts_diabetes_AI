import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# 1. Load dữ liệu
data = pd.read_csv('diabetes.csv')

# 2. Xử lý dữ liệu:
# Thay thế giá trị 0 (đối với các cột nhất định) bằng NaN và sau đó thay thế bằng giá trị trung bình
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

# 3. Loại bỏ các cột có Mutual Information (MI) thấp
X = data.drop('Outcome', axis=1)
y = data['Outcome']
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)
selected_features = mi_scores[mi_scores > 0.01].index  # chọn các feature có MI > 0.01
X = X[selected_features]

# 4. Cân bằng dữ liệu sử dụng SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# 5. Chuẩn hóa dữ liệu
# Có thể chọn sử dụng 1 trong 2 scaler, ở đây ví dụ sử dụng cả StandardScaler và MinMaxScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# 6. Reshape dữ liệu cho LSTM: (samples, timesteps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# 7. Chia tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(1, X.shape[2]), activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 9. Biên dịch mô hình với Adam optimizer và learning rate tùy chỉnh
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 10. Thêm Early Stopping để tránh overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 11. Huấn luyện mô hình
model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# 12. Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 13. Tính toán thêm các chỉ số Precision, Recall, F1-score
# Dự đoán xác suất trên tập test
y_pred_prob = model.predict(X_test)
# Chuyển đổi xác suất thành nhãn với ngưỡng 0.5
y_pred = (y_pred_prob > 0.5).astype("int32")

# In báo cáo phân loại chi tiết
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Tính riêng các chỉ số
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
