import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
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
# Xác định các cột số và cột phân loại
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']

# Với cột số: thay thế giá trị 0 bằng NaN và điền giá trị thiếu bằng trung bình
data[numeric_cols] = data[numeric_cols].replace(0, np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean(numeric_only=True))

# Với cột phân loại: điền giá trị thiếu bằng mode
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Chuyển 'gender' thành số: Male = 1, Female = 0
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

# One-hot encoding cho 'smoking_history'
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

# Kiểm tra lại xem có còn NaN không
if data.isnull().sum().sum() > 0:
    print("⚠️ Vẫn còn giá trị NaN, điền vào bằng median")
    data.fillna(data.median(numeric_only=True), inplace=True)

# --------------- 3. Chọn đặc trưng dựa trên Mutual Information ---------------
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Kiểm tra lại NaN trước khi tính Mutual Information
if X.isnull().sum().sum() > 0:
    X.fillna(X.median(numeric_only=True), inplace=True)

# Tính điểm Mutual Information
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)

# Chọn các đặc trưng có MI > 0.01
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

# --------------- 7. Giảm chiều dữ liệu bằng PCA ---------------
pca = PCA(n_components=0.99)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("Số lượng thành phần sau PCA:", X_train.shape[1])

# --------------- Chuẩn bị dữ liệu cho LSTM ---------------
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# --------------- 8. Xây dựng mô hình LSTM ---------------
model = Sequential()
# Lớp LSTM với 64 đơn vị, trả về chuỗi (với return_sequences=True để có thể xếp chồng thêm LSTM)
model.add(LSTM(96, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
# LSTM thứ hai, không cần trả về chuỗi nữa
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.3))

# Các lớp Dense để kết hợp các đặc trưng đã trích xuất từ LSTM
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# --------------- 9. Biên dịch mô hình ---------------
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --------------- 10. Thiết lập Early Stopping để tránh overfitting ---------------
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
print(f"accuracy: {accuracy:.2f}")