import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# --------------- Thiết lập Seed ---------------
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------- 1. Load dữ liệu ---------------
data = pd.read_csv('diabetes.csv')

# --------------- 2. Xử lý dữ liệu ---------------
# Thay thế giá trị 0 bằng NaN cho các cột có khả năng bị thiếu
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
# Điền giá trị thiếu bằng trung bình của mỗi cột
data.fillna(data.mean(), inplace=True)

# --------------- 3. Chọn đặc trưng dựa trên Mutual Information ---------------
X = data.drop('Outcome', axis=1)
y = data['Outcome']
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)
# Chọn các đặc trưng có MI > 0.01
selected_features = mi_scores[mi_scores > 0.01].index
X = X[selected_features]

# --------------- 4. Cân bằng dữ liệu sử dụng SMOTE ---------------
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# --------------- 5. Chuẩn hóa dữ liệu ---------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------- 6. Chia tập train và test ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------- 7. Giảm chiều dữ liệu bằng PCA ---------------
# Sử dụng PCA để giữ lại 95% phương sai (bạn có thể thay đổi số lượng thành phần nếu cần)
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("Số lượng thành phần sau PCA:", X_train.shape[1])

# --------------- 8. Xây dựng mô hình MLP ---------------
model = Sequential()
# Lớp Dense đầu tiên với 64 neuron, hàm kích hoạt relu và input_shape bằng số thành phần PCA
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
# Lớp Dense thứ hai với 32 neuron
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
# Lớp Dense thứ ba với 16 neuron
model.add(Dense(16, activation='relu'))
# Lớp đầu ra với 1 neuron và hàm kích hoạt sigmoid (cho bài toán phân loại nhị phân)
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