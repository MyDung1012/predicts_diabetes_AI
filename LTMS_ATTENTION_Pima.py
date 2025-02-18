import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Đặt seed cho tính tái lập
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# -------------------------------
# Định nghĩa lớp Attention tùy chỉnh
# -------------------------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        self.V = self.add_weight(name="att_var",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, inputs):
        # Tính toán score với tanh
        score = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        # Tính toán attention weights: shape (batch_size, timesteps, 1)
        attention_weights = tf.keras.backend.softmax(tf.keras.backend.dot(score, self.V), axis=1)
        # Nhân attention weights với inputs để có context vector
        context_vector = attention_weights * inputs
        # Cộng gộp theo chiều thời gian: (batch_size, features)
        context_vector = tf.keras.backend.sum(context_vector, axis=1)
        return context_vector

# -------------------------------
# Pipeline: Data Loading & Preprocessing
# -------------------------------

# 1. Load dữ liệu
data = pd.read_csv('diabetes.csv')

# 2. Xử lý dữ liệu: Thay thế giá trị 0 bằng NaN đối với một số cột, sau đó thay thế NaN bằng giá trị trung bình
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

# 4. Cân bằng dữ liệu sử dụng SMOTE (áp dụng trên dữ liệu 2D)
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

# -------------------------------
# Xây dựng mô hình LSTM với Attention
# -------------------------------
model = Sequential()

# Lớp LSTM 1: 128 units, trả về chuỗi (return_sequences=True)
model.add(LSTM(128, 
               return_sequences=True, 
               input_shape=(1, X_train.shape[2]), 
               activation='tanh',
               kernel_regularizer='l2',
               recurrent_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Lớp LSTM 2: 64 units, trả về chuỗi
model.add(LSTM(64, 
               return_sequences=True, 
               activation='tanh',
               kernel_regularizer='l2',
               recurrent_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Lớp LSTM 3: 32 units, trả về chuỗi để cung cấp dữ liệu cho Attention
model.add(LSTM(32, 
               return_sequences=True, 
               activation='tanh',
               kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Thêm lớp Attention: chuyển đổi output từ (batch_size, timesteps, features) -> (batch_size, features)
model.add(Attention())

# Lớp Dense đầu ra
model.add(Dense(1, activation='sigmoid'))

# -------------------------------
# Biên dịch mô hình
# -------------------------------
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Thiết lập callbacks
# -------------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# -------------------------------
# Huấn luyện mô hình
# -------------------------------
history = model.fit(
    X_train, 
    y_train, 
    epochs=300, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# -------------------------------
# Đánh giá mô hình trên tập test
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Dự đoán và tính toán các chỉ số
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
