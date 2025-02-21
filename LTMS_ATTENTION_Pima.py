import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Đặt seed để đảm bảo tái lập kết quả
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# ✅ 1. Định nghĩa lớp Attention tối ưu
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
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
        score = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        attention_weights = tf.keras.backend.softmax(tf.keras.backend.dot(score, self.V), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.keras.backend.sum(context_vector, axis=1)
        return context_vector

# ✅ 2. Load và xử lý dữ liệu
data = pd.read_csv('diabetes.csv')

# Thay thế giá trị 0 bằng NaN và điền bằng giá trị trung bình
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

# Chọn các feature dựa trên Mutual Information (MI)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)
selected_features = mi_scores[mi_scores > 0.01].index
X = X[selected_features]

# ✅ 3. Cân bằng dữ liệu bằng SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# ✅ 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ 5. Reshape dữ liệu cho LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

# ✅ 6. Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ 7. Xây dựng mô hình LSTM tối ưu với Attention
def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)

    # LSTM layers với Dropout thông minh
    x = LSTM(units=96, return_sequences=True, input_shape=input_shape,
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), activity_regularizer=l2(0.01))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)

    x = LSTM(units=96, return_sequences=True, input_shape=input_shape,
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(units=96, return_sequences=True, input_shape=input_shape,
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # Attention layer
    x = Attention()(x)  # 🔥 Thêm Attention sau LSTM có return_sequences=True

    # Fully Connected Layer
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)

    # Output layer
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    return model

# ✅ 8. Huấn luyện mô hình
def train_lstm_model(X_train, y_train, batch_size=64, learning_rate=0.001):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_attention_model(input_shape)

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=200, 
        batch_size=batch_size, 
        validation_split=0.2, 
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    return model, history

# ✅ Gọi huấn luyện mô hình
model, history = train_lstm_model(X_train, y_train)

# ✅ 9. Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# ✅ 10. Dự đoán và đánh giá mô hình
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
