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
    if (X < 0).any():
        print("⚠️ Warning: Negative values detected in X_train. Log transformation might fail.")
    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)


    return X, y


X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# PCA for visualization (optional)

print(X.shape)
# Prepare GAN input (only class 1)
data_diabetes = X_train[y_train == 1]

# Build GAN components
def build_generator(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(input_dim, activation='linear'))
    return model

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))
    return gan

# GAN setup
input_dim = data_diabetes.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)

def generate_synthetic_data(generator, n_samples):
    noise = np.random.normal(0, 1, size=(n_samples, generator.input_shape[1])).astype('float32')
    synthetic_data = generator.predict(noise)
    return synthetic_data

# GAN Training
num_epochs = 100
batch_size = 64
half_batch = batch_size // 2

for epoch in range(num_epochs):
# Lấy mẫu dữ liệu thật (chỉ lớp 1) và ép kiểu dữ liệu rõ ràng
    x_real = data_diabetes.sample(half_batch).values.astype('float32')
    y_real = np.ones((half_batch, 1), dtype='float32')

    # Sinh dữ liệu giả và đảm bảo đúng kiểu float32
    x_fake = generate_synthetic_data(generator, half_batch).astype('float32')
    y_fake = np.zeros((half_batch, 1), dtype='float32')


    discriminator.trainable = True
    discriminator.train_on_batch(x_real, y_real)
    discriminator.train_on_batch(x_fake, y_fake)

    noise = np.random.normal(0, 1, size=(batch_size, input_dim)).astype('float32')
    gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")

# Generate synthetic data (class 1)
synthetic_data = generate_synthetic_data(generator, 60000)
df_synthetic = pd.DataFrame(synthetic_data, columns=X.columns)
df_synthetic['diabetes'] = 1

X_train_smote = df_synthetic.drop('diabetes', axis=1)
y_train_smote = df_synthetic['diabetes']
# Combine synthetic and real data
X_train = pd.concat([X_train, X_train_smote])
y_train = pd.concat([y_train, y_train_smote])

X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test, y_test = data_preprocessing(X_test, y_test, fit=False)


# Xây dựng mô hình LSTM
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# --------------- 9. Biên dịch mô hình ---------------
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# Thiết lập Early Stopping và ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)


# Huấn luyện mô hình LSTM với class weights và callbacks
history = model.fit(X_train, y_train,
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
history_df.to_excel("GAN_MLP_history.xlsx", index=False)
# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Dự đoán và tính toán các chỉ số đánh giá
y_pred_prob = model.predict(X_test)


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

# Lưu mô hình K eras sang file .h5
model.save('MLP_gan_model.h5')

# Lưu scaler dùng trong preprocessing
joblib.dump(scaler, 'scaler.pkl')

# Lưu lịch sử huấn luyện (nếu muốn dùng lại)
with open('GAN+MLP.pkl', 'wb') as f:
    pickle.dump(history.history, f)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# === In & Lưu Confusion Matrix ===
os.makedirs("reports", exist_ok=True)

# 1. Tính confusion matrix (counts)
cm = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix (Counts):")
print(cm)

# Lưu ra CSV
cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cm_df.to_csv("reports/confusion_matrix_GAN_MLP.csv", index=True)

# Vẽ và lưu hình ảnh confusion matrix (counts)
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(ax=ax, values_format='d', colorbar=False)
ax.set_title("Confusion Matrix (Counts)")
plt.tight_layout()
fig.savefig("reports/confusion_matrix_GAN_MLP.png", dpi=300)
plt.close(fig)

# 2. Confusion matrix đã chuẩn hóa (theo hàng)
cm_norm = confusion_matrix(y_test, y_pred_adjusted, normalize='true')
print("\nConfusion Matrix (Row-normalized):")
print(np.round(cm_norm, 3))

# Lưu ra CSV
cm_norm_df = pd.DataFrame(cm_norm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cm_norm_df.to_csv("reports/confusion_matrix_normalized_GAN_MLP.csv", index=True)

# Vẽ và lưu hình ảnh normalized
fig2, ax2 = plt.subplots(figsize=(5, 5))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=[0, 1])
disp2.plot(ax=ax2, values_format=".2f", colorbar=False)
ax2.set_title("Confusion Matrix (Row-normalized)")
plt.tight_layout()
fig2.savefig("reports/confusion_matrix_normalized_GAN_MLP.png", dpi=300)
plt.close(fig2)

print("\n✅ Confusion matrices đã được lưu tại thư mục 'reports/'")
