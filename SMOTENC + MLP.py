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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# PCA for visualization (optional)
from imblearn.over_sampling import SMOTENC

sm = SMOTENC(categorical_features=[X.columns.get_loc('gender')], random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train)
print(X_train.shape[0])

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
               epochs=200,
               batch_size=32,
               validation_split=0.2,
               verbose=1,
               callbacks=[early_stopping, reduce_lr],
               )

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


import matplotlib.pyplot as plt

# Trích xuất dữ liệu từ lịch sử huấn luyện
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Training Loss', color='royalblue')
plt.plot(epochs, val_loss, label='Validation Loss', color='darkorange')
plt.title('SMOTENC and MLP model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

