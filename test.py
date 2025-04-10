import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tqdm import tqdm

# Đọc dữ liệu
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Chọn cột dự đoán (thay bằng cột phù hợp nếu cần)
target_col = 'blood_glucose_level'
data = df[[target_col]].astype('float32')

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Cài đặt tham số
test_size = 30
timestamp = 5
simulation_size = 10
epochs = 100
batch_size = 8

# Tạo tập dữ liệu theo kiểu chuỗi thời gian
def create_dataset(data, timestamp):
    X, y = [], []
    for i in range(len(data) - timestamp):
        X.append(data[i:i + timestamp])
        y.append(data[i + timestamp])
    return np.array(X), np.array(y)

# Tách dữ liệu train/test
train_data = data_scaled[:-test_size]
test_data = data_scaled[-(test_size + timestamp):]

X_train, y_train = create_dataset(train_data, timestamp)
X_test, y_test = create_dataset(test_data, timestamp)

# Hàm đánh giá độ chính xác (MAPE-based)
def calculate_accuracy(real, pred):
    real = np.array(real) + 1e-5
    pred = np.array(pred) + 1e-5
    mape = np.mean(np.abs((real - pred) / real))
    return (1 - mape) * 100

# Hàm làm mượt chuỗi dự đoán
def anchor(signal, weight):
    smoothed = []
    last = signal[0]
    for val in signal:
        last = weight * last + (1 - weight) * val
        smoothed.append(last)
    return smoothed

# Hàm huấn luyện và dự báo
def forecast():
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=(timestamp, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Dự báo trên test set
    predictions = []
    last_seq = test_data[:timestamp].reshape(1, timestamp, 1)
    for _ in range(test_size):
        next_val = model.predict(last_seq, verbose=0)[0][0]
        predictions.append(next_val)
        last_seq = np.append(last_seq[:, 1:, :], [[[next_val]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return anchor(predictions, 0.3)

# Chạy nhiều lần mô phỏng
results = []
for i in range(simulation_size):
    print(f"Simulation {i+1}")
    result = forecast()
    results.append(result)

# Tính độ chính xác trung bình
true_vals = data[target_col].values[-test_size:]
accuracies = [calculate_accuracy(true_vals, r) for r in results]

# Vẽ biểu đồ
plt.figure(figsize=(15, 6))
for i, r in enumerate(results):
    plt.plot(r, label=f'Forecast {i+1}')
plt.plot(true_vals, label='Actual', color='black', linewidth=2)
plt.title(f'LSTM Forecast | Avg Accuracy: {np.mean(accuracies):.2f}%')
plt.legend()
plt.grid(True)
plt.show()
