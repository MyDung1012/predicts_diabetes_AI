import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from imblearn.over_sampling import SMOTENC

# ===== Reproducibility =====
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# ===== Load & prepare =====
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

# ===== Preprocessing helper =====
scaler = None
pca = None

def data_preprocessing(X, y, fit):
    # Log1p
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    X = log_transformer.transform(X)
    X = pd.DataFrame(X)
    X.fillna(0, inplace=True)

    # Scale
    global scaler
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # PCA 99%
    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y

# ===== Train / Test split =====
X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===== SMOTENC (oversample, giữ 'gender' là categorical) =====
sm = SMOTENC(categorical_features=[X.columns.get_loc('gender')], random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("Train size after SMOTENC:", X_train.shape[0])

# ===== Preprocess =====
X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test,  y_test  = data_preprocessing(X_test,  y_test,  fit=False)

# Reshape for LSTM input: (samples, timesteps=1, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ===== Build LSTM model =====
lstm_model = Sequential([
    LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
    Dropout(0.4),
    LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0001)
lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ===== Train =====
history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# ===== Save training history =====
history_df = pd.DataFrame(history.history)
history_df.index += 1
history_df.reset_index(inplace=True)
history_df.rename(columns={"index": "Epoch"}, inplace=True)
history_df.to_excel("SMOTENC_LSTM_history.xlsx", index=False)

# ===== Evaluate =====
loss, accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# ===== Predict & metrics =====
y_pred_prob = lstm_model.predict(X_test, verbose=0).ravel()  # đảm bảo 1D cho AUC
threshold = 0.74
y_pred_adjusted = (y_pred_prob > threshold).astype("int32")

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

# ===== Confusion Matrix — print & save (counts + normalized) =====
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

# Counts
cm = confusion_matrix(y_test, y_pred_adjusted)
print("\nConfusion Matrix (Counts):")
print(cm)

cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cm_df.to_csv("reports/confusion_matrix_counts_SMOTENC_LSTM.csv", index=True)

fig_c, ax_c = plt.subplots(figsize=(6, 5))
disp_c = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetes', 'Diabetes'])
disp_c.plot(ax=ax_c, values_format='d', colorbar=False)
ax_c.set_title("Confusion Matrix (Counts)")
plt.tight_layout()
fig_c.savefig("reports/confusion_matrix_counts_SMOTENC_LSTM.png", dpi=300)
plt.close(fig_c)

# Row-normalized
cm_norm = confusion_matrix(y_test, y_pred_adjusted, normalize='true')
print("\nConfusion Matrix (Row-normalized):")
print(np.round(cm_norm, 3))

cmn_df = pd.DataFrame(cm_norm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cmn_df.to_csv("reports/confusion_matrix_normalized_SMOTENC_LSTM.csv", index=True)

fig_n, ax_n = plt.subplots(figsize=(6, 5))
disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Non-Diabetes', 'Diabetes'])
disp_n.plot(ax=ax_n, values_format='.2f', colorbar=False)
ax_n.set_title("Confusion Matrix (Row-normalized)")
plt.tight_layout()
fig_n.savefig("reports/confusion_matrix_normalized_SMOTENC_LSTM.png", dpi=300)
plt.close(fig_n)

print("\n✅ Confusion matrices saved in 'reports/':")
print(" - reports/confusion_matrix_counts_SMOTENC_LSTM.csv")
print(" - reports/confusion_matrix_counts_SMOTENC_LSTM.png")
print(" - reports/confusion_matrix_normalized_SMOTENC_LSTM.csv")
print(" - reports/confusion_matrix_normalized_SMOTENC_LSTM.png")

# ===== Save model & scaler =====
import joblib
import pickle

lstm_model.save('SMOTENC+LSTM_model.h5')
joblib.dump(scaler, 'scaler.pkl')

with open('SMOTENC+LSTM.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("\n✅ Saved artifacts:")
print(" - SMOTENC+LSTM_model.h5")
print(" - scaler.pkl")
print(" - SMOTENC+LSTM.pkl")
print(" - SMOTENC_LSTM_history.xlsx")
