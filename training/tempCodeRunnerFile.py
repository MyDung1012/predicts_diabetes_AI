import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import joblib
import pickle

# =========================
# Reproducibility
# =========================
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# =========================
# Load & prepare the dataset
# =========================
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

# =========================
# Preprocessing helper
# =========================
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

# =========================
# Train/Test split
# =========================
X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================
# SMOTENC (oversampling, giữ 'gender' là categorical)
# =========================
sm = SMOTENC(categorical_features=[X.columns.get_loc('gender')], random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("Train size after SMOTENC:", X_train.shape[0])

# =========================
# Preprocess
# =========================
X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test,  y_test  = data_preprocessing(X_test,  y_test,  fit=False)

# =========================
# Build MLP model
# =========================
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
