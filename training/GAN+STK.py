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
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import joblib

# ======================
# 1) Reproducibility
# ======================
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# ======================
# 2) Load & prepare
# ======================
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

# ======================
# 3) Preprocessing helper
# ======================
scaler = None
pca = None

def data_preprocessing(X, y, fit):
    # Log1p transform
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

    # Cảnh báo nếu còn giá trị âm (hiếm khi xảy ra sau log1p + MinMax)
    if (np.array(X) < 0).any():
        print("⚠️ Warning: Negative values detected after transform.")

    # PCA (giữ 99% phương sai)
    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y

# ======================
# 4) Split
# ======================
X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes'].astype('int32')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Original feature shape:", X.shape)

# ======================
# 5) Build simple GAN to synthesize class-1
# ======================
# Only positives for GAN
data_diabetes = X_train[y_train == 1]

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

# Init GAN
input_dim = data_diabetes.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)

def generate_synthetic_data(generator, n_samples):
    noise = np.random.normal(0, 1, size=(n_samples, generator.input_shape[1])).astype('float32')
    synthetic_data = generator.predict(noise, verbose=0)
    return synthetic_data

# Train GAN
num_epochs = 100
batch_size = 64
half_batch = batch_size // 2

for epoch in range(num_epochs):
    # Real
    x_real = data_diabetes.sample(half_batch).values.astype('float32')
    y_real = np.ones((half_batch, 1), dtype='float32')
    # Fake
    x_fake = generate_synthetic_data(generator, half_batch).astype('float32')
    y_fake = np.zeros((half_batch, 1), dtype='float32')

    discriminator.trainable = True
    discriminator.train_on_batch(x_real, y_real)
    discriminator.train_on_batch(x_fake, y_fake)

    noise = np.random.normal(0, 1, size=(batch_size, input_dim)).astype('float32')
    gan.train_on_batch(noise, np.ones((batch_size, 1), dtype='float32'))

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")

# Generate positives
synthetic_data = generate_synthetic_data(generator, 60000)
df_synthetic = pd.DataFrame(synthetic_data, columns=X.columns)
df_synthetic['diabetes'] = 1

# Merge synthetic + real
X_train_smote = df_synthetic.drop('diabetes', axis=1)
y_train_smote = df_synthetic['diabetes']
X_train = pd.concat([X_train, X_train_smote], axis=0)
y_train = pd.concat([y_train, y_train_smote], axis=0)

# ======================
# 6) Preprocess (fit train, transform test)
# ======================
X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test, y_test = data_preprocessing(X_test, y_test, fit=False)

# ======================
# 7) Stacking model
# ======================
def build_stacking_model(random_state=50):
    base_estimators = [
        ('xgb', XGBClassifier(random_state=random_state)),
        ('ada', AdaBoostClassifier(random_state=random_state))
    ]
    final_estimator = LogisticRegression(random_state=random_state, max_iter=1000)

    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        passthrough=True
    )
    return stacking_model

stack_model = build_stacking_model(random_state=42)
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

# ======================
# 8) Metrics
# ======================
print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.5f}")
print(f"Recall: {recall_score(y_test, y_pred):.5f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.5f}")
print(f"Accuracy: {stack_model.score(X_test, y_test):.5f}")

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.5f}")

# ======================
# 9) Confusion Matrix — print & save (counts + normalized)
# ======================
os.makedirs("reports", exist_ok=True)

# Counts
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Counts):")
print(cm)

cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cm_df.to_csv("reports/confusion_matrix_counts_GAN_STK.csv", index=True)

fig_c, ax_c = plt.subplots(figsize=(6, 5))
disp_c = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetes', 'Diabetes'])
disp_c.plot(ax=ax_c, values_format='d', colorbar=False)
ax_c.set_title("Confusion Matrix (Counts)")
plt.tight_layout()
fig_c.savefig("reports/confusion_matrix_counts_GAN_STK.png", dpi=300)
plt.close(fig_c)

# Normalized (row-wise)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
print("\nConfusion Matrix (Row-normalized):")
print(np.round(cm_norm, 3))

cmn_df = pd.DataFrame(cm_norm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
cmn_df.to_csv("reports/confusion_matrix_normalized_GAN_STK.csv", index=True)

fig_n, ax_n = plt.subplots(figsize=(6, 5))
disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Non-Diabetes', 'Diabetes'])
disp_n.plot(ax=ax_n, values_format='.2f', colorbar=False)
ax_n.set_title("Confusion Matrix (Row-normalized)")
plt.tight_layout()
fig_n.savefig("reports/confusion_matrix_normalized_GAN_STK.png", dpi=300)
plt.close(fig_n)

# ======================
# 10) Save model với preprocessing components
# ======================
# Lưu cả model và preprocessing components
model_package = {
    'model': stack_model,
    'scaler': scaler,
    'pca': pca,
    'log_transformer': FunctionTransformer(np.log1p, validate=True)
}

joblib.dump(model_package, 'model/GAN_stacking_model.joblib')
print("\n✅ Saved model with preprocessing components: model/GAN_stacking_model.joblib")
