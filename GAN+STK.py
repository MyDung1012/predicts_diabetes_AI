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
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

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



# --------------- 5. Xây dựng mô hình Stacking ---------------
def build_stacking_model(random_state= 50):
    base_estimators = [
        ('xgb', XGBClassifier(random_state=random_state)),
        ('ada', AdaBoostClassifier(random_state=random_state))
    ]
    final_estimator = LogisticRegression(random_state=random_state)

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

print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.5f}")
print(f"Recall: {recall_score(y_test, y_pred):.5f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.5f}")
print(f"Accuracy: {stack_model.score(X_test, y_test):.5f}")

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Tính Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.5f}")

# Vẽ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=['Non-Diabetes', 'Diabetes']).plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.show()

import joblib

# Lưu mô hình
joblib.dump(stack_model, 'GAN_stacking_model.joblib')

