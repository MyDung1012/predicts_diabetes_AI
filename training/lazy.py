import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import all_estimators

# Đặt seed
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load dữ liệu
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

# Tách X, y
X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Dữ liệu lớp 1 cho GAN
data_diabetes = X_train[y_train == 1]

# Xây GAN   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

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

input_dim = data_diabetes.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)

def generate_synthetic_data(generator, n_samples):
    noise = np.random.normal(0, 1, size=(n_samples, generator.input_shape[1])).astype('float32')
    synthetic_data = generator.predict(noise, verbose=0)
    return synthetic_data

# Huấn luyện GAN
num_epochs = 100
batch_size = 64
half_batch = batch_size // 2

for epoch in range(num_epochs):
    x_real = data_diabetes.sample(half_batch).values.astype('float32')
    y_real = np.ones((half_batch, 1), dtype='float32')

    x_fake = generate_synthetic_data(generator, half_batch).astype('float32')
    y_fake = np.zeros((half_batch, 1), dtype='float32')

    discriminator.trainable = True
    discriminator.train_on_batch(x_real, y_real)
    discriminator.train_on_batch(x_fake, y_fake)

    noise = np.random.normal(0, 1, size=(batch_size, input_dim)).astype('float32')
    gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")

# Tạo dữ liệu tổng hợp
synthetic_data = generate_synthetic_data(generator, 60000)
df_synthetic = pd.DataFrame(synthetic_data, columns=X.columns)
df_synthetic['diabetes'] = 1

X_train_smote = df_synthetic.drop('diabetes', axis=1)
y_train_smote = df_synthetic['diabetes']

X_train = pd.concat([X_train, X_train_smote])
y_train = pd.concat([y_train, y_train_smote])

# Tiền xử lý
def data_preprocessing(X, y, fit):
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    X = log_transformer.transform(X)
    X = pd.DataFrame(X)
    X.fillna(0, inplace=True)

    global scaler
    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    global pca
    if fit:
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
    else:
        X = pca.transform(X)

    return X, y

X_train, y_train = data_preprocessing(X_train, y_train, fit=True)
X_test, y_test = data_preprocessing(X_test, y_test, fit=False)

# 🧠 LazyPredict
clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Hiển thị tất cả model
pd.set_option('display.max_rows', None)
print(models)
# Hiển thị 10 mô hình tốt nhất theo Accuracy
top_10_models = models.sort_values(by="Accuracy", ascending=False).head(10)
print("\n📊 Top 10 mô hình tốt nhất theo Accuracy:")
print(top_10_models)
