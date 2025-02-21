import numpy as np
import pandas as pd
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier  # sử dụng scikeras để bọc mô hình Keras

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# --------------- Thiết lập Seed ---------------
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------- 1. Load dữ liệu ---------------
data = pd.read_csv('diabetes.csv')

# --------------- 2. Xử lý dữ liệu ---------------
# Thay thế giá trị 0 bằng NaN cho các cột có khả năng bị thiếu
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)
# Điền giá trị thiếu bằng trung bình của từng cột
data.fillna(data.mean(), inplace=True)

# --------------- 3. Chọn đặc trưng dựa trên Mutual Information ---------------
X = data.drop('Outcome', axis=1)
y = data['Outcome']
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)
# Giữ lại các đặc trưng có MI > 0.01
selected_features = mi_scores[mi_scores > 0.01].index
X = X[selected_features]

# --------------- 4. Cân bằng dữ liệu sử dụng SMOTE ---------------
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# --------------- 5. Chuẩn hóa dữ liệu ---------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------- 6. Chia tập train và test ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------- 7. Transformer để reshape dữ liệu cho RNN ---------------
# Chuyển đổi dữ liệu từ (samples, features) thành (samples, timesteps, features)
class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, timesteps, features):
        self.timesteps = timesteps
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.reshape(X.shape[0], self.timesteps, self.features)

# --------------- 8. Xây dựng mô hình RNN (LSTM) ---------------
def create_rnn_model():
    model = Sequential()
    # Sử dụng Input layer để xác định hình dạng input
    model.add(Input(shape=(1, X_train.shape[1])))
    model.add(LSTM(128, return_sequences=True, activation='tanh',
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   recurrent_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(LSTM(64, return_sequences=False, activation='tanh',
                   kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Bọc mô hình RNN với scikeras
rnn_classifier = KerasClassifier(model=create_rnn_model, epochs=100, batch_size=64, verbose=0)

# Tạo pipeline cho RNN: bước reshape + mô hình RNN
rnn_pipeline = SkPipeline([
    ('reshape', ReshapeTransformer(timesteps=1, features=X_train.shape[1])),
    ('rnn', rnn_classifier)
])

# --------------- 9. Xây dựng mô hình XGBoost ---------------
xgb_classifier = XGBClassifier(
xgb_classifier = XGBClassifier(
    random_state=42,
    learning_rate=0.05,
    n_estimators=300,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    eval_metric='logloss'
)

)

# --------------- 10. Xây dựng Ensemble Stacking ---------------
estimators = [
    ('rnn', rnn_pipeline),
    ('xgb', xgb_classifier)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True
)

# --------------- 11. Huấn luyện mô hình Ensemble ---------------
stack_model.fit(X_train, y_train)

# --------------- 12. Đánh giá mô hình trên tập test ---------------
y_pred = stack_model.predict(X_test)

print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print(f"Accuracy: {stack_model.score(X_test, y_test):.2f}")