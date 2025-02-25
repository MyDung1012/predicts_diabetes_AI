import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# --------------- Thiết lập Seed ---------------
seed_value = 50
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------- 1. Load dữ liệu -----------q----
data = pd.read_csv('diabetes_prediction_dataset.csv')

# --------------- 2. Xử lý dữ liệu ---------------
# Xác định các cột số và cột phân loại
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']

# Với cột số: thay thế giá trị 0 bằng NaN và điền giá trị thiếu bằng trung bình
data[numeric_cols] = data[numeric_cols].replace(0, np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean(numeric_only=True))

# Với cột phân loại: điền giá trị thiếu bằng mode
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Chuyển 'gender' thành số: Male = 1, Female = 0
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

# One-hot encoding cho 'smoking_history'
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

# Kiểm tra lại xem có còn NaN không
if data.isnull().sum().sum() > 0:
    print("⚠️ Vẫn còn giá trị NaN, điền vào bằng median")
    data.fillna(data.median(numeric_only=True), inplace=True)

# --------------- 3. Chọn đặc trưng dựa trên Mutual Information ---------------
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Kiểm tra lại NaN trước khi tính Mutual Information
if X.isnull().sum().sum() > 0:
    print("⚠️ Vẫn còn NaN trong X trước khi tính MI! Điền vào bằng median.")
    X.fillna(X.median(numeric_only=True), inplace=True)

# Tính điểm Mutual Information
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns)

# Chọn các đặc trưng có MI > 0.01
selected_features = mi_scores[mi_scores > 0.01].index
X = X[selected_features]

# --------------- 4. Cân bằng dữ liệu sử dụng SMOTE ---------------
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# --------------- 5. Chuẩn hóa dữ liệu ---------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# --------------- 6. Chia tập train và test ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu: Chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sử dụng LazyClassifier để huấn luyện và so sánh các mô hình phân loại
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Hiển thị kết quả so sánh các mô hình
print(models)
print(predictions)
