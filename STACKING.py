import numpy as np
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier  # sử dụng scikeras để bọc mô hình Keras
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

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

def build_stacking_model(random_state=42):
    """
    Xây dựng mô hình Stacking sử dụng XGBoost và AdaBoost làm base learner,
    Logistic Regression làm meta-classifier.
    """
    # Định nghĩa các base estimators
    base_estimators = [
        ('xgb', XGBClassifier(random_state=random_state)),
        ('ada', AdaBoostClassifier(random_state=random_state))
    ]
    
    # Meta-estimator: Logistic Regression
    final_estimator = LogisticRegression(random_state=random_state)
    
    # Xây dựng mô hình stacking
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,              # Sử dụng 5-fold cross-validation
        passthrough=True   # Truyền thêm đặc trưng gốc cho meta-estimator
    )
    return stacking_model
stack_model = build_stacking_model(random_state=42)
stack_model.fit(X_train, y_train)

# -------------------- 4. Đánh giá mô hình --------------------
y_pred = stack_model.predict(X_test)

print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print(f"Accuracy: {stack_model.score(X_test, y_test):.2f}")

