# main.py — Voting (average probability) across your trained models

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# =========================
# 1) Load & chuẩn hóa dữ liệu gốc (giống các script train)
# =========================
data = pd.read_csv('diabetes_prediction_dataset.csv')
data.dropna(inplace=True)
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
data.fillna(0, inplace=True)

X = data.drop(['diabetes'], axis=1).astype('float32')
y = data['diabetes'].astype('int32')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# =========================
# 2) Danh sách file mô hình đã train
#    (khớp với các script bạn cung cấp)
# =========================
model_files = [
    # Stacking (sklearn)
    "SMOTENC_stacking_model.joblib",  # từ SMOTENC+STK.py  :contentReference[oaicite:0]{index=0}
    "GAN_stacking_model.joblib",      # từ GAN+STK.py      :contentReference[oaicite:1]{index=1}

    # Keras (h5)
    "MLP_SMOTENC_model.h5",           # từ SMOTENC+MLP.py  (có PCA)  :contentReference[oaicite:2]{index=2}
    "SMOTENC+LSTM_model.h5",          # từ SMOTENC+LSTM.py (có PCA)  :contentReference[oaicite:3]{index=3}
    "lstm_gan_model.h5",              # từ GAN+LSTM.py     (có PCA)  :contentReference[oaicite:4]{index=4}
    "MLP_gan_model.h5",               # từ GAN+MLP.py      (không PCA) :contentReference[oaicite:5]{index=5}
]

# =========================
# 3) Helpers
# =========================
def is_lstm_model(name: str) -> bool:
    return "LSTM" in name.upper()

def needs_pca(filename: str) -> bool:
    """
    Suy luận mô hình này đã train với PCA hay chưa dựa vào script:
    - Có PCA: SMOTENC_stacking, GAN_stacking, SMOTENC+LSTM, GAN+LSTM, SMOTENC+MLP
    - Không PCA: GAN+MLP
    """
    fname = filename.lower()
    has_pca_keywords = [
        "smotenc_stacking_model",
        "gan_stacking_model",
        "smotenc+lstm_model",
        "lstm_gan_model",
        "mlp_smotenc_model",
    ]
    no_pca_keywords = [
        "mlp_gan_model",
    ]
    if any(k in fname for k in no_pca_keywords):
        return False
    if any(k in fname for k in has_pca_keywords):
        return True
    # Mặc định đoán là có PCA (an toàn hơn cho phần lớn file của bạn)
    return True

def load_any(path: str):
    """Tự động load theo đuôi file. Trả về object đã load."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    if path.endswith(".joblib"):
        import joblib
        return joblib.load(path)
    if path.endswith(".h5"):
        from tensorflow.keras.models import load_model
        return load_model(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def to_proba(model, X_in):
    """
    Trả về xác suất lớp dương (shape: (n,)).
    - sklearn: dùng predict_proba nếu có
    - keras: model.predict
    """
    # sklearn
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_in)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()

    # keras / tf
    if hasattr(model, "predict"):
        yhat = model.predict(X_in, verbose=0)
        yhat = np.asarray(yhat).ravel()
        # ép về [0,1] nếu cần
        if (yhat.min() < 0) or (yhat.max() > 1):
            yhat = 1.0 / (1.0 + np.exp(-yhat))
        return yhat

    # fallback
    yhat = model.predict(X_in)
    yhat = np.asarray(yhat).ravel()
    if (yhat.min() < 0) or (yhat.max() > 1):
        yhat = 1.0 / (1.0 + np.exp(-yhat))
    return yhat

def get_expected_features(model):
    """
    Suy ra số feature mà model kỳ vọng:
    - sklearn: n_features_in_
    - keras Dense: model.input_shape[-1]
    - keras LSTM: model.input_shape[-1] (số đặc trưng mỗi timestep)
    """
    exp = getattr(model, "n_features_in_", None)
    if exp is not None:
        return int(exp)

    in_shape = getattr(model, "input_shape", None)
    if in_shape is None:
        return None
    if isinstance(in_shape, list):
        shape = in_shape[0]
    else:
        shape = in_shape

    if shape is None:
        return None
    if len(shape) == 2:
        return None if shape[-1] is None else int(shape[-1])
    if len(shape) >= 3:
        return None if shape[-1] is None else int(shape[-1])
    return None

# Biến đổi log1p + scale (và tùy chọn PCA khớp số chiều exp_feat)
def make_transform(X_train_df, X_test_df, use_pca: bool, n_components: int | None):
    Xtr = np.asarray(X_train_df, dtype=np.float32)
    Xte = np.asarray(X_test_df, dtype=np.float32)

    # clip âm rồi log1p (theo các script)
    Xtr = np.log1p(np.clip(Xtr, a_min=0, a_max=None))
    Xte = np.log1p(np.clip(Xte, a_min=0, a_max=None))

    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    if use_pca:
        if n_components is None:
            raise ValueError("n_components=None nhưng yêu cầu PCA.")
        pca = PCA(n_components=n_components, random_state=42)
        Xtr = pca.fit_transform(Xtr)  # fit để xác lập không gian
        Xte = pca.transform(Xte)

    return Xte

# Cache theo (use_pca, n_components) để tiết kiệm tính toán
_cache = {}

def get_transformed_X(use_pca: bool, n_components: int | None):
    key = (use_pca, int(n_components) if n_components is not None else -1)
    if key in _cache:
        return _cache[key]
    Xte = make_transform(X_train, X_test, use_pca=use_pca, n_components=n_components)
    _cache[key] = Xte
    return Xte

# =========================
# 4) Ensemble predict
# =========================
all_probas = []
used_models = []

for file in model_files:
    if not os.path.exists(file):
        print(f"[SKIP] {file} không tồn tại, bỏ qua.")
        continue

    try:
        loaded = load_any(file)
    except Exception as e:
        print(f"[WARN] Không thể load {file}: {e}")
        continue

    model = loaded.get("model") if isinstance(loaded, dict) else loaded
    if model is None:
        print(f"[WARN] {file}: model=None (pickle Keras?). Bỏ qua.")
        continue

    exp_feat = get_expected_features(model)
    use_pca = needs_pca(file)

    X_in = None
    tried = []

    # Nếu biết số feature kỳ vọng:
    if exp_feat is not None:
        try:
            X_in_base = get_transformed_X(use_pca=use_pca, n_components=exp_feat if use_pca else None)
            tried.append((use_pca, exp_feat))
            X_in = X_in_base
        except Exception as e:
            print(f"[WARN] {file}: không tạo được biến đổi (use_pca={use_pca}, n={exp_feat}) -> {e}")

    # Fallback nếu exp_feat None: thử vài giá trị hay gặp
    if X_in is None:
        for n_try in (11, 10, 12, X_test.shape[1]):
            try:
                X_in_base = get_transformed_X(use_pca=use_pca, n_components=n_try if use_pca else None)
                tried.append((use_pca, n_try))
                X_in = X_in_base
                break
            except Exception:
                continue

    if X_in is None:
        print(f"[WARN] {file}: không tạo được bất kỳ biến đổi nào. Bỏ qua.")
        continue

    # LSTM cần reshape (n, 1, d)
    if is_lstm_model(file):
        X_in = X_in.reshape((X_in.shape[0], 1, X_in.shape[1]))

    try:
        proba = to_proba(model, X_in).ravel()
        if proba.shape[0] != X_test.shape[0]:
            raise ValueError(f"Output len {proba.shape[0]} != n_samples {X_test.shape[0]}")
        all_probas.append(proba)
        used_models.append(f"{file} | use_pca={use_pca} | d={X_in.shape[-1]} | tried={tried}")
        print(f"[OK] {file}: dùng {len(proba)} xác suất.")
    except Exception as e:
        print(f"[WARN] {file} dự đoán lỗi: {e}")

if not all_probas:
    raise RuntimeError("❌ Không có mô hình nào dự đoán được. Kiểm tra lại cách lưu model và pipeline.")

avg_proba = np.mean(np.vstack(all_probas), axis=0)

# =========================
# 5) Đánh giá & báo cáo
# =========================
threshold = 0.7  # có thể tinh chỉnh theo validation
y_pred = (avg_proba >= threshold).astype(int)

print("\n=== Models sử dụng trong ensemble ===")
for m in used_models:
    print(" -", m)

print("\n=== Metrics (threshold = {:.2f}) ===".format(threshold))
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred, zero_division=0))
print("ROC AUC  :", roc_auc_score(y_test, avg_proba))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, digits=5, zero_division=0))
