import os
import io
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
from training.preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =============================
# Utility: Safe TF/Keras import
# =============================
try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:  # TensorFlow not available
    keras_load_model = None

# =============================
# App Title & Layout
# =============================
st.set_page_config(page_title="Diabetes Modeling Suite", layout="wide")
st.title("🩺 Diabetes Modeling Suite — Train, Compare, Predict")
st.caption("Upload or use the bundled dataset, evaluate your trained models, and predict for a new patient.")

# =============================
# 0) Data loading helpers
# =============================
DEFAULT_DATA_PATH = "diabetes_prediction_dataset.csv"
SMOKING_LEVELS = ["never", "former", "current", "ever", "not current", "No Info"]

@st.cache_data(show_spinner=False)
def load_dataset(file: Optional[io.BytesIO] = None) -> pd.DataFrame:
    if file is None:
        if not os.path.exists(DEFAULT_DATA_PATH):
            st.error(f"❌ Không tìm thấy file `{DEFAULT_DATA_PATH}`.")
            st.stop()
        df = pd.read_csv(DEFAULT_DATA_PATH)
    else:
        df = pd.read_csv(file)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if "diabetes" not in df.columns:
        st.error("❌ Dataset thiếu cột `diabetes` (nhãn). Hãy tải dataset đúng định dạng.")
        st.stop()
    df.dropna(inplace=True)
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(0)
    df = pd.get_dummies(df, columns=["smoking_history"], drop_first=True)
    df.fillna(0, inplace=True)
    y = df["diabetes"].astype("int32")
    X = df.drop(["diabetes"], axis=1).astype("float32")
    return X, y

# ======================================
# 1) Model files present in the directory
# ======================================
MODEL_FILES = [
    "model/SMOTENC_stacking_model.joblib",
    "model/GAN_stacking_model.joblib",
    "model/MLP_SMOTENC_model.h5",
    "model/SMOTENC+LSTM_model.h5",
    "model/lstm_gan_model.h5",
    "model/MLP_gan_model.h5",
]

# =============================
# 2) Model/Pipeline utilities
# =============================

def is_lstm_model(name: str) -> bool:
    return "LSTM" in name.upper()

def needs_pca(filename: str) -> bool:
    fname = filename.lower()
    has_pca_keywords = [
        "smotenc_stacking_model",
        "gan_stacking_model",
        "smotenc+lstm_model",
        "lstm_gan_model",
        "mlp_smotenc_model",
    ]
    no_pca_keywords = ["mlp_gan_model"]
    if any(k in fname for k in no_pca_keywords):
        return False
    if any(k in fname for k in has_pca_keywords):
        return True
    return True  # default safer

def load_any(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".joblib"):
        import joblib
        return joblib.load(path)
    if path.endswith(".h5"):
        if keras_load_model is None:
            raise RuntimeError("TensorFlow/Keras chưa được cài để load .h5")
        return keras_load_model(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def to_proba(model, X_in: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_in)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "predict"):
        yhat = model.predict(X_in, verbose=0)
        yhat = np.asarray(yhat).ravel()
        if (yhat.min() < 0) or (yhat.max() > 1):
            yhat = 1.0 / (1.0 + np.exp(-yhat))
        return yhat
    yhat = model.predict(X_in)
    yhat = np.asarray(yhat).ravel()
    if (yhat.min() < 0) or (yhat.max() > 1):
        yhat = 1.0 / (1.0 + np.exp(-yhat))
    return yhat

def get_expected_features(model) -> Optional[int]:
    exp = getattr(model, "n_features_in_", None)
    if exp is not None:
        return int(exp)
    in_shape = getattr(model, "input_shape", None)
    if in_shape is None:
        return None
    shape = in_shape[0] if isinstance(in_shape, list) else in_shape
    if shape is None:
        return None
    if len(shape) == 2:
        return None if shape[-1] is None else int(shape[-1])
    if len(shape) >= 3:
        return None if shape[-1] is None else int(shape[-1])
    return None

class TransformerCache:
    """Fit log1p + MinMaxScaler (+ optional PCA) on X_train, transform any X."""
    def __init__(self, X_train_df: pd.DataFrame):
        self.X_train_df = X_train_df
        self.cache: Dict[Tuple[bool, int], Tuple[MinMaxScaler, Optional[PCA]]] = {}

    def _fit(self, use_pca: bool, n_components: Optional[int]):
        Xtr = np.asarray(self.X_train_df, dtype=np.float32)
        Xtr = np.log1p(np.clip(Xtr, a_min=0, a_max=None))
        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(Xtr)
        pca = None
        if use_pca:
            if n_components is None:
                raise ValueError("n_components=None nhưng yêu cầu PCA")
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(Xtr)
        self.cache[(use_pca, n_components or -1)] = (scaler, pca)

    def transform(self, X_df: pd.DataFrame, use_pca: bool, n_components: Optional[int]) -> np.ndarray:
        key = (use_pca, n_components or -1)
        if key not in self.cache:
            self._fit(use_pca, n_components)
        scaler, pca = self.cache[key]
        X = np.asarray(X_df, dtype=np.float32)
        X = np.log1p(np.clip(X, a_min=0, a_max=None))
        X = scaler.transform(X)
        if pca is not None:
            X = pca.transform(X)
        return X

def pretty_label(pred: int) -> str:
    return "🔴 Có khả năng mắc bệnh" if int(pred) == 1 else "🟢 Không mắc bệnh"

# =============================
# Sidebar - Data & Settings
# =============================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload CSV (tùy chọn)", type=["csv"])
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.70, 0.05)
    choice_eval_key = st.selectbox("Select best by", ["ROC AUC", "F1-score", "Accuracy"])
    use_ensemble = st.checkbox("Use ensemble (average probability)", value=True)
    st.markdown("---")
    st.caption("Model files scanned from working directory:")
    st.code("\n".join([m for m in MODEL_FILES if os.path.exists(m)]) or "(none)")

# Load data
raw_df = load_dataset(uploaded)
X_all, y_all = preprocess_dataframe(raw_df)

# Split (fit transformers & silent eval)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

transformer_cache = TransformerCache(X_train)

# =============================
# Silent evaluation (no UI table)
# =============================
@st.cache_data(show_spinner=True)
def evaluate_all_models_silent(
    X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> Tuple[pd.DataFrame, List[np.ndarray], List[Dict]]:
    rows, probas_list, model_debug = [], [], []
    for file in MODEL_FILES:
        if not os.path.exists(file):
            continue
        try:
            loaded = load_any(file)
        except Exception as e:
            rows.append({"model": file, "status": f"load error: {e}"})
            continue

        model = loaded.get("model") if isinstance(loaded, dict) else loaded
        if model is None:
            rows.append({"model": file, "status": "model=None"})
            continue

        exp_feat = get_expected_features(model)
        use_pca = needs_pca(file)

        X_in, tried = None, []
        if exp_feat is not None:
            try:
                X_in = transformer_cache.transform(X_test, use_pca=use_pca, n_components=exp_feat if use_pca else None)
                tried.append((use_pca, exp_feat))
            except Exception as e:
                tried.append((use_pca, exp_feat, f"fail:{e}"))
                X_in = None

        if X_in is None:
            for n_try in (11, 10, 12, X_test.shape[1]):
                try:
                    X_in = transformer_cache.transform(X_test, use_pca=use_pca, n_components=n_try if use_pca else None)
                    tried.append((use_pca, n_try))
                    break
                except Exception:
                    continue

        if X_in is None:
            rows.append({"model": file, "status": "transform fail"})
            continue

        if is_lstm_model(file):
            X_in = X_in.reshape((X_in.shape[0], 1, X_in.shape[1]))

        try:
            proba = to_proba(model, X_in).ravel()
        except Exception as e:
            rows.append({"model": file, "status": f"predict error: {e}"})
            continue

        if proba.shape[0] != X_test.shape[0]:
            rows.append({"model": file, "status": f"size mismatch: {proba.shape[0]} vs {X_test.shape[0]}"})
            continue

        y_pred = (proba >= threshold).astype(int)
        row = {
            "model": file,
            "status": "OK",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, proba),
            "use_pca": use_pca,
            "dim": int(X_in.shape[-1]),
            "tried": str(tried),
        }
        rows.append(row)
        probas_list.append(proba)
        model_debug.append({"file": file, "use_pca": use_pca, "dim": int(X_in.shape[-1]), "tried": tried})

    df = pd.DataFrame(rows)
    return df, probas_list, model_debug

with st.spinner("Loading models..."):
    metrics_df, probas, model_dbg = evaluate_all_models_silent(X_test, y_test, threshold)

ok_df = metrics_df[metrics_df["status"] == "OK"].copy()
if ok_df.empty:
    st.error("Không có mô hình nào load được hoặc dự đoán thành công.")
    st.stop()

# Best model (silent)
if choice_eval_key == "ROC AUC":
    best_row = ok_df.loc[ok_df["ROC AUC"].idxmax()]
elif choice_eval_key == "F1-score":
    best_row = ok_df.loc[ok_df["F1-score"].idxmax()]
else:
    best_row = ok_df.loc[ok_df["Accuracy"].idxmax()]

# Pre-compute ensemble metrics on test set (for table)
ens_acc = ens_prec = None
if len(probas) > 0:
    avg_proba_test = np.mean(np.vstack(probas), axis=0)
    y_pred_ens_test = (avg_proba_test >= threshold).astype(int)
    ens_acc = accuracy_score(y_test, y_pred_ens_test)
    ens_prec = precision_score(y_test, y_pred_ens_test, zero_division=0)

# Keep some state
st.session_state["metrics_df"] = metrics_df
st.session_state["best_model_name"] = best_row["model"]
st.session_state["use_ensemble"] = use_ensemble
st.session_state["threshold"] = threshold

# =============================
# 4) Single Patient Prediction — SHOW ONLY THIS TABLE
# =============================
st.markdown("---")
st.subheader("🧍 Predict for a New Patient")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=40, step=1)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
    bmi = weight / ((height / 100) ** 2)
    st.number_input("BMI (Calculated)", value=bmi, disabled=True, format="%.2f")
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    htn = st.selectbox("Hypertension", [0, 1], index=0)
with col3:
    heart = st.selectbox("Heart disease", [0, 1], index=0)

col4, col5 = st.columns(2)
with col4:
    hba1c = st.number_input("HbA1c level", 0.0, 20.0, 5.5, 0.1)
with col5:
    glucose = st.number_input("Blood glucose level", 0.0, 400.0, 120.0, 1.0)

smoke = st.selectbox("Smoking history", SMOKING_LEVELS, index=0)

if st.button("🔮 Predict"):
    try:
        # Build a one-row dataframe matching training preprocessing
        input_df = pd.DataFrame({
            "gender": [1 if gender == "Male" else 0],
            "age": [age],
            "hypertension": [htn],
            "heart_disease": [heart],
            "smoking_history": [smoke],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        })

        # Apply preprocessing
        input_df = pd.get_dummies(input_df, columns=['smoking_history'], drop_first=True)
        
        # Thêm các cột còn thiếu nếu có
        expected_columns = ['smoking_history_ever', 'smoking_history_former', 
                          'smoking_history_never', 'smoking_history_not current', 
                          'smoking_history_current']
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Load và áp dụng preprocessor
        preprocessor = DataPreprocessor.load('model/preprocessor.pkl')
        X_tmp = preprocessor.transform(input_df)
        
        # X_tmp đã là numpy array được xử lý sẵn, không cần transformer_cache nữa
        
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình tiền xử lý dữ liệu: {str(e)}")
        st.stop()

    rows_for_table: List[Dict] = []
    probs_for_ensemble: List[float] = []

    for file in MODEL_FILES:
        if not os.path.exists(file):
            continue
        try:
            loaded = load_any(file)
        except Exception:
            continue
        model = loaded.get("model") if isinstance(loaded, dict) else loaded
        if model is None:
            continue

        # Sử dụng X_tmp đã được preprocessor xử lý sẵn
        X_in = X_tmp.copy()
        
        # Reshape cho LSTM nếu cần
        if is_lstm_model(file):
            # X_tmp có shape (1, 11), cần reshape thành (1, 1, 11) cho LSTM
            X_in = X_in.reshape((X_in.shape[0], 1, X_in.shape[1]))

        try:
            proba = float(to_proba(model, X_in).ravel()[0])
        except Exception:
            continue

        # decide label for this case
        pred = int(proba >= st.session_state["threshold"])
        probs_for_ensemble.append(proba)

        rows_for_table.append({
            "model": file,
            "probability": round(proba, 4),
            "label": pretty_label(pred),
        })

    if len(rows_for_table) == 0:
        st.error(" Không thể dự đoán với bất kỳ mô hình nào.")
    else:
        # Ensemble row
        if st.session_state["use_ensemble"] and len(probs_for_ensemble) > 0:
            final_proba = float(np.mean(probs_for_ensemble))
            pred = int(final_proba >= st.session_state["threshold"])
            rows_for_table.append({
                "model": "Ensemble (average)",
                "probability": round(final_proba, 4),
                "label": pretty_label(pred),
            })

        result_df = pd.DataFrame(rows_for_table).sort_values("probability", ascending=False).reset_index(drop=True)
        # Reorder columns - chỉ còn 3 cột
        result_df = result_df[["model", "probability", "label"]]
        st.subheader("📋 Kết quả cho ca bệnh vừa nhập")
        st.dataframe(result_df, use_container_width=True)

# =============================
# Done
# =============================
st.caption("Tip: place this file alongside your saved model files and `diabetes_prediction_dataset.csv`, then run `streamlit run app_streamlit_diabetes.py`.")
