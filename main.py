import os
import io
import pickle
import warnings
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

import streamlit as st
from training.preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from sklearn.metrics.pairwise import cosine_similarity

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
st.title("🩺 Diabetes Modeling Suite — Suitable Diet Suggestions")

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
    df.dropna(inplace=True)
    if "gender" not in df.columns:
        df["gender"] = 0 
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(0)
    if "smoking_history" not in df.columns:
        df["smoking_history"] = "never"
    df = pd.get_dummies(df, columns=["smoking_history"], drop_first=True)
    df.fillna(0, inplace=True)
    if "diabetes" in df.columns:
        y = df["diabetes"].astype("int32")
        X = df.drop(["diabetes"], axis=1).astype("float32")
    else:
        y = None
        X = df.astype("float32")
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
    st.markdown("""
        <style>
            [data-testid="stSidebar"] img {
                width: 600px !important;  /* tăng size ảnh lên 600px */
                max-width: none !important;
                margin-left: -160px !important;  /* điều chỉnh margin để căn giữa */
                margin-top: -150px !important;  /* giảm khoảng trống phía trên */
                margin-bottom: -50px !important;  /* giảm khoảng cách phía dưới */
                position: relative !important;
                z-index: 1000 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    try:
        
        st.sidebar.image("logo.png", use_container_width=False)
    except:
        st.sidebar.markdown("""
            <div style="text-align: center; font-size: 3em; margin: 50px 0; color: #1f77b4;">
                🩺 Diabetes AI
            </div>
        """, unsafe_allow_html=True)



    
    st.header("⚙️ Settings")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.70, 0.05)
    choice_eval_key = st.selectbox("Select best by", ["ROC AUC", "F1-score", "Accuracy"])
    use_ensemble = st.checkbox("Use ensemble (average probability)", value=True)
    st.markdown("---")
    st.caption("Model files scanned from working directory:")
    st.code("\n".join([m for m in MODEL_FILES if os.path.exists(m)]) or "(none)")

raw_df = load_dataset(None)
X_all, y_all = preprocess_dataframe(raw_df)

if y_all is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    transformer_cache = TransformerCache(X_train)
else:
    transformer_cache = None

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
    if y_all is not None:
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

        ens_acc = ens_prec = None
        if len(probas) > 0:
            avg_proba_test = np.mean(np.vstack(probas), axis=0)
            y_pred_ens_test = (avg_proba_test >= threshold).astype(int)
            ens_acc = accuracy_score(y_test, y_pred_ens_test)
            ens_prec = precision_score(y_test, y_pred_ens_test, zero_division=0)

        st.session_state["metrics_df"] = metrics_df
        st.session_state["best_model_name"] = best_row["model"]
        st.session_state["use_ensemble"] = use_ensemble
        st.session_state["threshold"] = threshold
        
    else:
        st.info("Đã tải dữ liệu bệnh nhân mới. Chỉ thực hiện dự đoán, không đánh giá mô hình.")

# =============================
# Food Recommender Configuration
# =============================
FOOD_CSV_CANDIDATES = ["food/food_data.csv", "food_data.csv"]
FOOD_DEFAULT_OUTPUT = "food.json"
FOOD_FEATURES_BASE = ["Dietary Preference", "heart_disease", "hypertension", "BMI", "Daily Calorie Target"]
FOOD_WEIGHTS = {"Dietary Preference": 3.0, "heart_disease": 2.0, "hypertension": 2.0, "BMI": 1.0, "Daily Calorie Target": 1.0}
FOOD_REQUIRED_EXPORT_COLS = ["Breakfast Suggestion","Lunch Suggestion","Dinner Suggestion","Snack Suggestion","Protein","Sugar","Sodium","Carbohydrates","Fiber","Fat","Calories"]

def _food_kcal(weight, height, activity_level, gender, age):
    gender = str(gender).lower()
    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    activity_factors = [1.2, 1.375, 1.55, 1.725, 1.9]
    return bmr * activity_factors[int(activity_level)]

def _food_bmi(weight, height_cm):
    h_m = height_cm / 100.0
    return weight / (h_m ** 2)

def _food_standardize_numeric(train_values, user_values, cols):
    scaler = StandardScaler()
    train_values[cols] = scaler.fit_transform(train_values[cols])
    user_values[cols] = scaler.transform(user_values[cols])
    return train_values, user_values

def _food_build_weight_vector(columns, weights_dict):
    return np.array([weights_dict.get(c, 1.0) for c in columns], dtype=float)

def _food_recommend(user_profile, df, top_k=5):
    data = df[FOOD_FEATURES_BASE].copy()
    u = {
        "Dietary Preference": float(user_profile.get("Dietary Preference", data["Dietary Preference"].mode()[0])),
        "heart_disease": float(user_profile.get("heart_disease", 0)),
        "hypertension": float(user_profile.get("hypertension", 0)),
        "BMI": float(user_profile.get("BMI", float(data["BMI"].median()))),
        "Daily Calorie Target": float(user_profile.get("Daily Calorie Target", float(data["Daily Calorie Target"].median()))),
    }
    user_df = pd.DataFrame([u], columns=data.columns)
    numeric_continuous = ["BMI", "Daily Calorie Target"]
    data_scaled, user_scaled = _food_standardize_numeric(data.copy(), user_df.copy(), numeric_continuous)
    W = _food_build_weight_vector(data_scaled.columns.tolist(), FOOD_WEIGHTS)
    Xw = data_scaled.values * W
    Uw = user_scaled.values * W
    sims = cosine_similarity(Uw, Xw)[0]
    out = df.copy()
    out["similarity"] = sims
    out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out.head(top_k)

def _food_load_default_csv():
    for p in FOOD_CSV_CANDIDATES:
        if os.path.exists(p):
            return pd.read_csv(p), p
    raise FileNotFoundError(", ".join(FOOD_CSV_CANDIDATES))

# =============================
# 4) Tabs for Prediction and Meal Recommendation
# =============================
st.markdown("---")

tab1, tab2 = st.tabs(["🧍 Predict for a New Patient", "🥗 Personalized Meal Recommendation"])
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 12px 32px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# TAB 1: Single Patient Prediction
# =============================
with tab1:
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
            input_df = pd.DataFrame({
                "gender": [1 if gender == "Male" else 0],
                "age": [age],
                "hypertension": [htn],
                "heart_disease": [heart],
                "bmi": [bmi],
                "HbA1c_level": [hba1c],
                "blood_glucose_level": [glucose]
            })

            # Manually create smoking history columns (matching training preprocessing)
            smoking_columns = {
                'smoking_history_current': 1 if smoke == 'current' else 0,
                'smoking_history_ever': 1 if smoke == 'ever' else 0,
                'smoking_history_former': 1 if smoke == 'former' else 0,
                'smoking_history_never': 1 if smoke == 'never' else 0,
                'smoking_history_not current': 1 if smoke == 'not current' else 0
            }
            
            for col, value in smoking_columns.items():
                input_df[col] = value
            
            # Final column order (matches training data)
            expected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 
                              'HbA1c_level', 'blood_glucose_level',
                              'smoking_history_current', 'smoking_history_ever', 
                              'smoking_history_former', 'smoking_history_never', 
                              'smoking_history_not current']
            
            input_df = input_df[expected_columns]
                    
            preprocessor = DataPreprocessor.load('model/preprocessor.pkl')
            X_tmp = preprocessor.transform(input_df)
            
            
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

            X_in = X_tmp.copy()
            
            if "gan_stacking" in file.lower():

                if isinstance(loaded, dict) and 'scaler' in loaded and 'pca' in loaded:
                    try:
                        input_raw = pd.DataFrame({
                            "gender": [1 if gender == "Male" else 0],
                            "age": [float(age)],
                            "hypertension": [float(htn)],
                            "heart_disease": [float(heart)],
                            "bmi": [float(bmi)],
                            "HbA1c_level": [float(hba1c)],
                            "blood_glucose_level": [float(glucose)]
                        })
                        
                        # For GAN preprocessing (also uses drop_first=True, drops 'No Info')
                        smoking_cols = ['smoking_history_current', 'smoking_history_ever', 
                                      'smoking_history_former', 'smoking_history_never', 
                                      'smoking_history_not current']
                        for col in smoking_cols:
                            input_raw[col] = 0.0
                        
                        if smoke == "never":
                            input_raw["smoking_history_never"] = 1.0
                        elif smoke == "current":
                            input_raw["smoking_history_current"] = 1.0
                        elif smoke == "ever":
                            input_raw["smoking_history_ever"] = 1.0
                        elif smoke == "former":
                            input_raw["smoking_history_former"] = 1.0
                        elif smoke == "not current":
                            input_raw["smoking_history_not current"] = 1.0
                        # Note: 'No Info' results in all columns = 0 (dropped by drop_first=True)

                        log_transformer = loaded.get('log_transformer')
                        scaler_gan = loaded.get('scaler') 
                        pca_gan = loaded.get('pca')

                        X_gan = log_transformer.transform(input_raw.values)
                        X_gan = pd.DataFrame(X_gan, columns=input_raw.columns).fillna(0)
                        
                        X_gan_scaled = scaler_gan.transform(X_gan.values)
                        
                        X_in = pca_gan.transform(X_gan_scaled)
                        
                    except Exception as e:
                        X_in = X_tmp.copy()
                        if X_in.shape[1] > 10:
                            X_in = X_in[:, :10]
                else:
                    X_in = X_tmp.copy()
                    if X_in.shape[1] == 11:
                        X_in = X_in[:, [0,1,2,3,4,5,6,7,8,9]]  
                        
            elif "GAN" in file.upper() or "gan" in file.lower():
                if X_in.shape[1] == 11:
                    X_in = X_in[:, :10]
            
            if is_lstm_model(file):
                X_in = X_in.reshape((X_in.shape[0], 1, X_in.shape[1]))

            try:
                proba = float(to_proba(model, X_in).ravel()[0])
            except Exception:
                continue
            pred = int(proba >= st.session_state.get("threshold", 0.7))
            probs_for_ensemble.append(proba)

            model_name = os.path.basename(file).replace(".joblib", "").replace(".h5", "").replace("_", " ").title()
            
            rows_for_table.append({
                "model": model_name,
                "probability": round(proba, 4),
                "label": pretty_label(pred),
            })

        if st.session_state.get("use_ensemble", False) and len(probs_for_ensemble) > 0:
            final_proba = float(np.mean(probs_for_ensemble))
            pred = int(final_proba >= st.session_state.get("threshold", 0.7))
            rows_for_table.append({
                "model": "Ensemble (Average)",
                "probability": round(final_proba, 4),
                "label": pretty_label(pred),
            })

        if len(rows_for_table) == 0:
            st.error("❌ Không thể dự đoán với bất kỳ mô hình nào.")
        else:
            result_df = pd.DataFrame(rows_for_table).sort_values("probability", ascending=False).reset_index(drop=True)
            # Reorder columns - chỉ còn 3 cột
            result_df = result_df[["model", "probability", "label"]]
            st.subheader("📋 Kết quả dự đoán cho tất cả mô hình")
            st.dataframe(result_df, use_container_width=True)

# =============================
# TAB 2: Personalized Meal Recommendation  
# =============================
# TAB 2: Personalized Meal Recommendation  
# =============================
with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        food_age = st.number_input("Age", 1, 100, 25, key="food_age")
        food_height = st.number_input("Height (cm)", 80.0, 250.0, 175.0, 0.1, key="food_h")
        food_weight = st.number_input("Weight (kg)", 20.0, 250.0, 70.0, 0.1, key="food_w")
    with c2:
        food_activity = st.select_slider("Activity Level (0–4)", options=[0,1,2,3,4], value=1, key="food_act")
        food_htn = st.selectbox("Hypertension", [0, 1], index=1, key="food_htn")
        food_hd = st.selectbox("Heart disease", [0, 1], index=0, key="food_hd")
    with c3:
        food_gender = st.selectbox("Gender", ["Female", "Male"], index=0, key="food_gender")
        food_pref_label = st.selectbox("Dietary Preference", ["Omnivore", "Vegetarian"], index=0, key="food_pref")

    _food_bmi_val = _food_bmi(food_weight, food_height)
    _food_kcal_val = _food_kcal(food_weight, food_height, food_activity, food_gender, food_age)
    met1, met2 = st.columns(2)
    with met1:
        st.metric("BMI", f"{_food_bmi_val:.2f}")
    with met2:
        st.metric("Daily Calorie Target", f"{int(_food_kcal_val):,} kcal")

    colk, coln = st.columns(2)
    with colk:
        food_topk = st.slider("Top-K suggestions", 1, 10, 5, 1, key="food_topk")

    if st.button("🔎 Recommend Meals", key="food_btn"):

        try:
            df_food, found_path = _food_load_default_csv()
            csv_source = found_path
        except Exception as e:
            st.error(f"❌ Cannot load food dataset: {e}")
            st.stop()


        missing = [c for c in FOOD_FEATURES_BASE if c not in df_food.columns]
        if missing:
            st.error(f"Dataset is missing required feature columns: {missing}")
            st.stop()

        pref_map = {"Omnivore": 1.0, "Vegetarian": 0.0}
        user_profile = {
            "Dietary Preference": pref_map.get(food_pref_label, 1.0),
            "heart_disease": food_hd,
            "hypertension": food_htn,
            "BMI": _food_bmi_val,
            "Daily Calorie Target": _food_kcal_val,
        }

        try:
            top_df = _food_recommend(user_profile=user_profile, df=df_food, top_k=int(food_topk))
        except Exception as e:
            st.error(f"Recommendation failed: {e}")
            st.stop()

        st.dataframe(top_df, use_container_width=True)

# =============================
# Footer - Thông tin đề tài
# =============================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <p style="font-size: 14px; font-weight: bold; margin-bottom: 10px; color: #1f77b4;">
            GVHD: Hoàng Văn Dũng
        </p>
        <p style="font-size: 14px; margin-bottom: 0; color: #FFD700;">
            Sinh viên thực hiện đề tài: Đặng Cửu Dương, Lê Thị Mỹ Dung
        </p>
    </div>
""", unsafe_allow_html=True)

# =============================
# Done
# =============================
