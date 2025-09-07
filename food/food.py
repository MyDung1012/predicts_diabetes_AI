import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json

# ==== 1) Cấu hình ====
CSV_PATH = "food_data.csv"
DEFAULT_OUTPUT = "food.json"

FEATURES_BASE = [
    "Dietary Preference",
    "heart_disease",
    "hypertension",
    "BMI",
    "Daily Calorie Target",
]

WEIGHTS = {
    "Dietary Preference": 3.0,
    "heart_disease": 2.0,
    "hypertension": 2.0,
    "BMI": 1.0,
    "Daily Calorie Target": 1.0,
}

REQUIRED_EXPORT_COLS = [
    "Breakfast Suggestion","Lunch Suggestion","Dinner Suggestion","Snack Suggestion",
    "Protein","Sugar","Sodium","Carbohydrates","Fiber","Fat","Calories"
]

# ==== 2) Hàm tiện ích ====
def kcal_calculator(weight, height, activity_level, gender, age):
    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    activity_factors = [1.2, 1.375, 1.55, 1.725, 1.9]
    return bmr * activity_factors[int(activity_level)]

def bmi_calculator(weight, height):
    h_m = height / 100.0
    return weight / (h_m ** 2)

def standardize_numeric(train_values, user_values, cols):
    scaler = StandardScaler()
    train_values[cols] = scaler.fit_transform(train_values[cols])
    user_values[cols] = scaler.transform(user_values[cols])
    return train_values, user_values

def build_weight_vector(columns, weights_dict):
    return np.array([weights_dict.get(c, 1.0) for c in columns], dtype=float)

# ==== 3) Máy gợi ý ====
def recommend(user_profile, csv_path=CSV_PATH, top_k=5):
    df = pd.read_csv(csv_path)
    data = df[FEATURES_BASE].copy()

    u = {
        "Dietary Preference": float(user_profile.get("Dietary Preference", data["Dietary Preference"].mode()[0])),
        "heart_disease": float(user_profile.get("heart_disease", 0)),
        "hypertension": float(user_profile.get("hypertension", 0)),
        "BMI": float(user_profile.get("BMI", float(data["BMI"].median()))),
        "Daily Calorie Target": float(user_profile.get("Daily Calorie Target", float(data["Daily Calorie Target"].median()))),
    }
    user_df = pd.DataFrame([u], columns=data.columns)

    numeric_continuous = ["BMI", "Daily Calorie Target"]
    data_scaled, user_scaled = standardize_numeric(data.copy(), user_df.copy(), numeric_continuous)

    W  = build_weight_vector(data_scaled.columns.tolist(), WEIGHTS)
    Xw = data_scaled.values * W
    Uw = user_scaled.values * W
    sims = cosine_similarity(Uw, Xw)[0]

    out = df.copy()
    out["similarity"] = sims
    out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out.head(top_k)

# ==== 4) Hàm chạy & xuất JSON Top-2 ====
def run_recommender(weight, height, age, gender, activity_level,
                    dietary_preference=1, heart_disease=0, hypertension=0,
                    csv_path=CSV_PATH, top_k=5):
    kcal = kcal_calculator(weight, height, activity_level, gender, age)
    bmi  = bmi_calculator(weight, height)
    user = {
        "Dietary Preference": dietary_preference,
        "heart_disease": heart_disease,
        "hypertension": hypertension,
        "BMI": bmi,
        "Daily Calorie Target": kcal,
    }
    return recommend(user_profile=user, csv_path=csv_path, top_k=top_k)

def food_recommendation(weight, height, age, gender, activity_level,
                     dietary_preference=1, heart_disease=0, hypertension=0,
                     csv_path=CSV_PATH, output_path=DEFAULT_OUTPUT):
    df_top = run_recommender(
        weight=weight, height=height, age=age, gender=gender, activity_level=activity_level,
        dietary_preference=dietary_preference, heart_disease=heart_disease, hypertension=hypertension,
        csv_path=csv_path, top_k=2
    )

    export_df = df_top.reindex(columns=REQUIRED_EXPORT_COLS)
    records = export_df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return output_path

# ==== 5) Ví dụ gọi nhanh ====
if __name__ == "__main__":
    path = food_recommendation(
        weight=70,
        height=175,
        age=25,
        gender="female",
        activity_level=1,   
        dietary_preference=1,
        heart_disease=0,
        hypertension=1
    )
    print(f"Đã xuất JSON: {path}")
