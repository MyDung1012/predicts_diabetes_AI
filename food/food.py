import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re
import random

# Load API key
load_dotenv()

client = OpenAI(api_key=api_key)

MODEL = "gpt-4o"
TEMPERATURE = 1.0

# Tính kcal
def calculate_calories(weight, height, sex, activity_level, age, diabetes_type=False):
    if sex.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    multiplier = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9
    }.get(activity_level.lower(), 1.2)
    calories = bmr * multiplier
    if diabetes_type:
        calories *= 0.9
    return round(calories, 2)

# Prompt generator
def build_meal_plan_prompt(preferences, health, seed):
    recipe_form = {
        "name": "Dish name",
        "description": "Short description of the dish.",
        "nutrition": {``
            "calories": "230 kcal",
            "protein": "15g",
            "carbs": "30g",
            "fat": "10g",
            "fiber": "4g",
            "sugar": "6g",
            "sodium": "450mg"
        },
        "ingredients": [{"name": "example", "quantity": "100g"}],
        "steps": ["1. Step one", "2. Step two"],
        "spice_level": "medium",
        "diet_type": "eat clean",
        "suitable_for": ["muscle gain"],
        "highlighted_ingredients": ["chicken breast"],
        "avoided_ingredients": ["butter"],
    }

    pref_text = "\n".join([f"- {k}: {v}" for k, v in preferences.items()])
    health_text = "\n".join([f"- {k}: {v}" for k, v in health.items()])

    return f"""
You are a professional nutrition assistant. Please create a **1-day meal plan** for one person.

❤️ Preferences:
{pref_text}

💪 Health:
{health_text}

🔀 Randomization seed: {seed}

🧠 Rules:
- 3 meals: breakfast, lunch, dinner
- Each meal has 1 recipe
- Avoid allergens, respect dislikes
- Include: name, description, nutrition (calories, protein, carbs, fat, fiber, sugar, sodium), ingredients (with quantity), steps, spice level, diet type, suitable_for, highlighted_ingredients, avoided_ingredients
- Ensure variety. DO NOT repeat dishes across generations.
- Use international cuisines (Asian, Mediterranean, etc.)
- Calories goal: {health.get("calories_target", "N/A")} kcal

✍️ Output:
Return **pure JSON**, no markdown, using this format:

[
  {{
    "day": "Day 1",
    "meals": [
      {{
        "meal_name": "Breakfast",
        "recipes": [
          {json.dumps(recipe_form, indent=4, ensure_ascii=False)}
        ]
      }}
    ]
  }}
]
"""

# Xử lý JSON "giả"
def extract_json_from_text(text):
    if text.startswith("```json"):
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        first = text.index("[")
        last = text.rindex("]") + 1
        return text[first:last]
    except:
        return text

# Giao diện Streamlit
st.title("🥗 Gợi ý thực đơn đa dạng (3 bữa/ngày)")

with st.form("form"):
    st.subheader("📋 Thông tin cá nhân & sức khỏe")
    gender = st.selectbox("Giới tính", ["male", "female"])
    age = st.number_input("Tuổi", 10, 100, value=44)
    weight = st.number_input("Cân nặng (kg)", 30, 150, value=55)
    height = st.number_input("Chiều cao (cm)", 130, 220, value=160)
    activity_level = st.selectbox("Hoạt động", ["sedentary", "lightly active", "moderately active", "very active", "extra active"])
    hypertension = st.selectbox("Tăng huyết áp", ["yes", "no"])
    heart_disease = st.selectbox("Bệnh tim", ["yes", "no"])
    smoking_history = st.selectbox("Hút thuốc", ["yes", "no"])
    HbA1c = st.number_input("HbA1c (%)", 4.0, 15.0, value=6.5)
    glucose = st.number_input("Đường huyết (mg/dL)", 50, 400, value=200)
    diabetes = st.selectbox("Loại tiểu đường", ["type 1", "type 2", "none"])

    st.subheader("🥦 Sở thích ăn uống")
    spice = st.selectbox("Mức độ cay", ["mild", "medium", "spicy"])
    diet_type = st.selectbox("Chế độ ăn", ["vegetarian", "vegan", "eat clean", "normal"])
    allergies = st.text_input("Dị ứng (phân cách bằng dấu phẩy)", value="peanuts")
    favorite_foods = st.text_input("Món yêu thích", value="mushrooms")

    submitted = st.form_submit_button("Tạo thực đơn")

if submitted:
    with st.spinner("🔄 Đang tạo thực đơn..."):
        preferences = {
            "preferred_spice_level": spice,
            "diet_type": diet_type,
            "allergies": allergies,
            "favorite_foods": favorite_foods
        }

        bmi = round(weight / ((height / 100) ** 2), 1)
        health = {
            "gender": gender,
            "age": age,
            "bmi": bmi,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "HbA1c_level": HbA1c,
            "blood_glucose_level": glucose,
            "diabetes": diabetes,
            "activity_level": activity_level
        }

        diabetes_type = diabetes in ["type 1", "type 2"]
        target_kcal = calculate_calories(weight, height, gender, activity_level, age, diabetes_type)
        health["calories_target"] = target_kcal

        st.info(f"🎯 Lượng kcal cần thiết mỗi ngày: **{target_kcal} kcal**")

        random_seed = random.randint(0, 9999)
        prompt = build_meal_plan_prompt(preferences, health, seed=random_seed)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and professional nutrition assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )

        raw_text = response.choices[0].message.content
        json_text = extract_json_from_text(raw_text)

        try:
            meal_data = json.loads(json_text)
            day_plan = meal_data[0]["meals"]

            st.success("✅ Thực đơn đã tạo!")

            for meal in day_plan:
                meal_name = meal.get("meal_name", "Meal")
                total_kcal = 0
                for recipe in meal["recipes"]:
                    try:
                        kcal_str = recipe["nutrition"]["calories"]
                        kcal = float(kcal_str.lower().replace("kcal", "").strip())
                        total_kcal += kcal
                    except:
                        pass
                st.markdown(f"### 🍽️ {meal_name} — **{round(total_kcal)} kcal**")

                # 👇 Hiển thị chi tiết món ăn
                for idx, recipe in enumerate(meal["recipes"], start=1):
                    st.markdown(f"#### 🥘 Món {idx}: {recipe['name']}")
                    st.write(f"**Mô tả:** {recipe['description']}")

                    st.write("**📊 Dinh dưỡng:**")
                    for k, v in recipe["nutrition"].items():
                        st.write(f"- {k.capitalize()}: {v}")

                    st.write("**🧂 Nguyên liệu:**")
                    for ing in recipe["ingredients"]:
                        st.write(f"- {ing['name']}: {ing['quantity']}")

                    st.write("**👨‍🍳 Cách nấu:**")
                    for step in recipe["steps"]:
                        st.write(f"- {step}")

                    st.write(f"**Cấp độ cay:** {recipe.get('spice_level', 'N/A')}")
                    st.write(f"**Chế độ ăn:** {recipe.get('diet_type', 'N/A')}")
                    st.write(f"**Phù hợp cho:** {', '.join(recipe.get('suitable_for', []))}")
                    st.write(f"**Nguyên liệu chính:** {', '.join(recipe.get('highlighted_ingredients', []))}")
                    st.write(f"**Nguyên liệu tránh:** {', '.join(recipe.get('avoided_ingredients', []))}")
                    st.markdown("---")

            st.download_button(
                "📥 Tải xuống thực đơn (.json)",
                data=json.dumps(meal_data, ensure_ascii=False, indent=2),
                file_name="meal_plan_result.json",
                mime="application/json"
            )

        except Exception as e:
            st.error("❌ Không thể xử lý JSON từ GPT.")
            st.code(raw_text)
