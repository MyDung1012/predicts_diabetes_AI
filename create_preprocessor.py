import pandas as pd
import numpy as np
from training.preprocessor import DataPreprocessor
import os

# Đọc dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

print("Đang tạo preprocessor...")

# Khởi tạo preprocessor
preprocessor = DataPreprocessor()

# Fit preprocessor với dữ liệu training
X = df.drop('diabetes', axis=1)
y = df['diabetes']

preprocessor.fit(X, y)

# Tạo thư mục model nếu chưa có
os.makedirs('model', exist_ok=True)

# Lưu preprocessor
preprocessor.save('model/preprocessor.pkl')

print("✅ Đã tạo xong file model/preprocessor.pkl")
print("Preprocessor đã được train với:")
print(f"- Số mẫu: {len(df)}")
print(f"- Các cột: {list(X.columns)}")