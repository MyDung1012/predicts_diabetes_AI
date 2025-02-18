import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('diabetic_data.csv')

# Hiển thị thông tin ban đầu
print("5 dòng đầu tiên của dữ liệu:")
print(df.head())
print("\nThông tin dữ liệu:")
print(df.info())
print("\nThống kê mô tả:")
print(df.describe())
print("\nSố lượng giá trị null ban đầu:")
print(df.isnull().sum())
print("\nDanh sách các cột:")
print(df.keys())

# 2. Thay thế giá trị '?' bằng NaN
df = df.replace('?', np.nan)
print("\nSố lượng giá trị null trên mỗi cột sau khi thay thế:")
print(df.isnull().sum())

# 3. Loại bỏ các cột có tỷ lệ giá trị null lớn hơn 50%
df = df.loc[:, df.isnull().mean() <= 0.5]
print("\nCác cột sau khi loại bỏ cột có giá trị null > 50%:")
print(df.columns)

# 4. Xác định biến mục tiêu
# Giả sử cột 'Outcome' là biến mục tiêu. Nếu không tồn tại, sử dụng cột cuối cùng làm biến mục tiêu.
target_column = 'Outcome'
if target_column in df.columns:
    X = df.drop(columns=target_column)
    y = df[target_column]
else:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

# 5. Tính mutual information cho từng đặc trưng với biến mục tiêu
# Sử dụng X.fillna(0) để thay thế giá trị thiếu khi tính toán
mi_scores = mutual_info_classif(X.fillna(0), y, random_state=0)
print("\nMutual Information Scores cho từng đặc trưng:")
for col, score in zip(X.columns, mi_scores):
    print(f"{col}: {score}")

# 6. Chọn các đặc trưng có mutual information >= ngưỡng (ví dụ: threshold = 0.4)
threshold = 0.4
selected_cols = X.columns[mi_scores >= threshold]
print(f"\nCác cột được chọn với mutual information >= {threshold}:")
print(selected_cols)

# 7. Loại bỏ các hàng có dữ liệu thiếu trong các cột đã chọn
X_selected = X[selected_cols].dropna()
y_selected = y.loc[X_selected.index]

# 8. Cập nhật DataFrame với các đặc trưng đã chọn và biến mục tiêu
df_selected = pd.concat([X_selected, y_selected], axis=1)
print("\nDataFrame sau khi xử lý và chọn đặc trưng:")
print(df_selected.head())
print("\nDanh sách các cột cuối cùng:")
print(df_selected.columns)
