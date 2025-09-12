import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.decomposition import PCA
import pickle

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.is_fitted = False

    def fit_transform(self, X):
        """Fit và transform dữ liệu training"""
        # Log transform
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        X = log_transformer.transform(X)
        X = pd.DataFrame(X)
        X.fillna(0, inplace=True)

        # Scale dữ liệu
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)

        # PCA với 99% variance
        self.pca = PCA(n_components=0.99)
        X = self.pca.fit_transform(X)
        
        self.is_fitted = True
        return X

    def transform(self, X):
        """Transform dữ liệu mới (dùng cho prediction)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor chưa được fit. Hãy gọi fit_transform trước.")

        # Log transform
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        X = log_transformer.transform(X)
        X = pd.DataFrame(X)
        X.fillna(0, inplace=True)

        # Scale dữ liệu
        X = self.scaler.transform(X)

        # PCA transform
        X = self.pca.transform(X)
        
        return X

    def save(self, filepath):
        """Lưu preprocessor để sử dụng sau"""
        if not self.is_fitted:
            raise ValueError("Preprocessor chưa được fit. Không thể lưu.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load preprocessor đã lưu"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_feature_names(self):
        """Trả về tên các feature sau khi xử lý"""
        if not self.is_fitted:
            raise ValueError("Preprocessor chưa được fit.")
            
        return [f"PC{i+1}" for i in range(self.pca.n_components_)]

if __name__ == "__main__":
    # Test thử preprocessor
    import pandas as pd
    
    # Load dữ liệu test
    data = pd.read_csv('diabetes_prediction_dataset.csv')
    data.dropna(inplace=True)
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)
    data.fillna(0, inplace=True)
    
    X = data.drop(['diabetes'], axis=1)
    y = data['diabetes']
    
    # Thử nghiệm preprocessor
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    print(f"Số feature ban đầu: {X.shape[1]}")
    print(f"Số feature sau PCA: {X_processed.shape[1]}")
    print(f"Tỉ lệ variance giữ lại: {sum(preprocessor.pca.explained_variance_ratio_):.4f}")
    
    # Lưu preprocessor
    preprocessor.save('preprocessor.pkl')