import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def load_data(file_path):
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    return missing_df[missing_df['Missing_Count'] > 0]

def handle_missing_values(df):
    """Handle missing values with systematic approach"""
    df_processed = df.copy()
    
    # Features where NaN means "None" or "Not Applicable"
    none_features = {
        'PoolQC': 'No Pool',
        'MiscFeature': 'None', 
        'Alley': 'No Alley',
        'Fence': 'No Fence',
        'FireplaceQu': 'No Fireplace',
        'GarageType': 'No Garage',
        'GarageFinish': 'No Garage',
        'GarageQual': 'No Garage',
        'GarageCond': 'No Garage',
        'BsmtQual': 'No Basement',
        'BsmtCond': 'No Basement',
        'BsmtExposure': 'No Basement',
        'BsmtFinType1': 'No Basement',
        'BsmtFinType2': 'No Basement',
        'MasVnrType': 'None'
    }
    
    # Fill with meaningful values
    for feature, fill_value in none_features.items():
        if feature in df_processed.columns:
            df_processed[feature].fillna(fill_value, inplace=True)
    
    # Numerical features where 0 makes sense
    zero_features = {
        'GarageYrBlt': 0,
        'GarageArea': 0,
        'GarageCars': 0,
        'BsmtFinSF1': 0,
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 0,
        'TotalBsmtSF': 0,
        'BsmtFullBath': 0,
        'BsmtHalfBath': 0,
        'MasVnrArea': 0
    }
    
    for feature, fill_value in zero_features.items():
        if feature in df_processed.columns:
            df_processed[feature].fillna(fill_value, inplace=True)
    
    # Fill categorical features with mode
    categorical_features = df_processed.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        if df_processed[feature].isnull().sum() > 0:
            mode_value = df_processed[feature].mode()[0] if len(df_processed[feature].mode()) > 0 else 'Unknown'
            df_processed[feature].fillna(mode_value, inplace=True)
    
    # Fill numerical features with median
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns
    for feature in numerical_features:
        if df_processed[feature].isnull().sum() > 0:
            median_value = df_processed[feature].median()
            df_processed[feature].fillna(median_value, inplace=True)
    
    return df_processed

def encode_categorical_features(X):
    """Apply label encoding to categorical features"""
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
    
    return X_encoded, label_encoders

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def preprocess_data(file_path, save_encoders=True):
    """Complete preprocessing pipeline"""
    # Load data
    df = load_data(file_path)
    
    # Handle missing values
    df_processed = handle_missing_values(df)
    
    # Separate features and target
    X = df_processed.drop(['Id', 'SalePrice'], axis=1)
    y = df_processed['SalePrice']
    
    # Encode categorical features
    X_encoded, label_encoders = encode_categorical_features(X)
    
    # Save label encoders if requested
    if save_encoders:
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
    
    return X_encoded, y, label_encoders

if __name__ == "__main__":
    X, y, encoders = preprocess_data('data/train.csv')
    print(f"Preprocessing completed. Shape: {X.shape}")
    print(f"Missing values remaining: {X.isnull().sum().sum()}")
