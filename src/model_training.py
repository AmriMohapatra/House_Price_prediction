import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import pickle
from data_preprocessing import preprocess_data

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def train_baseline_model(X_train, y_train, X_val, y_val):
    """Train baseline XGBoost model"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_train, y_train)
    
    # Calculate performance
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred)
    
    return model, train_metrics, val_metrics

def perform_cross_validation(model, X_train, y_train, cv_folds=5):
    """Perform cross-validation evaluation"""
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
        greater_is_better=False
    )
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=rmse_scorer, n_jobs=-1)
    
    cv_rmse_scores = -cv_scores
    
    return {
        'scores': cv_rmse_scores,
        'mean': cv_rmse_scores.mean(),
        'std': cv_rmse_scores.std()
    }

def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze feature importance from trained model"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    total_importance = importance_df['importance'].sum()
    importance_df['cumulative_percentage'] = (importance_df['cumulative_importance'] / total_importance) * 100
    
    return importance_df.head(top_n)

def save_model(model, filepath):
    """Save trained model to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load trained model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def train_and_evaluate():
    """Complete training and evaluation pipeline"""
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, _ = preprocess_data('data/train.csv')
    
    # Split data
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train baseline model
    print("Training baseline model...")
    model, train_metrics, val_metrics = train_baseline_model(X_train, y_train, X_val, y_val)
    
    print("Baseline Model Performance:")
    print(f"Train RMSE: ${train_metrics['rmse']:,.0f}, R²: {train_metrics['r2']:.3f}")
    print(f"Val RMSE: ${val_metrics['rmse']:,.0f}, R²: {val_metrics['r2']:.3f}")
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_results = perform_cross_validation(model, X_train, y_train)
    print(f"CV RMSE: ${cv_results['mean']:,.0f} ± ${cv_results['std']:,.0f}")
    
    # Feature importance
    print("Analyzing feature importance...")
    importance_df = analyze_feature_importance(model, X.columns.tolist())
    print("Top 10 features:")
    print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
    
    # Save model
    save_model(model, 'models/baseline_model.pkl')
    print("Model saved to models/baseline_model.pkl")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'cv_results': cv_results,
        'importance': importance_df,
        'data_splits': (X_train, X_val, X_test, y_train, y_val, y_test)
    }

if __name__ == "__main__":
    results = train_and_evaluate()
