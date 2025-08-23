import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def load_model_and_encoders(model_path='models/final_model.pkl', encoders_path='models/label_encoders.pkl'):
    """Load trained model and label encoders"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    return model, encoders

def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 8)):
    """Plot feature importance from trained model"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['importance'], 
             color='lightblue', edgecolor='navy', alpha=0.7)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual", figsize=(8, 6)):
    """Plot predicted vs actual values"""
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(title)
    
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, figsize=(12, 5)):
    """Plot residual analysis"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=figsize)
    
    # Residual scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Residual histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def create_performance_summary(results_dict):
    """Create performance summary DataFrame"""
    summary_data = []
    for dataset_name, metrics in results_dict.items():
        summary_data.append({
            'Dataset': dataset_name.capitalize(),
            'RMSE': f"${metrics['rmse']:,.0f}",
            'MAE': f"${metrics['mae']:,.0f}",
            'R²': f"{metrics['r2']:.3f}",
            'MAPE': f"{metrics.get('mape', 0):.2f}%"
        })
    
    return pd.DataFrame(summary_data)

def prepare_input_data(input_dict, encoders):
    """Prepare input data for prediction using saved encoders"""
    input_df = pd.DataFrame([input_dict])
    
    # Apply label encoders to categorical features
    for column, encoder in encoders.items():
        if column in input_df.columns:
            try:
                input_df[column] = encoder.transform(input_df[column].astype(str))
            except ValueError:
                # Handle unseen categories by using most frequent category
                input_df[column] = encoder.transform([encoder.classes_[0]])[0]
    
    return input_df

def make_prediction(model, input_data):
    """Make price prediction with confidence interval"""
    prediction = model.predict(input_data)[0]
    
    # Estimate confidence interval using standard error approximation
    # This is a simplified approach for demonstration
    se = prediction * 0.1  # Approximate 10% standard error
    confidence_interval = (prediction - 1.96 * se, prediction + 1.96 * se)
    
    return {
        'prediction': prediction,
        'lower_bound': max(0, confidence_interval[0]),
        'upper_bound': confidence_interval[1]
    }

def analyze_prediction_error(y_true, y_pred, percentiles=[25, 50, 75, 90, 95]):
    """Analyze prediction errors at different percentiles"""
    absolute_errors = np.abs(y_true - y_pred)
    relative_errors = absolute_errors / y_true * 100
    
    error_analysis = {}
    for p in percentiles:
        error_analysis[f'{p}th_percentile_abs_error'] = np.percentile(absolute_errors, p)
        error_analysis[f'{p}th_percentile_rel_error'] = np.percentile(relative_errors, p)
    
    return error_analysis

def save_results(results, filename):
    """Save results to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    """Load results from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def print_model_summary(model, feature_names):
    """Print comprehensive model summary"""
    print("Model Summary")
    print("=" * 50)
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of estimators: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"Learning rate: {model.learning_rate}")
    
    # Feature importance summary
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 most important features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nFeatures with zero importance: {sum(model.feature_importances_ == 0)}")
