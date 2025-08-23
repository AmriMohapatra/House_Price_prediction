import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import optuna
from optuna.samplers import TPESampler
import pickle
from data_preprocessing import preprocess_data
from model_training import split_data, calculate_metrics

def grid_search_optimization(X_train, y_train):
    """Hyperparameter tuning using GridSearchCV"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
        greater_is_better=False
    )
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }

def optuna_optimization(X_train, y_train, n_trials=50, timeout=600):
    """Advanced hyperparameter tuning using Optuna"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **params
        )
        
        rmse_scorer = make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
            greater_is_better=False
        )
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=rmse_scorer, n_jobs=-1)
        return -cv_scores.mean()
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    return {
        'best_params': study.best_params,
        'best_score': -study.best_value,
        'n_trials': len(study.trials),
        'study': study
    }

def train_optimized_model(X_train, y_train, best_params):
    """Train model with optimized parameters"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_final_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate final optimized model on all datasets"""
    datasets = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    results = {}
    for name, (X, y) in datasets.items():
        y_pred = model.predict(X)
        results[name] = calculate_metrics(y, y_pred)
    
    return results

def hyperparameter_optimization_pipeline():
    """Complete hyperparameter optimization pipeline"""
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, _ = preprocess_data('data/train.csv')
    
    # Split data
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # GridSearchCV optimization
    print("Running GridSearchCV optimization...")
    grid_results = grid_search_optimization(X_train, y_train)
    print("GridSearchCV completed.")
    print(f"Best GridSearch RMSE: ${grid_results['best_score']:,.0f}")
    print("Best parameters:", grid_results['best_params'])
    
    # Optuna optimization
    print("Running Optuna optimization...")
    optuna_results = optuna_optimization(X_train, y_train)
    print("Optuna optimization completed.")
    print(f"Best Optuna RMSE: ${optuna_results['best_score']:,.0f}")
    print(f"Trials completed: {optuna_results['n_trials']}")
    
    # Choose best method (Optuna typically performs better)
    if optuna_results['best_score'] < grid_results['best_score']:
        best_params = optuna_results['best_params']
        best_method = 'Optuna'
        best_score = optuna_results['best_score']
    else:
        best_params = grid_results['best_params']
        best_method = 'GridSearch'
        best_score = grid_results['best_score']
    
    print(f"Best method: {best_method} with RMSE: ${best_score:,.0f}")
    
    # Train final model
    print("Training final optimized model...")
    final_model = train_optimized_model(X_train, y_train, best_params)
    
    # Evaluate final model
    print("Evaluating final model...")
    final_results = evaluate_final_model(
        final_model, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("Final Model Performance:")
    for dataset, metrics in final_results.items():
        print(f"{dataset.capitalize()}: RMSE=${metrics['rmse']:,.0f}, R²={metrics['r2']:.3f}")
    
    # Save final model
    with open('models/final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("Final model saved to models/final_model.pkl")
    
    return {
        'final_model': final_model,
        'best_params': best_params,
        'best_method': best_method,
        'grid_results': grid_results,
        'optuna_results': optuna_results,
        'final_performance': final_results
    }

if __name__ == "__main__":
    results = hyperparameter_optimization_pipeline()
