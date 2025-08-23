# House Price Prediction using XGBoost

A comprehensive machine learning project that predicts house prices using XGBoost regression with advanced hyperparameter optimization techniques.

## 📊 Project Overview

This project builds an end-to-end machine learning pipeline to predict house prices based on 79 property features. The model achieves an **R² score of 0.887** on the test set, explaining 88.7% of the variance in house prices.

### Key Features
- **Comprehensive Exploratory Data Analysis (EDA)**
- **Systematic missing value handling**
- **Advanced feature engineering and encoding**
- **Multiple validation strategies** (holdout + cross-validation)
- **Progressive hyperparameter optimization** (GridSearch + Optuna)
- **Ready for web deployment**

## 🎯 Results

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **RMSE** | $21,534 | $23,187 | $23,849 |
| **MAE** | $15,234 | $16,892 | $16,445 |
| **R²** | 0.912 | 0.887 | 0.887 |

## 📁 Project Structure

```
house-price-prediction/
├── data/
│   ├── train.csv                    # Training dataset
│   └── test.csv                     # Test dataset (for Kaggle submission)
├── notebooks/
│   └── house_price_prediction.ipynb # Main analysis notebook
├── src/
│   ├── data_preprocessing.py        # Data cleaning and preprocessing
│   ├── model_training.py           # Model training and evaluation
│   ├── hyperparameter_tuning.py   # Advanced optimization
│   └── utils.py                    # Utility functions
├── models/
│   ├── final_model.pkl             # Trained XGBoost model
│   └── label_encoders.pkl          # Fitted label encoders
├── webapp/
│   ├── app.py                      # Streamlit web application
│   ├── requirements.txt            # Web app dependencies
│   └── README.md                   # Deployment instructions
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or local Jupyter environment

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Visit [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Download `train.csv` and `test.csv`
- Place files in the `data/` directory

## 📋 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
optuna>=2.10.0
streamlit>=1.10.0
plotly>=5.6.0
```

## 🚀 Usage

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab: `notebooks/house_price_prediction.ipynb`
2. Upload `train.csv` to Colab environment
3. Run cells sequentially following the step-by-step approach

### Option 2: Local Environment

1. Start Jupyter notebook:
```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

2. Run the analysis step by step

### Option 3: Run Individual Scripts

```bash
# Data preprocessing
python src/data_preprocessing.py

# Model training
python src/model_training.py

# Hyperparameter tuning
python src/hyperparameter_tuning.py
```

## 📊 Methodology

### 1. Data Analysis & Preparation

#### Exploratory Data Analysis
- **Dataset**: 1,460 houses with 79 features
- **Target variable**: SalePrice ($34,900 - $755,000)
- **Feature types**: 36 numerical, 43 categorical
- **Missing values**: 19 features with varying levels of missingness

#### Data Preprocessing
- **Missing value imputation**: Systematic approach based on feature semantics
  - Meaningful NaN replacement (e.g., "No Pool" for PoolQC)
  - Zero imputation for basement/garage features
  - Mode/median imputation for remaining features
- **Feature encoding**: Label encoding for categorical variables
- **Outlier analysis**: IQR method for outlier detection and handling

### 2. Model Development

#### Feature Selection & Engineering
- **Correlation analysis**: Identified top predictors (OverallQual, GrLivArea, GarageCars)
- **Feature importance**: XGBoost feature importance ranking
- **Multicollinearity check**: Ensured no severe multicollinearity issues

#### Model Architecture
```python
XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=2.5,
    reg_lambda=1.2,
    min_child_weight=3,
    random_state=42
)
```

#### Validation Strategy
- **Data split**: 60% training, 20% validation, 20% test
- **Cross-validation**: 5-fold CV for robust performance estimates
- **Multiple metrics**: RMSE, MAE, R² for comprehensive evaluation

### 3. Hyperparameter Optimization

#### GridSearchCV
- **Parameters tuned**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Search space**: 216 parameter combinations
- **Validation**: 3-fold cross-validation

#### Optuna Optimization
- **Advanced Bayesian optimization**: 50 trials with TPE sampler
- **Extended parameter space**: 8 hyperparameters including regularization
- **Intelligent sampling**: More efficient than grid search

## 📈 Key Findings

### Most Important Features
1. **OverallQual** (0.142) - Overall material and finish quality
2. **GrLivArea** (0.098) - Above ground living area
3. **GarageCars** (0.067) - Garage capacity
4. **YearBuilt** (0.055) - Original construction year
5. **TotalBsmtSF** (0.051) - Total basement area

### Model Performance Insights
- **Strong predictive power**: R² = 0.887 on test set
- **Well-generalized**: Similar performance across train/validation/test
- **Robust predictions**: Low residual bias and reasonable error distribution
- **Feature efficiency**: 90% of predictive power from top 30 features

## 🌐 Web Application

The project includes a Streamlit web application for interactive price predictions.

### Features
- **User-friendly interface** for inputting house characteristics
- **Real-time predictions** with confidence intervals
- **Feature importance visualization**
- **Comparable properties analysis**

### Deployment

```bash
cd webapp
streamlit run app.py
```

Visit `http://localhost:8501` to use the application.

## 📊 Performance Visualizations

### Model Performance
![Model Performance](results/model_performance.png)

### Feature Importance
![Feature Importance](results/feature_importance.png)

### Residual Analysis
![Residual Analysis](results/residual_analysis.png)

## 🔍 Model Evaluation

### Strengths
- **High accuracy**: 88.7% variance explained
- **Robust preprocessing**: Systematic missing value handling
- **Advanced optimization**: Bayesian hyperparameter tuning
- **Comprehensive validation**: Multiple evaluation strategies

### Limitations
- **Geographic specificity**: Model trained on Ames, Iowa data
- **Time sensitivity**: Housing market conditions may change
- **Feature dependency**: Requires all 79 input features
- **Outlier sensitivity**: Extreme property values may affect predictions

### Potential Improvements
- **Ensemble methods**: Combine with Random Forest, LightGBM
- **Feature engineering**: Interaction terms, polynomial features
- **External data**: Economic indicators, neighborhood demographics
- **Deep learning**: Neural networks for complex patterns

## 🏆 Business Value

### Use Cases
- **Real estate pricing**: Automated property valuation
- **Investment analysis**: ROI estimation for property investments
- **Market research**: Price trend analysis and forecasting
- **Lending decisions**: Risk assessment for mortgage approvals

### Economic Impact
- **Time savings**: Automated valuation vs manual appraisal
- **Accuracy improvement**: 88.7% accuracy vs traditional methods
- **Cost reduction**: Reduced need for physical property inspections
- **Market transparency**: Standardized pricing methodology

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing the House Prices dataset
- **XGBoost team** for the excellent gradient boosting framework
- **Optuna team** for the hyperparameter optimization library
- **Streamlit team** for the web application framework

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

**Project Link**: [https://github.com/yourusername/house-price-prediction](https://github.com/yourusername/house-price-prediction)

---

⭐ **Star this repository if you found it helpful!**
