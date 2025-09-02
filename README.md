# Heart Failure Prediction Machine Learning Pipeline

A comprehensive and modular machine learning pipeline for predicting heart failure events using clinical data.

## Features

- **Data Validation**: Automatic data quality checks and missing value detection
- **Class Imbalance Handling**: Support for SMOTE and undersampling techniques
- **Multiple Models**: Random Forest, XGBoost, and Stacking Ensemble
- **Hyperparameter Tuning**: Automated GridSearchCV for optimal parameters
- **Model Interpretation**: SHAP explanations for feature importance
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Model Persistence**: Save and load trained models

## Dataset

The pipeline uses the Heart Failure Clinical Records dataset with the following features:

- `age`: Age of the patient (years)
- `anaemia`: Decrease of red blood cells or hemoglobin (boolean)
- `creatinine_phosphokinase`: Level of the CPK enzyme in the blood (mcg/L)
- `diabetes`: If the patient has diabetes (boolean)
- `ejection_fraction`: Percentage of blood leaving the heart at each contraction (percentage)
- `high_blood_pressure`: If the patient has hypertension (boolean)
- `platelets`: Platelets in the blood (kiloplatelets/mL)
- `serum_creatinine`: Level of serum creatinine in the blood (mg/dL)
- `serum_sodium`: Level of serum sodium in the blood (mEq/L)
- `sex`: Woman or man (binary)
- `smoking`: If the patient smokes (boolean)
- `time`: Follow-up period (days)
- `DEATH_EVENT`: If the patient died during the follow-up period (target variable)

## Installation

```bash
# Install required packages
pip install scikit-learn imbalanced-learn xgboost shap matplotlib seaborn joblib pandas numpy
```

## Usage

### Basic Usage

```python
from heart_failure_ml_pipeline import HeartFailurePipeline

# Initialize and run the complete pipeline
pipeline = HeartFailurePipeline()
pipeline.run_pipeline()
```

### Custom Configuration

```python
# Custom pipeline configuration
pipeline = HeartFailurePipeline(
    data_path='your_dataset.csv',  # Custom dataset path
    random_state=42                # Random seed for reproducibility
)

# Run specific components
pipeline.load_and_validate_data()
pipeline.preprocess_data(test_size=0.2)
X_res, y_res = pipeline.handle_class_imbalance(method='smote')
pipeline.train_models(X_res, y_res)
results = pipeline.evaluate_models()
pipeline.explain_model('xgb')
pipeline.save_model('custom_model.pkl')
```

## Pipeline Components

### 1. Data Loading and Validation
- Loads CSV dataset
- Validates target variable presence
- Checks for missing values
- Reports class distribution

### 2. Data Preprocessing
- Feature-target separation
- Train-test split with stratification
- Standard scaling of features

### 3. Class Imbalance Handling
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Undersampling**: Random under-sampling of majority class
- **None**: Use original imbalanced data

### 4. Model Training
- **Random Forest**: Ensemble of decision trees with balanced class weights
- **XGBoost**: Gradient boosting with scale_pos_weight for imbalance
- **Stacking Ensemble**: Combines base models with logistic regression meta-model
- **Hyperparameter Tuning**: Grid search for optimal parameters

### 5. Model Evaluation
- AUC-ROC scores
- Classification reports (precision, recall, f1-score)
- Confusion matrices
- Cross-validation scores

### 6. Model Interpretation
- **SHAP Values**: Explain individual predictions
- Feature importance plots
- Force plots for specific predictions

### 7. Visualization
- Model comparison charts
- ROC curves
- Precision-recall curves
- SHAP summary plots

## Output Files

- `shap_feature_importance_{model}.png`: SHAP feature importance plots
- `shap_force_plot_{model}.png`: SHAP force plots
- `model_comparison.png`: Model performance comparison
- `roc_curve_best_model.png`: ROC curve for best model
- `heart_failure_model.pkl`: Saved model (configurable)

## Customization

### Adding New Models

```python
# Add new models to the training process
new_models = [
    ('logreg', LogisticRegression(class_weight='balanced')),
    ('gbm', GradientBoostingClassifier())
]
```

### Custom Hyperparameter Tuning

```python
# Modify the parameter grid
custom_param_grid = {
    'final_estimator__C': [0.01, 0.1, 1, 10, 100],
    'final_estimator__solver': ['liblinear', 'newton-cg', 'lbfgs']
}
```

### Custom Evaluation Metrics

```python
# Add custom evaluation metrics
from sklearn.metrics import f1_score, precision_score, recall_score

def custom_evaluation(y_true, y_pred, y_proba):
    return {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_proba)
    }
```

## Performance Tips

1. **Data Quality**: Ensure clean data with proper feature engineering
2. **Class Imbalance**: Experiment with different imbalance handling techniques
3. **Hyperparameter Tuning**: Adjust grid search parameters based on dataset size
4. **Feature Selection**: Consider feature importance for dimensionality reduction
5. **Cross-Validation**: Use appropriate CV strategy for small datasets

## Troubleshooting

### Common Issues

1. **Missing Data**: Pipeline includes basic missing value detection
2. **Class Imbalance**: Automatic handling with configurable methods
3. **Memory Issues**: Reduce grid search complexity for large datasets
4. **SHAP Errors**: Ensure tree-based models for SHAP explanations

### Error Handling

The pipeline includes comprehensive error handling with informative messages for:
- File not found errors
- Data validation failures
- Model training errors
- Evaluation metric computation errors

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please check the troubleshooting section or create an issue in the repository.
