# Experiments Module

Supervised learning experiments framework for antibiotic resistance prediction tasks.

## Overview

This module provides a comprehensive framework for running supervised learning experiments with:
- Six algorithms (Random Forest, Logistic Regression, SVM, GBM, KNN, Naive Bayes)
- Nested cross-validation for hyperparameter tuning
- MLflow experiment tracking
- Comprehensive evaluation metrics
- Model persistence and reproducibility

## Experiments

### 1. Binary ESBL Classification (`esbl_classifier.py`)

Predicts ESBL status (positive/negative) from resistance profile and metadata.

**Features Used**:
- Binary resistance indicators per antibiotic
- Antibiogram fingerprints
- Aggregate resistance metrics
- Encoded metadata

**Metrics**:
- AUROC, AUPRC
- Accuracy, Precision, Recall, F1
- Brier score (calibration)
- Sensitivity at 95% specificity
- Confusion matrix

**Usage**:
```python
from experiments import ESBLClassifierExperiment

exp = ESBLClassifierExperiment()
X_train, y_train, features = exp.prepare_data(train_df)
X_test, y_test, _ = exp.prepare_data(test_df)

results = exp.run_all_algorithms(X_train, y_train, X_test, y_test)
```

### 2. Species Classification (`species_classifier.py`)

Predicts bacterial species from antibiogram profile.

**Features Used**:
- Antibiogram vectors
- Binary resistance patterns

**Metrics**:
- Accuracy
- Macro/Micro F1
- Confusion matrix (13 classes)

### 3. MAR Index Regression (`mar_regression.py`)

Predicts Multiple Antibiotic Resistance index.

**Algorithms**: 5 regressors (RF, Ridge, GBM, SVR, KNN)

**Metrics**:
- RMSE, MAE, R²
- Residual diagnostics

### 4. Multi-label R/S Prediction (`multilabel_prediction.py`)

Predicts R/S status for multiple antibiotics simultaneously.

**Approaches**:
- Binary relevance
- Classifier chains

**Metrics**:
- Hamming loss
- Subset accuracy
- Per-antibiotic AUROC/AUPRC

## Base Framework (`base_experiment.py`)

The `BaseExperiment` class provides:

### Nested Cross-Validation
- Outer CV: Performance estimation (5 folds)
- Inner CV: Hyperparameter tuning (3 folds)
- Stratified splits for classification

### Six Algorithms

1. **Random Forest**
   - Parameters: n_estimators, max_depth, min_samples_split
   - No scaling required

2. **Logistic Regression**
   - Parameters: C, penalty, solver
   - Requires scaling

3. **SVM**
   - Parameters: C, kernel, gamma
   - Requires scaling

4. **Gradient Boosting**
   - Parameters: n_estimators, learning_rate, max_depth
   - No scaling required

5. **KNN**
   - Parameters: n_neighbors, weights, metric
   - Requires scaling

6. **Naive Bayes**
   - Parameters: var_smoothing
   - No scaling required

### MLflow Integration

All experiments automatically logged to MLflow:
- Hyperparameters
- Evaluation metrics
- Model artifacts
- Confusion matrices

View results:
```bash
mlflow ui
```

## Running Experiments

### Quick Start

```bash
# Run all experiments
python phase2_supervised_learning.py
```

This will:
1. Load preprocessed train/test data
2. Run ESBL classification (6 algorithms)
3. Run species classification (6 algorithms)
4. Run MAR regression (5 algorithms)
5. Generate summary report
6. Save all model artifacts

### Programmatic Usage

```python
from experiments import ESBLClassifierExperiment

# Create experiment
exp = ESBLClassifierExperiment(output_dir='models')

# Prepare data
X_train, y_train, features = exp.prepare_data(train_df)
X_test, y_test, _ = exp.prepare_data(test_df)

# Run all algorithms with nested CV
results = exp.run_all_algorithms(
    X_train, y_train, X_test, y_test,
    use_nested_cv=True
)

# Save results
exp.save_results()

# Access best models
best_model = exp.best_models['random_forest']['model']
scaler = exp.best_models['random_forest']['scaler']
```

### Custom Algorithm Training

```python
# Train specific algorithm
model, metrics, scaler = exp.train_final_model(
    X_train, y_train, X_test, y_test,
    algorithm_name='random_forest',
    params={'n_estimators': 200, 'max_depth': 20}
)
```

## Output Structure

```
models/
├── esbl_classifier_random_forest_model.pkl
├── esbl_classifier_logistic_regression_model.pkl
├── esbl_classifier_svm_model.pkl
├── esbl_classifier_gradient_boosting_model.pkl
├── esbl_classifier_knn_model.pkl
├── esbl_classifier_naive_bayes_model.pkl
├── esbl_classifier_results.json
├── species_classifier_*.pkl
├── species_classifier_results.json
├── mar_regression_*.pkl
└── mar_regression_results.json

mlruns/
└── [experiment_id]/
    └── [run_id]/
        ├── metrics/
        ├── params/
        ├── artifacts/
        └── meta.yaml
```

## Evaluation Metrics

### Classification Tasks

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve
- **Brier Score**: Calibration metric (lower is better)

### Regression Tasks

- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination

## Model Selection

Results are ranked by:
1. **Classification**: AUROC or F1 score
2. **Regression**: R² or RMSE

Consider Pareto front:
- Accuracy vs. interpretability
- Performance vs. inference cost
- Calibration quality

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
mlflow >= 2.0
xgboost >= 2.0
lightgbm >= 4.0
shap >= 0.40
matplotlib >= 3.5
seaborn >= 0.12
```

## Next Steps

1. **Interpretability Analysis**
   - SHAP values for feature importance
   - Partial dependence plots
   - Feature interaction analysis

2. **Model Calibration**
   - Calibration curves
   - Platt scaling or isotonic regression
   - Decision curve analysis

3. **Ensemble Methods**
   - Voting classifiers
   - Stacking
   - Blending top models

4. **Deployment**
   - Model serving with MLflow
   - API endpoint creation
   - Production monitoring

## Troubleshooting

### Issue: Class imbalance warnings
**Solution**: Use `class_weight='balanced'` in algorithms that support it

### Issue: Memory errors during GridSearchCV
**Solution**: Reduce `n_jobs` or parameter grid size

### Issue: MLflow UI not starting
**Solution**: Check port 5000 availability or specify different port:
```bash
mlflow ui --port 5001
```

## References

- Nested CV: [Cawley & Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html)
- ESBL prediction: Clinical breakpoints from CLSI/EUCAST
- Model selection: Pareto optimality in ML [Zunic et al., 2020]
