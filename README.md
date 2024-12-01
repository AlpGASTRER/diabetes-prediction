# Diabetes Prediction Model

An advanced machine learning system for diabetes prediction using ensemble methods and medical-grade preprocessing.

## Project Overview

This project implements a robust diabetes prediction model using a combination of state-of-the-art machine learning techniques. It features medical-grade data validation, advanced preprocessing, and an ensemble of models optimized for both accuracy and clinical relevance.

## Architecture

The project is structured into three main components:

### 1. Preprocessor (`src/preprocessor.py`)
- Medical-grade data validation
- Age-specific feature normalization
- Clinical feature engineering
- Advanced outlier detection
- Feature importance calculation using multiple methods

### 2. Ensemble Model (`src/ensemble_model.py`)
- Two-stage prediction approach:
  - Stage 1: XGBoost for initial screening
  - Stage 2: LightGBM and CatBoost for refined prediction
- Uncertainty quantification via Monte Carlo dropout
- Dynamic cross-validation based on dataset size
- Demographic subgroup analysis

### 3. Training Pipeline (`train_ensemble.py`)
- Multi-factor stratified cross-validation
- Comprehensive performance metrics
- Demographic fairness analysis
- Detailed logging and visualization

## Key Features

- **Robust Preprocessing**: 
  - Medical threshold-based outlier detection
  - Age-specific feature normalization
  - Clinical feature interactions (BMI-Age risk, cardiovascular score)

- **Advanced Model Architecture**:
  - Ensemble learning with three complementary models
  - Uncertainty quantification
  - Batch processing for large datasets
  - Clinical risk stratification

- **Performance Optimization**:
  - Adaptive parameters based on dataset size
  - Efficient feature importance calculation
  - Parallel processing for large-scale training

## Dependencies

```
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
xgboost>=1.4.0
lightgbm>=3.2.0
catboost>=0.26.0
imblearn>=0.8.0
```

## Usage

### Quick Test
```python
python train_ensemble.py --mode test
```

### Full Training
```python
python train_ensemble.py --mode train
```

## Code Structure

```
diabetes-prediction/
├── src/
│   ├── preprocessor.py      # Data preprocessing and feature engineering
│   ├── ensemble_model.py    # Core model implementation
│   └── utils/              # Utility functions
├── train_ensemble.py        # Main training pipeline
├── requirements.txt         # Project dependencies
└── README.md               # This documentation
```

## Model Parameters

The model automatically adapts its parameters based on dataset size:

### Test Mode (< 10k samples)
- Lighter preprocessing
- Reduced ensemble complexity
- More emphasis on validation

### Production Mode (500k+ samples)
- Full feature engineering
- Comprehensive ensemble training
- Parallel processing optimization

## Future Development

Areas for potential improvement:
1. Integration of temporal medical data
2. Enhanced demographic fairness metrics
3. API deployment infrastructure
4. Real-time prediction capabilities

## Contributing

When contributing to this project:
1. Follow the existing code structure and documentation patterns
2. Add comprehensive docstrings to new functions
3. Update this README for significant changes
4. Include unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.
