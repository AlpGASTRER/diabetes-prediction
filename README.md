# Diabetes Prediction with Uncertainty Estimation

A robust machine learning pipeline for diabetes prediction with uncertainty estimation, using a two-stage ensemble approach and advanced preprocessing techniques. The model is trained on the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from the BRFSS 2015 survey.

## Dataset

The dataset contains health indicators that may be used to predict diabetes, collected from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 health survey. It includes:

- **Size**: 253,680 survey responses
- **Features**: 21 health indicators
- **Target**: Binary classification (0: No diabetes, 1: Diabetes/Prediabetes)
- **Class Distribution**: Imbalanced dataset (~88% non-diabetic, ~12% diabetic)

### Key Features
- Demographic: Age, Education, Income
- Health Conditions: High Blood Pressure, High Cholesterol, Heart Disease
- Lifestyle: Physical Activity, Smoking, Alcohol Consumption
- Health Metrics: BMI, Mental Health, Physical Health
- Healthcare Access: Insurance, Doctor Visits

## Features

### Preprocessing Pipeline
- Advanced data cleaning and feature engineering
- Medical risk score calculation
- Outlier detection using Local Outlier Factor (LOF)
- Feature importance analysis using Random Forest
- ADASYN for handling class imbalance
- Robust scaling with proper dtype handling

### Model Architecture
- Two-stage ensemble prediction
- Multiple calibrated models for robustness
- Uncertainty estimation via:
  - Monte Carlo dropout
  - Model variance
  - Ensemble disagreement
- Comprehensive uncertainty calibration

### Visualization Tools
- Interactive dashboards using Plotly
- Feature importance plots
- Uncertainty distribution analysis
- Model calibration curves
- Performance metrics visualization

## Reproducibility

The model uses a fixed random state (42) across all stochastic components to ensure reproducible results:
- All random number generators (numpy, Python's random, PyTorch)
- Model initialization (XGBoost, LightGBM, CatBoost)
- Data splitting and sampling operations
- ADASYN synthetic sampling

To maintain reproducibility:
1. Do not modify the random state settings in the code
2. Use the same Python environment and package versions
3. Process the data in the same order
4. Run on the same hardware architecture when possible

## Installation

1. Create and activate conda environment:
```bash
conda create -n diabetes_pred python=3.11
conda activate diabetes_pred
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Visit [Kaggle Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Download `diabetes_binary_health_indicators_BRFSS2015.csv`
- Place it directly in the project root directory (not in a subdirectory)

## Usage

Make sure to activate the conda environment before running any commands:
```bash
conda activate diabetes_pred
```

### Training
```bash
python src/main.py --mode train
```

### Prediction
```bash
python src/main.py --mode predict
```

### Testing

The project includes two types of tests:

#### Quick Integration Test
Run a quick smoke test to verify the basic functionality:
```bash
python src/main.py --mode test
```

Or using the full conda path:
```bash
C:\Users\ahmed\.conda\envs\diabetes_pred\python.exe src/main.py --mode test
```

This will:
- Train the model on a small subset (10%) of the dataset
- Test prediction functionality
- Provide immediate pass/fail feedback

#### Development Tests
Run the full test suite for thorough testing during development:
```bash
pytest tests/test_pipeline.py -v
```

## Project Structure
```
diabetes-prediction/
├── data/                # Dataset directory
├── src/
│   ├── main.py         # Main pipeline
│   ├── preprocessor.py  # Data preprocessing
│   ├── ensemble_model.py# Model implementation
│   └── visualization.py # Visualization tools
├── tests/              # Test suite
├── models/            # Saved models
├── plots/             # Generated plots
├── requirements.txt   # Dependencies
├── UPDATES.md        # Changelog
└── README.md         # Documentation
```

## Performance Metrics

The model provides comprehensive evaluation metrics:
- ROC-AUC Score
- PR-AUC Score (for imbalanced data)
- Calibration metrics
- Uncertainty correlation
- Confusion matrix metrics

## Contributing

Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from BRFSS 2015
- Scikit-learn, XGBoost, and LightGBM for model implementations
- Imbalanced-learn for ADASYN implementation
- Plotly for interactive visualizations
