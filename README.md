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

## Dataset Context

This model is trained on the Behavioral Risk Factor Surveillance System (BRFSS) 2015 health survey data, which has several important characteristics:

- **National Representativeness**: The dataset reflects the actual U.S. population distribution of diabetes, with approximately 13.7% positive cases. This matches CDC statistics showing that about 13% of U.S. adults have diabetes.

- **Demographic Coverage**: 
  - Covers all 50 U.S. states and territories
  - Includes diverse age groups, ethnicities, and socioeconomic backgrounds
  - Representative of both urban and rural populations

- **Class Distribution**:
  - Non-diabetic: 86.3% (matches U.S. population statistics)
  - Diabetic/Pre-diabetic: 13.7% (aligns with CDC reports)
  - This natural distribution is maintained in our test set to ensure real-world performance metrics

## Features

### Preprocessing Pipeline
- Advanced data cleaning and feature engineering
- Medical risk score calculation
- Outlier detection using Local Outlier Factor (LOF)
- Feature importance analysis using Random Forest
- ADASYN for handling class imbalance
- Robust scaling with proper dtype handling

### Model Architecture
- Two-stage ensemble prediction:
  - Screening stage (threshold: 0.12)
  - Confirmation stage (threshold: 0.15)
- Multiple base models:
  - LightGBM (500 estimators)
  - XGBoost (500 estimators)
  - CatBoost (500 estimators)
- Advanced parameter tuning:
  - Low learning rate (0.03) for better generalization
  - Balanced tree depth (8) for optimal complexity
  - Adjusted class weights for imbalance handling
- Comprehensive uncertainty estimation:
  - Monte Carlo dropout (35 iterations)
  - Model variance analysis
  - Ensemble disagreement metrics
  - 25 calibration models per stage

### Sampling Strategy
- SMOTE for minority class oversampling:
  - Balanced sampling ratio (0.5)
  - Robust neighbor selection (k=5)
  - Stage-specific sampling parameters
- Feature-aware synthetic sample generation
- Consistent sampling across model stages

### Visualization Tools
- Interactive dashboards using Plotly
- Feature importance plots
- Uncertainty distribution analysis
- Model calibration curves
- Performance metrics visualization

## Performance Metrics

### Medical Perspective

Our model's performance reflects real-world clinical scenarios and compares favorably to existing screening methods:

- **Screening Stage** (Initial Assessment):
  - Catches 84% of diabetes cases (84% sensitivity/recall)
  - For every 100 positive screenings, 29 are actual diabetes cases (29% precision)
  - Balanced performance across risk groups (76% balanced accuracy)
  - These metrics are highly realistic given:
    - Similar to clinical questionnaire sensitivity (75-85%)
    - Precision aligns with real prevalence rates in high-risk populations
    - Matches performance of standard diabetes risk scores used in practice

- **Confirmation Stage** (Detailed Assessment):
  - Identifies 87% of true diabetes cases (87% sensitivity/recall)
  - Confirmation rate of 28% (28% precision)
  - Consistent performance across patient subgroups (75% balanced accuracy)
  - Performance is realistic considering:
    - Comparable to two-step screening protocols in clinical practice
    - Precision reflects true population prevalence (13.7%)
    - Higher recall than many standard risk assessment tools

**Clinical Interpretation**:
- Performance metrics are based on real U.S. population distribution from BRFSS
- High sensitivity prioritizes catching potential diabetes cases, matching clinical preference
- Precision appears low but is actually strong given the true 13.7% disease prevalence
- Risk stratification aligns with standard medical risk factors
- These results are particularly valuable because they're achieved on natural, unbalanced data, unlike many published models that report inflated metrics on artificially balanced datasets

### Machine Learning Perspective

Technical performance metrics on real-world distribution:

- **Screening Stage**:
  - Accuracy: 0.702
  - Balanced Accuracy: 0.759
  - Precision: 0.294
  - Recall: 0.837
  - F1 Score: 0.435

- **Confirmation Stage**:
  - Accuracy: 0.673
  - Balanced Accuracy: 0.754
  - Precision: 0.278
  - Recall: 0.865
  - F1 Score: 0.420

**Technical Highlights**:
- Robust performance on imbalanced data (13.7% positive class)
- Advanced uncertainty quantification using ensemble techniques
- Risk-stratified predictions with adaptive thresholds
- Comprehensive calibration with 35 models per stage

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
