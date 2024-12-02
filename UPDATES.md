# Project Updates Log

## Latest Changes (Most Recent First)

### 2024-03-22: Model Parameter Optimization for Balanced Performance

#### Model Architecture Improvements
- **Threshold Adjustments**:
  - Screening threshold increased to 0.12 (from 0.08)
  - Confirmation threshold increased to 0.15 (from 0.10)
  - Better balance between false positives and false negatives

- **Enhanced Model Parameters**:
  - Increased `n_estimators` to 500 for better convergence
  - Reduced `learning_rate` to 0.03 for improved generalization
  - Optimized `max_depth` to 8 for balanced model complexity
  - Adjusted `scale_pos_weight` to 6 for more balanced predictions
  - Increased sampling rates to 0.85 for better stability
  - Fine-tuned tree-specific parameters for robustness

- **Sampling Strategy Optimization**:
  - Modified SMOTE sampling strategy from 1.0 to 0.5
  - Increased k_neighbors in SMOTE from 3 to 5
  - Consistent sampling parameters across stages

- **Calibration Enhancements**:
  - Increased calibration models to 25 (from 20)
  - Increased Monte Carlo iterations to 35 (from 30)

#### Expected Improvements
1. **Better Balance**:
   - More balanced precision-recall trade-off
   - Improved handling of minority class
   - Reduced false positive rate while maintaining sensitivity

2. **Model Robustness**:
   - Better generalization through reduced learning rate
   - More stable predictions with increased ensemble size
   - Improved synthetic sample quality

3. **Uncertainty Estimation**:
   - More reliable uncertainty estimates
   - Better calibrated probabilities
   - Enhanced confidence scoring

#### Next Steps
1. Evaluate model performance with new parameters
2. Fine-tune thresholds based on results
3. Consider additional feature engineering if needed

### 2024-03-21: Model Performance Analysis and Interactive Visualization

#### Performance Metrics Review
- **Current Performance**:
  - Screening Stage (Threshold: 0.2)
    - Accuracy: 0.786
    - Precision: 0.346
    - Recall: 0.607
    - F1 Score: 0.441
  - Confirmation Stage (Threshold: 0.6)
    - Accuracy: 0.861
    - Precision: 0.531
    - Recall: 0.037
    - F1 Score: 0.069
  - Overall Metrics
    - ROC AUC: 0.806
    - Average Precision: 0.387
    - Brier Score: 0.102

#### Areas for Improvement
- **Identified Issues**:
  - Low recall in confirmation stage (0.037)
  - Moderate precision in confirmation stage (0.531)
  - Class imbalance affecting model performance

#### Proposed Enhancements
1. **Threshold Optimization**
   - Consider adjusting screening threshold for better precision
   - Lower confirmation threshold to improve recall
   - Need to find optimal balance between precision and recall

2. **Model Calibration**
   - Increase calibration models from 5 to 10 for better uncertainty estimates
   - Implement class weights to handle imbalanced data
   - Consider SMOTE or other advanced sampling techniques

#### Visualization Updates
- Added interactive Plotly dashboard with:
  - Prediction Uncertainty vs Probability plot
  - Distribution of Uncertainties visualization
  - Calibration Curve
  - Decision Boundaries with uncertainty regions
- Dashboard accessible via local server (http://127.0.0.1:9799/)

#### Next Steps
1. Implement proposed model enhancements
2. Conduct thorough performance evaluation after changes
3. Consider adding static plot options for non-interactive use cases

### 2024-03-20: Feature Scaling and Uncertainty Handling Fixes

#### Preprocessor Improvements
- **Purpose**: Fix feature scaling and data type compatibility issues
- **Changes**:
  - Refactored feature scaling to handle all continuous features together
  - Fixed data type inconsistencies in feature transformation
  - Improved handling of missing features during transform
  - Enhanced error handling and logging

#### Key Improvements
1. **Feature Scaling**
   - Now scales all continuous features together using a single StandardScaler
   - Maintains feature name consistency between fit and transform
   - Properly handles missing features during transformation
   - Added type casting to float32 for numerical stability

2. **Uncertainty Handling**
   - Updated prediction code to handle detailed uncertainty metrics
   - Now properly processes epistemic, aleatoric, and total uncertainty
   - Enhanced confidence score calculation using total uncertainty
   - Added detailed uncertainty components to prediction output

3. **Code Robustness**
   - Improved error messages for missing features
   - Added validation for continuous feature presence
   - Enhanced data type consistency checks
   - Better handling of edge cases in transformation

These improvements ensure:
- Consistent feature scaling across all stages
- Proper handling of uncertainty metrics
- More robust data type handling
- Better error reporting and debugging

### 2024-03-19: Data Type Compatibility and Warning Fixes

#### Preprocessor Improvements
- **Purpose**: Fix data type compatibility warnings and improve data handling
- **Changes**:
  - Updated scaling operations in `transform` and `fit_transform` methods to handle data types properly
  - Implemented column-by-column scaling with explicit `float32` casting
  - Improved DataFrame operations to avoid dtype incompatibility warnings
  - Enhanced robustness of numeric data handling

#### Key Improvements
1. **Data Type Handling**
   - Explicit casting to `float32` for all scaled features
   - Column-wise assignment to avoid pandas warnings
   - Improved compatibility with numpy arrays

2. **Code Optimization**
   - More efficient DataFrame operations
   - Better memory usage with proper data types
   - Cleaner code structure for scaling operations

3. **Testing**
   - All tests passing without warnings
   - Verified data type consistency
   - Maintained all existing functionality

These improvements ensure:
- Clean execution without dtype warnings
- Consistent numeric precision
- Better memory efficiency
- Maintained data integrity during transformations

### 2024-03-19: Data Type Compatibility and Documentation Updates

#### Preprocessor Improvements
- **Purpose**: Fix data type compatibility warnings and improve data handling
- **Changes**:
  - Updated scaling operations in `transform` and `fit_transform` methods
  - Implemented column-by-column scaling with explicit `float32` casting
  - Improved DataFrame operations to avoid dtype incompatibility warnings

#### Documentation Enhancements
- Added comprehensive dataset information to README.md
- Updated installation instructions with conda environment setup
- Enhanced project structure documentation
- Added detailed feature descriptions
- Included performance metrics section
- Added acknowledgments for data sources and libraries

#### Testing
- All tests passing without warnings (8 passed, 3 skipped)
- Verified data type consistency across operations
- Maintained existing functionality while improving robustness

### 2024-03-18: Test Suite Enhancement and Bug Fixes

#### Test Suite Updates
- **Purpose**: Improve test coverage and fix test failures
- **Changes**:
  - Updated test data to include all required medical features
  - Fixed fit_transform calls to properly handle target variable
  - Added more robust tests for medical score validation
  - Enhanced batch processing tests
  - Added uncertainty calibration validation

#### Test Cases Added/Fixed
1. Medical Risk Score Validation
   - Verifies risk scores are within valid ranges (0-1)
   - Checks for presence of medical scoring features

2. Uncertainty Calibration
   - Tests correlation between uncertainty and prediction confidence
   - Validates uncertainty estimates are properly bounded

3. Batch Processing
   - Tests large dataset handling
   - Verifies consistency across batch processing

4. Clinical Thresholds
   - Validates medical domain constraints
   - Tests handling of out-of-range values

#### Known Issues to Address
1. Need to implement proper error handling for edge cases
2. Consider adding more specific medical domain tests
3. Add performance benchmarking tests
4. Consider adding cross-validation tests

### 2024-01-09: Code Optimization and Cleanup

#### Ensemble Model Optimization
- **Purpose**: Improve model efficiency and reduce training time
- **Changes**:
  - Reduced calibration models from 5 to 3
  - Added early stopping in model training
  - Optimized model parameters for XGBoost, LightGBM, and CatBoost
  - Enhanced uncertainty estimation using dropout-like approach

#### Preprocessor Enhancement
- **Purpose**: Improve data quality and processing efficiency
- **Changes**:
  - Added robust scaling (5-95 quantile range)
  - Enhanced outlier detection with LOF and parallel processing
  - Implemented vectorized operations for medical scores
  - Improved feature validation with better error messages
  - Optimized correlation calculations

#### Visualization Improvements
- **Purpose**: Better model interpretability and analysis
- **Changes**:
  - Added density plots for distributions
  - Enhanced KDE plots for uncertainties
  - Improved plot aesthetics and configurability
  - Added interactive dashboard using Plotly

#### Code Cleanup
- **Purpose**: Improve maintainability and remove redundancy
- **Changes**:
  - Removed empty/unused files:
    - `src/data_loader.py`
    - `src/utils.py`
  - Removed redundant test file from root directory
  - Removed empty notebooks directory
  - Updated test suite to match current implementation

### Preprocessing Pipeline Enhancements

### Class Imbalance Handling
- Implemented ADASYN (Adaptive Synthetic Sampling) to handle class imbalance in the diabetes dataset
- Chosen over SMOTE for its adaptive nature and better handling of medical data:
  - Generates more synthetic samples for harder-to-learn diabetic cases
  - Focuses on decision boundary cases where classification is most critical
  - Maintains clinical relevance of synthetic samples

### ADASYN Configuration
- Set sampling_strategy to 0.75 to maintain some natural class imbalance while improving minority class representation
- Using 5 nearest neighbors for synthetic sample generation
- Improved handling of feature spaces
- Added detailed class distribution reporting:
  - Pre-resampling distribution
  - Post-resampling distribution
  - Percentage breakdowns

### Data Validation and Cleaning
- Improved duplicate handling:
  - Removing exact duplicates before applying ADASYN
  - Using required columns as criteria for duplicate detection
- Added logging of class distribution before and after resampling
- Enhanced error handling with informative messages

### Medical Domain Considerations
- Preserved medical feature relationships by applying ADASYN after:
  - Outlier detection and handling
  - Medical risk score calculation
  - Clinical interaction feature creation
- This ensures synthetic samples maintain clinically relevant patterns and relationships

### Class Balance Improvements

### Enhanced Class Balance Handling
- Implemented dynamic class weight calculation based on actual data distribution
- Added PR-AUC (Precision-Recall Area Under Curve) as an additional evaluation metric
- Both screening and confirmation stages now use balanced weights

#### Key Changes
1. **Dynamic Class Weights**
   - Automatically calculates optimal weights based on class distribution
   - Ensures fair treatment of both positive and negative cases
   - Adapts to changes in data distribution

2. **Evaluation Metrics**
   - Added PR-AUC alongside ROC-AUC
   - Better evaluation of model performance on imbalanced datasets
   - More sensitive to improvements in minority class prediction

3. **Balanced Training**
   - Both stages (screening and confirmation) use balanced weights
   - Removed fixed weight biases that favored one class over another
   - Maintained stratified sampling for consistent class ratios

These improvements ensure:
- Equal importance to both diabetic and non-diabetic predictions
- Better handling of class imbalance without bias
- More reliable performance metrics for imbalanced data

### Improved Preprocessing and Class Balance

### Improved Missing Value Handling
- Implemented intelligent imputation strategies:
  - Median imputation for clinical measurements (BMI, Age)
  - Mode imputation for categorical variables
- Added detailed logging of missing value counts per feature

### Robust Outlier Detection
- Switched to Local Outlier Factor (LOF) with improved configuration:
  - `n_neighbors=20` for better local density estimation
  - `contamination=0.1` to identify top 10% anomalies
  - `novelty=False` for inlier/outlier prediction
  - Parallel processing with `n_jobs=-1`
- Added proper handling of missing values before outlier detection
- Implemented index-safe outlier masking

### Enhanced Feature Processing
- Separated features into distinct groups:
  - Continuous: BMI, PhysHlth, MentHlth, Age
  - Binary: HighBP, HighChol, etc.
  - Categorical: GenHlth, Education, Income
- Standardized continuous features using StandardScaler
- Added validation for binary features

### Refined ADASYN Configuration
- Target ratio set to 75% of majority class
- Using 5 nearest neighbors for synthetic sample generation
- Improved handling of feature spaces
- Added detailed class distribution reporting:
  - Pre-resampling distribution
  - Post-resampling distribution
  - Percentage breakdowns

### Testing Framework
- Added test mode for quick validation:
  - Uses stratified 10% sample of full dataset
  - Maintains class proportions
  - Enables rapid iteration and validation
- Enhanced visualization capabilities:
  - Class distribution plots
  - Performance metrics by class
  - Confidence score analysis

### Data Quality Checks
- Added comprehensive validation steps:
  - Feature value range checks
  - Binary feature validation
  - Categorical value verification
- Improved error reporting and logging

These improvements ensure:
1. More reliable handling of missing data
2. Better outlier detection without index errors
3. Fair treatment of both diabetic and non-diabetic cases
4. Rapid testing and validation of changes
5. Comprehensive monitoring of data quality

### Latest Updates (Test Fixes and Preprocessing Improvements)

### Fixed Test Suite Issues

1. **Preprocessor Improvements**
   - Fixed missing value handling in `fit` method to ensure LOF gets clean data
   - Improved index alignment during duplicate removal to maintain data consistency
   - Enhanced outlier detection workflow by handling missing values before LOF

2. **Test Suite Enhancements**
   - Updated `test_batch_processing` to properly handle missing values before LOF
   - Fixed `test_ensemble_model` to compare against processed data length instead of original
   - Enhanced `test_uncertainty_calibration` to handle both dictionary and array outputs
   - Added proper array reshaping for correlation calculations

3. **Data Processing Flow Improvements**
   - Reordered preprocessing steps for better data quality:
     1. Data validation
     2. Missing value handling
     3. Duplicate removal (with index alignment)
     4. Outlier detection
     5. Feature engineering
     6. Scaling
     7. Feature selection

4. **Code Quality Improvements**
   - Added better error handling and validation in preprocessing pipeline
   - Improved test assertions for more robust validation
   - Enhanced shape checking for uncertainty calculations
   - Added proper data copying to prevent unintended modifications

These updates improve the robustness of the preprocessing pipeline and ensure consistent behavior across different data scenarios.

### 2024-01-07: Improved Two-Stage Model Architecture

### Enhanced Model Training Strategy
- Modified the two-stage model architecture to maximize learning from all available data:
  - Both screening and confirmation models now train on the complete dataset
  - Removed previous limitation where confirmation stage only trained on screened samples
  - Maintained distinct roles through different thresholds rather than data filtering

### Balanced Learning Improvements
- Standardized SMOTE application across both stages:
  - Consistent sampling_strategy=0.5 for both screening and confirmation
  - Both stages now benefit from balanced class distributions during training
  - Equal k_neighbors=5 setting for consistent synthetic sample quality
  - Maintained random_state=42 for reproducibility

### Benefits of New Architecture
- More robust model training:
  - Both stages have access to complete data patterns
  - No information loss from premature filtering
  - Better pattern recognition across the full dataset
  - Maintained model specialization through thresholds rather than data subsetting

### Technical Implementation
- Removed conditional training in confirmation stage
- Simplified training pipeline with consistent SMOTE application
- Maintained calibration approach for uncertainty estimation
- Preserved ensemble structure with LightGBM, XGBoost, and CatBoost

This update represents a significant improvement in our model's architecture, allowing both stages to learn from all available data while maintaining their specialized roles in the prediction pipeline.

### 2024-01-07: Enhanced Model Evaluation and Test Set Balancing

### Test Set Balancing
- Implemented balanced test set creation while maintaining original distribution in training data
- Added `create_balanced_test_set` function that ensures:
  - Equal representation of diabetic and non-diabetic cases in test set
  - Preserves real-world class distribution in training set
  - Prevents over-extraction from minority class
  - Maintains reproducibility with consistent random state

### Enhanced Evaluation Metrics
- Added comprehensive evaluation metrics to better assess model performance:
  - Balanced accuracy score (normalized by class size)
  - Weighted versions of precision, recall, and F1 score
  - Class distribution information in test set
- Updated metrics output to show:
  - Test set class distribution
  - Both standard and weighted metrics for each model stage
  - Clear separation between screening and confirmation metrics

### Benefits
- More reliable model evaluation on balanced test set
- Better understanding of model performance across both classes
- Maintained real-world training conditions while ensuring fair testing
- Enhanced visibility into model behavior through comprehensive metrics

### 2024-01-07: Enhanced Ensemble Prediction Strategy

### Advanced Model Collaboration
- Implemented sophisticated ensemble prediction combining three strategies:
  1. **Dynamic Performance-Based Weights**
     - Weights adapt based on each model's ROC-AUC score
     - Models that perform better get higher influence
     - Separate weights for screening and confirmation stages
  
  2. **Range-Specific Weighting**
     - Different weights for different prediction ranges:
       - Very low risk (0.0-0.2)
       - Low risk (0.2-0.4)
       - Medium risk (0.4-0.6)
       - High risk (0.6-0.8)
       - Very high risk (0.8-1.0)
     - Models get higher weights in ranges where they excel
     - Helps handle different risk levels more accurately

  3. **Adaptive Voting System**
     - Uses weighted averaging for low uncertainty predictions
     - Switches to majority voting for high uncertainty cases (>0.2)
     - Provides more robust predictions when models disagree

### Weight Calculation
- Combined weighting scheme:
  - 70% based on overall model performance
  - 30% based on range-specific performance
  - Weights normalized to ensure proper probability distribution
  - Updated separately for screening and confirmation stages

### Technical Improvements
- Added performance tracking during training:
  - Overall ROC-AUC scores for each model
  - Range-specific performance metrics
  - Uncertainty monitoring across prediction ranges
- Enhanced model collaboration:
  - LightGBM, XGBoost, and CatBoost now work together more effectively
  - Each model contributes based on its strengths
  - System adapts to model performance in different scenarios

### Benefits
1. **More Reliable Predictions**
   - Better handling of edge cases
   - Reduced impact of individual model weaknesses
   - Improved confidence in high-uncertainty regions

2. **Better Risk Assessment**
   - More accurate predictions in different risk ranges
   - Clearer indication when models disagree
   - More reliable uncertainty estimates

3. **Clinical Relevance**
   - Different strategies for different risk levels
   - More conservative approach in high-uncertainty cases
   - Better support for medical decision-making

This update represents a significant advancement in how our models collaborate to make predictions, making the system more robust and clinically relevant.

### 2024-01-21: Probability Calibration Improvements

### Added
- New visualization tools in `visualization.py`:
  - Enhanced calibration curve plotting
  - Probability distribution visualization for each class
- Probability calibration using isotonic regression to improve Brier Score
- Additional metrics reporting for model calibration

### Modified
- Updated `main.py` to include probability calibration step
- Enhanced model evaluation with both initial and calibrated Brier Scores
- Improved visualization module with more detailed probability analysis tools

### Technical Details
- Added `CalibratedClassifierCV` with isotonic regression for better probability estimates
- Enhanced probability metrics reporting
- Added new visualization functions:
  - `plot_calibration_curve`: Shows reliability of probability predictions
  - `plot_probability_distribution`: Displays distribution of predicted probabilities by class

### 2024-01-XX: Enhanced Model Evaluation and Visualization

### Changes Made
1. **Improved Metrics Visualization**
   - Added comprehensive metrics summary visualization showing:
     - Basic classification metrics (accuracy, precision, recall, F1)
     - Calibration metrics (Brier score, log loss)
     - ROC curve with AUC score
     - Precision-Recall curve with average precision
   - Created detailed calibration analysis plots:
     - Reliability diagram showing probability calibration
     - Prediction distribution histogram
   - Enhanced feature importance visualization with:
     - More readable feature names
     - Error bars showing variation across models
     - Value labels for easier interpretation

2. **Model Performance Updates**
   - Improved probability calibration (lower Brier score)
   - Better uncertainty quantification
   - More reliable probability estimates
   - Trade-off between raw accuracy and calibration quality

### Technical Details
- Updated `visualization.py` with new plotting functions:
  - `plot_metrics_summary`: Comprehensive 4-panel metrics visualization
  - `plot_calibration_analysis`: Detailed probability calibration analysis
- Modified `main.py` to incorporate new visualizations in the training pipeline
- Enhanced feature name mapping for better interpretability

### Next Steps
- Monitor model performance with new metrics
- Fine-tune probability calibration if needed
- Consider adding model explainability features
- Prepare for potential deployment

### Current Project Structure
```
diabetes-prediction/
├── src/
│   ├── main.py           # Main execution script
│   ├── preprocessor.py   # Data preprocessing pipeline
│   ├── ensemble_model.py # Two-stage ensemble model
│   └── visualization.py  # Visualization tools
├── tests/
│   └── test_pipeline.py  # Test suite
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
└── UPDATES.md           # This file
```

### Challenges and Solutions

#### Performance Issues
- **Problem**: Initial model was too slow during training
- **Solution**: 
  - Reduced calibration models from 5 to 3
  - Implemented early stopping
  - Added vectorized operations in preprocessing
  - Parallelized outlier detection

#### Memory Usage
- **Problem**: High memory consumption with large datasets
- **Solution**:
  - Optimized correlation calculations
  - Implemented batch processing for large datasets
  - Reduced duplicate data storage in ensemble models

#### Code Organization
- **Problem**: Scattered functionality across too many files
- **Solution**:
  - Consolidated core functionality into four main files
  - Removed empty/redundant files
  - Improved code modularity and reusability

#### Model Calibration
- **Problem**: Initial predictions were not well-calibrated
- **Solution**:
  - Implemented two-stage approach
  - Added uncertainty estimation
  - Used multiple calibrated models

#### Data Quality
- **Problem**: Outliers and missing values affecting model performance
- **Solution**:
  - Added robust scaling (5-95 quantile range)
  - Enhanced outlier detection with LOF
  - Improved feature validation

#### Testing Issues
- **Problem**: Test suite was using outdated class names
- **Solution**:
  - Updated test suite to match current implementation
  - Added more comprehensive test cases
  - Improved test documentation

#### Visualization Clarity
- **Problem**: Initial plots were not intuitive for medical professionals
- **Solution**:
  - Added density plots for better distribution visualization
  - Enhanced plot aesthetics
  - Created interactive dashboard
  - Added configurable plot parameters

### Model Performance Metrics
- Screening Stage (Threshold: 0.2)
  - Optimized for high recall
  - Uses ensemble of calibrated models
- Confirmation Stage (Threshold: 0.6)
  - Optimized for high precision
  - Includes uncertainty estimation

### Known Limitations
1. Current implementation assumes batch processing (not suitable for real-time predictions)
2. Uncertainty estimation adds computational overhead
3. Model requires all features to be present (no partial prediction support)
4. Interactive dashboard may be slow with very large datasets

### Tips for Future Development
1. Always run test suite before making major changes
2. Keep memory usage in mind when working with large datasets
3. Maintain balance between model complexity and performance
4. Document any new features or changes in this file
5. Consider medical domain requirements when making changes

### Next Steps
- [ ] Run comprehensive tests on the updated pipeline
- [ ] Gather performance metrics after optimizations
- [ ] Consider adding model explainability features
- [ ] Prepare for potential deployment

---
Note: This file will be continuously updated as new changes are made to the project.

### Diabetes Prediction Model Updates

## Latest Updates (2024)

### Recent Changes and Fixes

1. **Test Mode Improvements**
   - Reduced test dataset size to 1% for faster testing
   - Added lightweight model parameters for test mode
   - Fixed feature consistency issues between training and prediction
   - Improved missing value handling in preprocessor

2. **Preprocessor Enhancements**
   - Added proper feature name tracking during fit_transform
   - Improved missing value handling with appropriate defaults for each feature type
   - Fixed feature count mismatch issues between training and prediction
   - Added validation for required columns

3. **Model Architecture**
   - Implemented two-stage ensemble model (screening and confirmation)
   - Added model calibration for better probability estimates
   - Integrated uncertainty estimation
   - Added feature importance tracking

### Known Issues and TODOs

1. **Performance**
   - [ ] Model calibration is computationally expensive
   - [ ] SMOTE oversampling slows down training
   - [ ] Need to optimize feature selection process

2. **Features**
   - [ ] Add feature importance visualization
   - [ ] Implement cross-validation for more robust evaluation
   - [ ] Add model interpretability tools (SHAP values)
   - [ ] Create interactive dashboard for predictions

3. **Testing**
   - [ ] Add unit tests for core components
   - [ ] Create integration tests
   - [ ] Add performance benchmarks

### Project Structure

```
diabetes-prediction/
├── src/
│   ├── main.py            # Main training and prediction pipeline
│   ├── preprocessor.py    # Data preprocessing and feature engineering
│   ├── ensemble_model.py  # Two-stage ensemble model implementation
│   ├── visualization.py   # Plotting and visualization utilities
│   └── predict_example.py # Example prediction script
├── models/                # Saved model files
├── metrics/              # Performance metrics and logs
└── data/                 # Dataset directory
```

### Model Components

1. **Preprocessor**
   - Feature validation and cleaning
   - Missing value imputation
   - Feature scaling
   - Outlier detection
   - Feature engineering

2. **Ensemble Model**
   - First stage: Screening model (high recall)
   - Second stage: Confirmation model (high precision)
   - Model calibration
   - Uncertainty estimation
   - Feature importance analysis

3. **Visualization**
   - Performance metrics plots
   - ROC and PR curves
   - Calibration plots
   - Feature importance charts

### Dependencies

- scikit-learn
- pandas
- numpy
- xgboost
- lightgbm
- catboost
- imbalanced-learn
- matplotlib
- seaborn

### Usage Instructions

1. **Training**
   ```bash
   python src/main.py --mode train --data path/to/data.csv --visualize
   ```

2. **Testing**
   ```bash
   python src/main.py --mode test
   ```

3. **Prediction**
   ```bash
   python src/main.py --mode predict --data path/to/predict.csv
   ```

### Performance Metrics

1. **Screening Stage**
   - Optimized for high recall
   - Current metrics:
     - Accuracy: ~0.69
     - Balanced Accuracy: ~0.71
     - Recall: ~0.73
     - F1 Score: ~0.39

2. **Confirmation Stage**
   - Optimized for high precision
   - Current metrics:
     - Accuracy: ~0.73
     - Balanced Accuracy: ~0.70
     - Precision: ~0.28
     - F1 Score: ~0.40

3. **Overall Performance**
   - ROC AUC: ~0.78
   - Average Precision: ~0.35
   - Brier Score: ~0.10

### Future Improvements

1. **Model Enhancement**
   - Implement stacking with neural networks
   - Add time-series features for longitudinal data
   - Explore deep learning approaches

2. **Feature Engineering**
   - Add interaction terms
   - Implement polynomial features
   - Add domain-specific medical features

3. **Production Readiness**
   - Add API endpoints
   - Implement model versioning
   - Add monitoring and logging
   - Create Docker container

4. **Documentation**
   - Add detailed API documentation
   - Create user guide
   - Add contribution guidelines
   - Add acknowledgments for data sources and libraries
