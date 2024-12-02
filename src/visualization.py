import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def plot_feature_importance(feature_importance):
    """Plot feature importance with proper feature names."""
    plt.clf()  # Clear the current figure
    
    # Create a mapping for feature names
    feature_name_mapping = {
        # Base features
        'BMI': 'Body Mass Index',
        'Age': 'Age',
        'GenHlth': 'General Health',
        'MentHlth': 'Mental Health Days',
        'PhysHlth': 'Physical Health Days',
        'HighBP': 'High Blood Pressure',
        'HighChol': 'High Cholesterol',
        'Smoker': 'Smoking Status',
        'Stroke': 'Stroke History',
        'HeartDiseaseorAttack': 'Heart Disease/Attack',
        'PhysActivity': 'Physical Activity',
        'Fruits': 'Fruit Consumption',
        'Veggies': 'Vegetable Consumption',
        'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
        'AnyHealthcare': 'Has Healthcare Coverage',
        'NoDocbcCost': 'No Doctor due to Cost',
        'DiffWalk': 'Difficulty Walking',
        'Education': 'Education Level',
        'Income': 'Income Level',
        
        # Engineered features
        'BasicRiskScore': 'Combined Health Risk Score',
        'RiskCategory': 'Health Risk Category',
        'BMI_Category': 'BMI Category',
        'Age_Group': 'Age Group',
        'Lifestyle_Score': 'Lifestyle Health Score',
        
        # Interaction terms
        'BP_Chol_Interaction': 'Blood Pressure × Cholesterol',
        'BMI_BP_Interaction': 'BMI × Blood Pressure',
        'Age_BMI_Interaction': 'Age × BMI',
        'Health_Lifestyle_Interaction': 'Health Status × Lifestyle',
        'Risk_Age_Interaction': 'Risk Score × Age',
        
        # Composite features
        'CardiovascularRisk': 'Cardiovascular Risk Index',
        'LifestyleRisk': 'Lifestyle Risk Index',
        'MetabolicRisk': 'Metabolic Risk Score',
        
        # Common variations in feature names
        'BP_Chol': 'Blood Pressure × Cholesterol',
        'Heart_BP': 'Heart Disease × Blood Pressure',
        'BMI_BP': 'BMI × Blood Pressure',
        'Age_Risk': 'Age × Risk Score',
        'Health_Activity': 'Health Status × Physical Activity'
    }
    
    try:
        # Calculate average importance across all models
        importance_cols = [col for col in feature_importance.columns if col != 'Feature']
        feature_importance['Avg_Importance'] = feature_importance[importance_cols].mean(axis=1)
        feature_importance['Std_Importance'] = feature_importance[importance_cols].std(axis=1)
        
        # Map feature names to more readable versions
        feature_importance['Display_Name'] = feature_importance['Feature'].map(
            lambda x: feature_name_mapping.get(x, x)
        )
        
        # Sort by average importance and get top 20
        sorted_data = feature_importance.sort_values('Avg_Importance', ascending=True).tail(20)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(15, 10))
        bars = ax.barh(range(len(sorted_data)), sorted_data['Avg_Importance'],
                      xerr=sorted_data['Std_Importance'], align='center',
                      color='skyblue', ecolor='black', capsize=5)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['Display_Name'], fontsize=10)
        ax.set_xlabel('Average Feature Importance', fontsize=12)
        ax.set_title('Top 20 Most Important Features\n(Error bars show standard deviation across models)', 
                    fontsize=14, pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels on the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + sorted_data['Std_Importance'].iloc[i],
                   bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}',
                   ha='left', va='center', fontsize=9)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('metrics/plots/feature_importance.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_feature_importance: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_uncertainty_distribution(probas: np.ndarray, 
                                uncertainties: dict,
                                sample_size: int = 10000) -> None:
    """Plot probability vs uncertainty distribution with sampling."""
    plt.clf()  # Clear the current figure
    
    # Sample data if larger than sample_size
    if len(probas) > sample_size:
        idx = np.random.choice(len(probas), sample_size, replace=False)
        probas_sample = probas[idx]
        uncertainties_sample = {k: v[idx] for k, v in uncertainties.items()}
    else:
        probas_sample = probas
        uncertainties_sample = uncertainties
    
    try:
        # Create subplots for each uncertainty type
        n_types = len(uncertainties_sample)
        fig, axes = plt.subplots(n_types, 2, figsize=(12, 4*n_types))
        
        for i, (unc_type, unc_values) in enumerate(uncertainties_sample.items()):
            ax1, ax2 = axes[i] if n_types > 1 else axes
            
            # Scatter plot
            ax1.scatter(probas_sample, unc_values, alpha=0.3, s=5)
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel(f'{unc_type.title()} Uncertainty')
            ax1.set_title(f'{unc_type.title()} Uncertainty vs Probability')
            
            # Histogram
            ax2.hist(unc_values, bins=50, alpha=0.7, color='skyblue')
            ax2.set_xlabel(f'{unc_type.title()} Uncertainty')
            ax2.set_ylabel('Count')
            ax2.set_title(f'{unc_type.title()} Uncertainty Distribution')
        
        plt.tight_layout()
        plt.savefig('metrics/plots/uncertainty_distribution.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_uncertainty_distribution: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """Plot calibration curve to show reliability of probability predictions."""
    plt.clf()  # Clear the current figure
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Plot model calibration
        plt.plot(prob_pred, prob_true, 'o-', label='Model Calibration')
        
        # Customize plot
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('True Probability in Each Bin')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('metrics/plots/calibration_curve.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_calibration_curve: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_probability_distribution(y_true, y_prob):
    """Plot distribution of predicted probabilities for each class."""
    plt.clf()  # Clear the current figure
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot distributions
        plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='No Diabetes (Actual)',
                 density=True, color='blue')
        plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Diabetes (Actual)',
                 density=True, color='red')
        
        # Customize plot
        plt.xlabel('Predicted Probability of Diabetes')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities by Actual Class')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('metrics/plots/probability_distribution.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_probability_distribution: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_class_distribution(y_train, y_train_resampled, title="Class Distribution"):
    """Plot class distribution before and after resampling."""
    plt.clf()  # Clear the current figure
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Original distribution
        pd.Series(y_train).value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Original Distribution')
        ax1.set_ylabel('Count')
        
        # Resampled distribution
        pd.Series(y_train_resampled).value_counts().plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Resampled Distribution')
        
        plt.tight_layout()
        plt.savefig('metrics/plots/class_distribution.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_class_distribution: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_metrics_summary(metrics_dict: dict, y_true=None, y_pred=None, y_prob=None) -> None:
    """
    Plot comprehensive metrics summary including calibration analysis.
    
    Args:
        metrics_dict: Dictionary containing various metric scores
        y_true: True labels (optional)
        y_pred: Predicted labels (optional)
        y_prob: Predicted probabilities (optional)
    """
    plt.clf()  # Clear the current figure
    
    try:
        plt.figure(figsize=(15, 10))
        
        # Create a grid for different metric groups
        gs = plt.GridSpec(2, 2)
        
        # 1. Basic Metrics
        ax1 = plt.subplot(gs[0, 0])
        basic_metrics = {k: v for k, v in metrics_dict['screening_metrics'].items() 
                        if k in ['accuracy', 'precision', 'recall', 'f1']}
        bars = ax1.bar(range(len(basic_metrics)), basic_metrics.values(), 
                       color='skyblue')
        ax1.set_xticks(range(len(basic_metrics)))
        ax1.set_xticklabels(basic_metrics.keys(), rotation=45)
        ax1.set_title('Basic Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Probability Metrics
        ax2 = plt.subplot(gs[0, 1])
        prob_metrics = metrics_dict['probability_metrics']
        bars = ax2.bar(range(len(prob_metrics)), prob_metrics.values(),
                       color='lightgreen')
        ax2.set_xticks(range(len(prob_metrics)))
        ax2.set_xticklabels(prob_metrics.keys(), rotation=45)
        ax2.set_title('Probability Metrics')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('metrics/plots/metrics_summary.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_metrics_summary: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_calibration_analysis(y_true, y_prob, n_bins=10):
    """
    Plot detailed calibration analysis including reliability diagram and histogram.
    """
    plt.clf()  # Clear the current figure
    
    try:
        # If y_prob is 2D, take the probability of positive class (class 1)
        if len(y_prob.shape) > 1:
            y_prob = y_prob[:, 1]
            
        plt.figure(figsize=(15, 5))
        
        # 1. Reliability Diagram
        ax1 = plt.subplot(121)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        ax1.plot(prob_pred, prob_true, "s-", label="Model")
        
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram")
        ax1.legend()
        
        # 2. Prediction Histogram
        ax2 = plt.subplot(122)
        ax2.hist(y_prob, range=(0, 1), bins=n_bins, histtype="step",
                 lw=2, label="Model")
        ax2.set_xlabel("Mean Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Prediction Distribution")
        
        plt.tight_layout()
        plt.savefig('metrics/plots/calibration_analysis.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in plot_calibration_analysis: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_class_performance(y_true, y_pred, probas, title="Class Performance Analysis"):
    """Plot performance metrics for each class."""
    plt.clf()  # Clear the current figure
    
    try:
        # If probas is 2D, take the probability of positive class (class 1)
        if len(probas.shape) > 1:
            probas = probas[:, 1]
            
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Confusion Matrix
        plt.subplot(131)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot 2: ROC Curve per class
        plt.subplot(132)
        fpr, tpr, _ = roc_curve(y_true, probas)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, probas):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Plot 3: Precision-Recall Curve
        plt.subplot(133)
        precision, recall, _ = precision_recall_curve(y_true, probas)
        plt.plot(recall, precision, label=f'PR (AP = {average_precision_score(y_true, probas):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('metrics/plots/class_performance.png', bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Error in plot_class_performance: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

def plot_confidence_by_class(y_true, confidences, predictions, title="Confidence Distribution by Class"):
    """Plot confidence score distribution for each class."""
    plt.clf()  # Clear the current figure
    
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Confidence distribution
        plt.subplot(121)
        for label in [0, 1]:
            mask = y_true == label
            sns.kdeplot(confidences[mask], label=f'Class {label}')
        plt.title('Confidence Distribution by True Class')
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.legend()
        
        # Plot 2: Confidence vs Accuracy
        plt.subplot(122)
        conf_bins = np.linspace(0, 1, 11)
        accuracies = []
        conf_means = []
        
        for i in range(len(conf_bins)-1):
            mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
            if np.sum(mask) > 0:
                acc = accuracy_score(y_true[mask], predictions[mask])
                accuracies.append(acc)
                conf_means.append((conf_bins[i] + conf_bins[i+1])/2)
        
        plt.plot(conf_means, accuracies, 'o-')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Reliability Diagram')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('metrics/plots/confidence_analysis.png', bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Error in plot_confidence_by_class: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed