import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def plot_feature_importance(feature_importance: pd.DataFrame,
                          figsize: tuple = (10, 6),
                          max_features: int = 15) -> None:
    """Plot feature importance with error bars."""
    plt.figure(figsize=figsize)
    
    # Prepare data - limit to top features
    importance_data = pd.DataFrame({
        'Feature': feature_importance['Feature'],
        'Importance': feature_importance['Avg_Importance'],
        'Std': feature_importance['Std_Importance']
    }).sort_values('Importance', ascending=False).head(max_features)
    
    # Create horizontal bar plot
    plt.barh(
        y=range(len(importance_data)), 
        width=importance_data['Importance'],
        xerr=importance_data['Std'],
        capsize=3,
        color='skyblue',
        alpha=0.7
    )
    
    plt.yticks(range(len(importance_data)), importance_data['Feature'])
    plt.xlabel('Importance Score')
    plt.title('Top Feature Importance')
    plt.tight_layout()
    plt.savefig('metrics/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_distribution(probas: np.ndarray, 
                                uncertainties: dict,
                                sample_size: int = 10000) -> None:
    """Plot probability vs uncertainty distribution with sampling."""
    # Sample data if larger than sample_size
    if len(probas) > sample_size:
        idx = np.random.choice(len(probas), sample_size, replace=False)
        probas_sample = probas[idx]
        uncertainties_sample = {k: v[idx] for k, v in uncertainties.items()}
    else:
        probas_sample = probas
        uncertainties_sample = uncertainties
    
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
    plt.savefig('metrics/uncertainty_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_curve(y_true: np.ndarray, 
                          y_prob: np.ndarray,
                          n_bins: int = 10) -> None:
    """Plot calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
    plt.plot(prob_pred, prob_true, 's-', label='Model calibration')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('metrics/calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(y_train, y_train_resampled, title="Class Distribution"):
    """Plot class distribution before and after resampling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original distribution
    pd.Series(y_train).value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Original Distribution')
    ax1.set_ylabel('Count')
    
    # Resampled distribution
    pd.Series(y_train_resampled).value_counts().plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Resampled Distribution')
    
    plt.tight_layout()
    plt.savefig('metrics/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_summary(metrics_dict: dict) -> None:
    """Plot summary of key metrics."""
    plt.figure(figsize=(10, 6))
    
    # Extract metrics
    stages = ['Screening', 'Confirmation']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create grouped bar plot
    x = np.arange(len(metrics))
    width = 0.35
    
    screen_vals = [metrics_dict['screening_metrics'][m] for m in metrics]
    confirm_vals = [metrics_dict['confirmation_metrics'][m] for m in metrics]
    
    plt.bar(x - width/2, screen_vals, width, label='Screening', color='skyblue')
    plt.bar(x + width/2, confirm_vals, width, label='Confirmation', color='lightgreen')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Summary')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('metrics/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_performance(y_true, y_pred, probas, title="Class Performance Analysis"):
    """Plot performance metrics for each class."""
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
    plt.savefig('plots/class_performance.png')
    plt.close()

def plot_confidence_by_class(y_true, confidences, predictions, title="Confidence Distribution by Class"):
    """Plot confidence score distribution for each class."""
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
    plt.savefig('plots/confidence_analysis.png')
    plt.close()