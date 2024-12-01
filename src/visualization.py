import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import os

def plot_evaluation_metrics(fold_results, title="Cross-Validation Performance", filename="cv_metrics.png"):
    """Plot evaluation metrics across folds with confidence intervals."""
    metrics = ['roc_auc', 'pr_auc', 'accuracy']
    n_folds = len(fold_results)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    # Calculate means and confidence intervals
    means = []
    ci_95 = []
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci = 1.96 * std_val / np.sqrt(n_folds)  # 95% confidence interval
        means.append(mean_val)
        ci_95.append(ci)
    
    # Plot bars with error bars
    plt.bar(x, means, width, yerr=ci_95, capsize=5)
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Score')
    
    # Add value labels on top of bars
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f'{v:.3f}±{ci_95[i]:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'plots', filename))
    plt.close()

def plot_evaluation_curves(model, X, y, filename_prefix="evaluation"):
    """Plot ROC and PR curves with uncertainty bands."""
    # Get predictions with uncertainty
    y_pred_proba = model.predict_proba(X)
    y_pred_mean = np.mean(y_pred_proba, axis=0)
    y_pred_std = np.std(y_pred_proba, axis=0)
    
    # ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y, y_pred_mean)
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {model.evaluate(X, y)["roc_auc"]:.3f})')
    
    # Add uncertainty bands
    tpr_upper = np.minimum(tpr + y_pred_std.mean(), 1)
    tpr_lower = np.maximum(tpr - y_pred_std.mean(), 0)
    plt.fill_between(fpr, tpr_lower, tpr_upper, color='b', alpha=0.2, label='Uncertainty')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Uncertainty')
    plt.legend(loc="lower right")
    
    # PR curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y, y_pred_mean)
    plt.plot(recall, precision, 'r-', label=f'PR (AUC = {model.evaluate(X, y)["pr_auc"]:.3f})')
    
    # Add uncertainty bands
    precision_upper = np.minimum(precision + y_pred_std.mean(), 1)
    precision_lower = np.maximum(precision - y_pred_std.mean(), 0)
    plt.fill_between(recall, precision_lower, precision_upper, color='r', alpha=0.2, label='Uncertainty')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Uncertainty')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'plots', f'{filename_prefix}_curves.png'))
    plt.close()

def plot_feature_importance(feature_names, importance_scores, title="Feature Importance", filename="feature_importance.png"):
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 6))
    
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.barh(pos, importance_scores[sorted_idx])
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Importance Score')
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'plots', filename))
    plt.close()

def plot_subgroup_performance(subgroup_results, group_type, filename=None):
    """Plot performance metrics across different subgroups."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    subgroups = list(subgroup_results[0].keys())
    metrics = ['roc_auc', 'pr_auc', 'accuracy']
    
    # Calculate mean and std for each subgroup and metric
    means = {metric: [] for metric in metrics}
    stds = {metric: [] for metric in metrics}
    
    for subgroup in subgroups:
        for metric in metrics:
            values = [fold[subgroup][metric] for fold in subgroup_results]
            means[metric].append(np.mean(values))
            stds[metric].append(np.std(values))
    
    # Plot
    x = np.arange(len(subgroups))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, means[metric], width, 
               yerr=stds[metric], label=metric.upper())
    
    plt.xlabel('Subgroups')
    plt.ylabel('Score')
    plt.title(f'Performance Metrics by {group_type}')
    plt.xticks(x + width, subgroups, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join('models', 'plots', filename))
    plt.close()
