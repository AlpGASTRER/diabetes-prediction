import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score

def plot_feature_importance(feature_importance: pd.DataFrame,
                          figsize: tuple = (12, 8)) -> None:
    """Plot feature importance with error bars.
    
    Args:
        feature_importance: DataFrame with feature importance scores
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Prepare data
    importance_data = pd.DataFrame({
        'Feature': feature_importance['Feature'],
        'Importance': feature_importance['Avg_Importance'],
        'Std': np.std([
            feature_importance['RF_Importance'],
            feature_importance['MI_Importance']
        ], axis=0)
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot with error bars
    plt.barh(
        y=range(len(importance_data)), 
        width=importance_data['Importance'],
        xerr=importance_data['Std'],
        capsize=5,
        color='skyblue',
        alpha=0.7
    )
    
    plt.yticks(range(len(importance_data)), importance_data['Feature'])
    plt.xlabel('Feature Importance Score')
    plt.title('Feature Importance with Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_uncertainty_distribution(probas: np.ndarray, 
                                uncertainties: np.ndarray,
                                figsize: tuple = (15, 5)) -> None:
    """Plot probability vs uncertainty distribution.
    
    Args:
        probas: Predicted probabilities
        uncertainties: Prediction uncertainties
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot with density coloring
    density = sns.kdeplot(
        x=probas,
        y=uncertainties,
        fill=True,
        cmap='viridis',
        ax=ax1
    )
    ax1.scatter(probas, uncertainties, alpha=0.3, color='white', s=10)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Uncertainty')
    ax1.set_title('Prediction Uncertainty vs Probability')
    
    # Histogram with KDE
    sns.histplot(
        uncertainties,
        bins=50,
        kde=True,
        ax=ax2,
        color='skyblue',
        alpha=0.7
    )
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Prediction Uncertainties')
    
    plt.tight_layout()

def plot_calibration_curve(y_true: np.ndarray, 
                          y_prob: np.ndarray,
                          n_bins: int = 10,
                          figsize: tuple = (8, 8)) -> None:
    """Plot calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        figsize: Figure size (width, height)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=figsize)
    
    # Plot calibration curve
    plt.plot(
        prob_pred,
        prob_true,
        marker='o',
        linewidth=2,
        label='Model Calibration'
    )
    
    # Plot perfect calibration line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='gray',
        label='Perfect Calibration'
    )
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_decision_boundary(probas: np.ndarray,
                         uncertainties: np.ndarray,
                         thresholds: dict,
                         figsize: tuple = (10, 6)) -> None:
    """Plot decision boundaries with uncertainty regions.
    
    Args:
        probas: Predicted probabilities
        uncertainties: Prediction uncertainties
        thresholds: Dictionary with screening and confirmation thresholds
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot with density
    sns.kdeplot(
        x=probas,
        y=uncertainties,
        fill=True,
        cmap='viridis',
        alpha=0.5
    )
    sc = plt.scatter(
        probas,
        uncertainties,
        c=probas,
        cmap='viridis',
        alpha=0.3,
        s=30
    )
    
    # Add threshold lines
    plt.axvline(
        x=thresholds['screening'],
        color='r',
        linestyle='--',
        label='Screening Threshold',
        alpha=0.7
    )
    plt.axvline(
        x=thresholds['confirmation'],
        color='g',
        linestyle='--',
        label='Confirmation Threshold',
        alpha=0.7
    )
    
    plt.colorbar(sc, label='Prediction Probability')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Uncertainty')
    plt.title('Decision Boundaries with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def create_interactive_dashboard(probas: np.ndarray,
                               uncertainties: np.ndarray,
                               y_true: np.ndarray,
                               thresholds: dict) -> None:
    """Create interactive Plotly dashboard.
    
    Args:
        probas: Predicted probabilities
        uncertainties: Prediction uncertainties
        y_true: True labels (optional for prediction mode)
        thresholds: Dictionary with screening and confirmation thresholds
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Prediction Uncertainty vs Probability',
            'Distribution of Uncertainties',
            'Calibration Curve',
            'Decision Boundaries'
        )
    )
    
    # 1. Scatter plot with density
    fig.add_trace(
        go.Histogram2dContour(
            x=probas,
            y=uncertainties,
            colorscale='Viridis',
            showscale=False,
            name='Density'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=probas,
            y=uncertainties,
            mode='markers',
            marker=dict(
                color=probas,
                colorscale='Viridis',
                showscale=True,
                size=5,
                opacity=0.5
            ),
            name='Predictions'
        ),
        row=1, col=1
    )
    
    # 2. Uncertainty distribution
    fig.add_trace(
        go.Histogram(
            x=uncertainties,
            nbinsx=50,
            name='Uncertainty Distribution',
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=1, col=2
    )
    
    # 3. Calibration curve (if y_true is provided)
    if y_true is not None:
        prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=10)
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode='lines+markers',
                name='Calibration Curve',
                line=dict(color='rgb(0, 150, 255)', width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Calibration'
            ),
            row=2, col=1
        )
    
    # 4. Decision boundaries with density
    fig.add_trace(
        go.Histogram2dContour(
            x=probas,
            y=uncertainties,
            colorscale='Viridis',
            showscale=False,
            name='Density'
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=probas,
            y=uncertainties,
            mode='markers',
            marker=dict(
                color=probas,
                colorscale='Viridis',
                showscale=True,
                size=5,
                opacity=0.5
            ),
            name='Decision Regions'
        ),
        row=2, col=2
    )
    
    # Add threshold lines
    for threshold, color in zip(
        ['screening', 'confirmation'],
        ['red', 'green']
    ):
        fig.add_vline(
            x=thresholds[threshold],
            line_color=color,
            line_dash='dash',
            row=2,
            col=2,
            annotation_text=f"{threshold.title()} Threshold",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        title_text='Diabetes Prediction Model Analysis',
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Predicted Probability', row=1, col=1)
    fig.update_yaxes(title_text='Uncertainty', row=1, col=1)
    
    fig.update_xaxes(title_text='Uncertainty', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    
    if y_true is not None:
        fig.update_xaxes(title_text='Mean Predicted Probability', row=2, col=1)
        fig.update_yaxes(title_text='Fraction of Positives', row=2, col=1)
    
    fig.update_xaxes(title_text='Predicted Probability', row=2, col=2)
    fig.update_yaxes(title_text='Uncertainty', row=2, col=2)
    
    # Show figure
    fig.show()

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

def plot_class_distribution(y_train, y_train_resampled, title="Class Distribution Before and After Resampling"):
    """Plot class distribution before and after resampling."""
    plt.figure(figsize=(10, 5))
    
    # Original distribution
    plt.subplot(121)
    sns.countplot(x=y_train)
    plt.title('Original Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Resampled distribution
    plt.subplot(122)
    sns.countplot(x=y_train_resampled)
    plt.title('Resampled Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
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