"""
Evaluation Metrics Module
=========================
Provides comprehensive metrics calculation and visualization functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Compute and return comprehensive metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_proba),
        'Confusion Matrix': cm,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }


def print_confusion_matrix_analysis(cm, model_name):
    """Print detailed confusion matrix analysis."""
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print(f"\n{'='*70}")
    print(f"  CONFUSION MATRIX ANALYSIS - {model_name}")
    print(f"{'='*70}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>15} {'Predicted 0':>15} {'Predicted 1':>15}")
    print(f"  {'Actual 0':>15} {tn:>15} {fp:>15}")
    print(f"  {'Actual 1':>15} {fn:>15} {tp:>15}")
    
    print(f"\n  Detailed Breakdown:")
    print(f"  {'True Negatives (TN)':<30}: {tn:>5} - Correctly predicted non-spam")
    print(f"  {'False Positives (FP)':<30}: {fp:>5} - Non-spam misclassified as spam (Type I Error)")
    print(f"  {'False Negatives (FN)':<30}: {fn:>5} - Spam misclassified as non-spam (Type II Error)")
    print(f"  {'True Positives (TP)':<30}: {tp:>5} - Correctly predicted spam")
    
    print(f"\n  Error Analysis:")
    print(f"  {'False Positive Rate':<30}: {fp/(fp+tn):.4f} ({fp}/{fp+tn})")
    print(f"  {'False Negative Rate':<30}: {fn/(fn+tp):.4f} ({fn}/{fn+tp})")
    print(f"  {'Total Errors':<30}: {fp+fn}/{total} ({100*(fp+fn)/total:.2f}%)")


def plot_roc_curves(y_test, y_proba_scratch, y_proba_sklearn, output_dir=None, title_suffix="Spam Detection"):
    """Plot ROC curves for both models."""
    fpr_scratch, tpr_scratch, _ = roc_curve(y_test, y_proba_scratch)
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_proba_sklearn)
    
    auc_scratch = auc(fpr_scratch, tpr_scratch)
    auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_scratch, tpr_scratch, label=f'Scratch (AUC = {auc_scratch:.4f})', 
             linewidth=2, color='#00e5ff')
    plt.plot(fpr_sklearn, tpr_sklearn, label=f'sklearn (AUC = {auc_sklearn:.4f})', 
             linewidth=2, color='#ff6b35', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve Comparison - {title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, 'roc_curve_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ ROC Curve saved → {output_path}")
    plt.close()
    
    return auc_scratch, auc_sklearn


def plot_pr_curve(y_true, scratch_scores, sklearn_scores, scratch_ap, sklearn_ap, output_dir=None):
    """Plot Precision-Recall curves for both models."""
    plt.figure(figsize=(10, 8))
    
    # Scratch PR
    prec_scratch, rec_scratch, _ = precision_recall_curve(y_true, scratch_scores)
    plt.plot(rec_scratch, prec_scratch, color='#00e5ff', linewidth=2.5,
             label=f'Scratch Implementation (AP = {scratch_ap:.4f})')
    
    # sklearn PR
    prec_sklearn, rec_sklearn, _ = precision_recall_curve(y_true, sklearn_scores)
    plt.plot(rec_sklearn, prec_sklearn, color='#ff6b35', linewidth=2.5,
             label=f'sklearn (AP = {sklearn_ap:.4f})')
    
    # Baseline (proportion of positives)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                alpha=0.5, label=f'Baseline (AP = {baseline:.4f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Isolation Forest Comparison', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, 'pr_curve_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ PR Curve saved → {output_path}")
    plt.close()


def plot_score_distributions(y_true, scratch_scores, sklearn_scores, 
                            scratch_threshold, sklearn_threshold, output_dir=None):
    """Plot score distributions for normal vs fraud."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scratch scores
    normal_scores = scratch_scores[y_true == 0]
    fraud_scores = scratch_scores[y_true == 1]
    
    ax1.hist(normal_scores, bins=60, alpha=0.6, color='#00e5ff', 
             label='Normal Transactions', density=True, edgecolor='none')
    ax1.hist(fraud_scores, bins=60, alpha=0.8, color='#ff2d78', 
             label='Fraud Transactions', density=True, edgecolor='none')
    ax1.axvline(scratch_threshold, color='#39ff14', linestyle='--', 
                linewidth=2, label=f'Threshold = {scratch_threshold:.4f}')
    ax1.set_xlabel('Anomaly Score', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Scratch Implementation - Score Distribution', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # sklearn scores
    normal_scores_sk = sklearn_scores[y_true == 0]
    fraud_scores_sk = sklearn_scores[y_true == 1]
    
    ax2.hist(normal_scores_sk, bins=60, alpha=0.6, color='#ff6b35', 
             label='Normal Transactions', density=True, edgecolor='none')
    ax2.hist(fraud_scores_sk, bins=60, alpha=0.8, color='#ff2d78', 
             label='Fraud Transactions', density=True, edgecolor='none')
    ax2.axvline(sklearn_threshold, color='#39ff14', linestyle='--', 
                linewidth=2, label=f'Threshold = {sklearn_threshold:.4f}')
    ax2.set_xlabel('Anomaly Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('sklearn - Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Score Distributions saved → {output_path}")
    plt.close()


def print_confusion_matrix_analysis_fraud(y_true, y_pred, model_name):
    """Print detailed confusion matrix analysis for fraud detection."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*70}")
    print(f"  CONFUSION MATRIX ANALYSIS - {model_name}")
    print(f"{'='*70}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>20} {'Predicted Normal':>20} {'Predicted Fraud':>20}")
    print(f"  {'Actual Normal':>20} {tn:>20} {fp:>20}")
    print(f"  {'Actual Fraud':>20} {fn:>20} {tp:>20}")
    
    print(f"\n  Detailed Breakdown:")
    print(f"  {'True Negatives (TN)':<30}: {tn:>6} - Correctly predicted normal transactions")
    print(f"  {'False Positives (FP)':<30}: {fp:>6} - Normal transactions flagged as fraud (Type I Error)")
    print(f"  {'False Negatives (FN)':<30}: {fn:>6} - Fraud transactions missed (Type II Error)")
    print(f"  {'True Positives (TP)':<30}: {tp:>6} - Correctly detected fraud")
    
    total = tn + fp + fn + tp
    print(f"\n  Error Analysis:")
    print(f"  {'False Positive Rate':<30}: {fp/(tn+fp):>6.4f} ({fp}/{tn+fp})")
    print(f"  {'False Negative Rate':<30}: {fn/(fn+tp):>6.4f} ({fn}/{fn+tp})")
    print(f"  {'Total Errors':<30}: {fp+fn:>6}/{total} ({100*(fp+fn)/total:.2f}%)")
    print(f"  {'Accuracy':<30}: {(tn+tp)/total:>6.4f}")


def analyze_complexity_isolation_forest(X, scratch_time, sklearn_time):
    """Analyze computational complexity for Isolation Forest."""
    n_samples, n_features = X.shape
    
    print(f"\n{'='*70}")
    print(f"  COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Dataset Characteristics:")
    print(f"  {'Samples':<25}: {n_samples:,}")
    print(f"  {'Features':<25}: {n_features:,}")
    
    print(f"\n  Time Complexity:")
    print(f"  {'Scratch Implementation':<25}: O(n_estimators × max_samples × log(max_samples))")
    print(f"  {'sklearn IsolationForest':<25}: O(n_estimators × max_samples × log(max_samples))")
    print(f"  {'Note':<25}: Both use Isolation Tree construction")
    
    print(f"\n  Training Time Comparison:")
    print(f"  {'Scratch':<25}: {scratch_time:.4f}s")
    print(f"  {'sklearn':<25}: {sklearn_time:.4f}s")
    speedup = sklearn_time / scratch_time if scratch_time > 0 else 0
    print(f"  {'Speedup Factor':<25}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    print(f"\n  Space Complexity:")
    print(f"  {'Scratch':<25}: O(n_estimators × max_nodes × features)")
    print(f"  {'sklearn':<25}: O(n_estimators × max_nodes × features)")
    print(f"  {'Memory Usage':<25}: Similar for both implementations")
    
    print(f"\n  Optimization Techniques:")
    print(f"  {'Scratch':<25}: Parallel tree building, vectorized scoring, flat tree storage")
    print(f"  {'sklearn':<25}: Cython-optimized, BLAS acceleration, single-threaded by default")


def explain_metric_differences_isolation_forest(scratch_metrics, sklearn_metrics):
    """Explain why metrics differ between Isolation Forest implementations."""
    print(f"\n{'='*70}")
    print(f"  METRIC DIFFERENCES ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Key Differences:")
    print(f"  {'Metric':<25} {'Scratch':<15} {'sklearn':<15} {'Difference':<15}")
    print(f"  {'-'*70}")
    for metric in ['ROC-AUC', 'Avg Precision', 'Precision', 'Recall', 'F1', 'Time (s)']:
        scratch_val = scratch_metrics.get(metric, np.nan)
        sklearn_val = sklearn_metrics.get(metric, np.nan)
        if not np.isnan(scratch_val) and not np.isnan(sklearn_val):
            diff = scratch_val - sklearn_val
            print(f"  {metric:<25} {scratch_val:<15.4f} {sklearn_val:<15.4f} {diff:+.4f}")
    
    print(f"\n  Possible Reasons for Differences:")
    print(f"  1. Split Selection:")
    print(f"     - Scratch: Extended Isolation Forest (hyperplane cuts)")
    print(f"     - sklearn: Classic axis-aligned splits")
    print(f"     - Extended splits reduce axis-aligned bias, improving detection")
    
    print(f"\n  2. Parallelization:")
    print(f"     - Scratch: Parallel tree building via joblib")
    print(f"     - sklearn: Single-threaded by default")
    print(f"     - Parallelization provides speed advantages")
    
    print(f"\n  3. Scoring Method:")
    print(f"     - Scratch: Vectorized batch scoring")
    print(f"     - sklearn: Optimized Cython implementation")
    print(f"     - Different implementations may converge to slightly different scores")
    
    print(f"\n  4. Threshold Selection:")
    print(f"     - Scratch: Auto-threshold with valley detection")
    print(f"     - sklearn: Percentile-based threshold")
    print(f"     - Different threshold methods affect precision/recall tradeoff")
    
    print(f"\n  5. Numerical Precision:")
    print(f"     - Scratch: NumPy float64 operations")
    print(f"     - sklearn: Cython with potential precision differences")
    print(f"     - Slight numerical differences in tree construction")


def plot_training_curve(loss_history, model_name, output_dir=None):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history, linewidth=2, color='#00e5ff', label='Training Loss')
    plt.fill_between(epochs, loss_history, alpha=0.2, color='#00e5ff')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.title(f'Training Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Training Curve saved → {output_path}")
    plt.close()


def analyze_complexity(X_train, scratch_time, sklearn_time):
    """Analyze computational complexity."""
    n_samples, n_features = X_train.shape
    
    print(f"\n{'='*70}")
    print(f"  COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Dataset Characteristics:")
    print(f"  {'Samples':<25}: {n_samples:,}")
    print(f"  {'Features':<25}: {n_features:,}")
    if hasattr(X_train, 'nnz'):
        print(f"  {'Sparsity':<25}: {(1 - X_train.nnz / (n_samples * n_features)) * 100:.2f}%")
    
    print(f"\n  Time Complexity:")
    print(f"  {'Scratch Implementation':<25}: O(epochs × n_samples × n_features)")
    print(f"  {'sklearn (LBFGS)':<25}: O(iterations × n_samples × n_features)")
    print(f"  {'Note':<25}: sklearn uses optimized BLAS/LAPACK libraries")
    
    print(f"\n  Training Time Comparison:")
    print(f"  {'Scratch':<25}: {scratch_time:.4f}s")
    print(f"  {'sklearn':<25}: {sklearn_time:.4f}s")
    print(f"  {'Speedup Factor':<25}: {scratch_time/sklearn_time:.2f}x slower")
    
    print(f"\n  Space Complexity:")
    print(f"  {'Scratch':<25}: O(n_features) - stores weights + bias")
    print(f"  {'sklearn':<25}: O(n_features) - stores coefficients + intercept")
    print(f"  {'Memory Usage':<25}: Similar for both implementations")


def explain_metric_differences(scratch_metrics, sklearn_metrics):
    """Explain why metrics differ between implementations."""
    print(f"\n{'='*70}")
    print(f"  METRIC DIFFERENCES ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Key Differences:")
    print(f"  {'Metric':<20} {'Scratch':<15} {'sklearn':<15} {'Difference':<15}")
    print(f"  {'-'*65}")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']:
        scratch_val = scratch_metrics.get(metric, 0)
        sklearn_val = sklearn_metrics.get(metric, 0)
        diff = scratch_val - sklearn_val
        print(f"  {metric:<20} {scratch_val:<15.4f} {sklearn_val:<15.4f} {diff:+.4f}")
    
    print(f"\n  Possible Reasons for Differences:")
    print(f"  1. Regularization:")
    print(f"     - Scratch uses L2 regularization (λ = 0.05)")
    print(f"     - sklearn uses default regularization (C = 1.0, λ = 1/C = 1.0)")
    print(f"     - Different regularization strengths affect model complexity")
    
    print(f"\n  2. Optimization Method:")
    print(f"     - Scratch: Batch Gradient Descent with adaptive learning rate")
    print(f"     - sklearn: LBFGS (Limited-memory BFGS) - quasi-Newton method")
    print(f"     - LBFGS converges faster but may find different local minima")
    
    print(f"\n  3. Convergence Tolerance:")
    print(f"     - Scratch: Early stopping with patience (150 epochs)")
    print(f"     - sklearn: Convergence tolerance based on gradient norm")
    print(f"     - Different stopping criteria lead to different convergence points")
    
    print(f"\n  4. Numerical Precision:")
    print(f"     - Scratch: NumPy float64 operations")
    print(f"     - sklearn: Optimized BLAS/LAPACK with potential precision differences")
    print(f"     - Sparse matrix operations may have slight numerical differences")
    
    print(f"\n  5. Class Weight Handling:")
    print(f"     - Scratch: Automatic inverse frequency weighting")
    print(f"     - sklearn: Default balanced='auto' or equal weights")
    print(f"     - Different class weighting affects decision boundaries")
    
    print(f"\n  6. Initialization:")
    print(f"     - Scratch: Xavier/Glorot initialization with bias adjustment")
    print(f"     - sklearn: Default initialization (usually zeros)")
    print(f"     - Better initialization can lead to better convergence")


def print_comparison_table(results, algorithm_type='classification'):
    """Print structured comparison table."""
    if algorithm_type == 'classification':
        print("\n" + "="*90)
        print("  LOGISTIC REGRESSION COMPARISON - SPAM DETECTION")
        print("="*90)
        print(f"\n{'Algorithm':<25} {'Implementation':<15} {'Accuracy':<12} {'Precision':<12} "
              f"{'Recall':<12} {'F1 Score':<12} {'AUC':<12} {'Training Time':<15}")
        print("-"*90)
        
        for result in results:
            print(f"{result['Algorithm']:<25} {result['Implementation']:<15} "
                  f"{result['Accuracy']:<12.4f} {result['Precision']:<12.4f} "
                  f"{result['Recall']:<12.4f} {result['F1 Score']:<12.4f} "
                  f"{result['AUC']:<12.4f} {result['Training Time']:<15.2f}s")
        print("="*90)
    elif algorithm_type == 'clustering':
        print("\n" + "="*90)
        print("  K-MEANS CLUSTERING COMPARISON - STORE CUSTOMERS")
        print("="*90)
        print(f"\n{'Algorithm':<25} {'Implementation':<15} {'Inertia':<15} {'Silhouette':<15} "
              f"{'Davies-Bouldin':<18} {'Calinski-Harabasz':<20} {'Training Time':<15}")
        print("-"*90)
        
        for result in results:
            print(f"{result['Algorithm']:<25} {result['Implementation']:<15} "
                  f"{result.get('Inertia', 0):<15.2f} {result.get('Silhouette', 0):<15.4f} "
                  f"{result.get('Davies-Bouldin', 0):<18.4f} {result.get('Calinski-Harabasz', 0):<20.2f} "
                  f"{result.get('Time (s)', 0):<15.4f}s")
        print("="*90)
    elif algorithm_type == 'anomaly_detection':
        print("\n" + "="*90)
        print("  ISOLATION FOREST COMPARISON - CREDIT CARD FRAUD DETECTION")
        print("="*90)
        print(f"\n{'Algorithm':<25} {'Implementation':<15} {'ROC-AUC':<12} {'Avg Precision':<15} "
              f"{'Precision':<12} {'Recall':<12} {'F1':<12} {'Training Time':<15}")
        print("-"*90)
        
        for result in results:
            print(f"{result['Algorithm']:<25} {result['Implementation']:<15} "
                  f"{result.get('ROC-AUC', 0):<12.4f} {result.get('Avg Precision', 0):<15.4f} "
                  f"{result.get('Precision', 0):<12.4f} {result.get('Recall', 0):<12.4f} "
                  f"{result.get('F1', 0):<12.4f} {result.get('Time (s)', 0):<15.4f}s")
        print("="*90)


# ─────────────────────────────────────────────────────────────────────────────
#  Clustering-specific metrics
# ─────────────────────────────────────────────────────────────────────────────

def print_cluster_analysis(labels, centroids, model_name, X, output_dir=None):
    """Print detailed cluster analysis."""
    k = len(centroids)
    print(f"\n{'='*70}")
    print(f"  CLUSTER ANALYSIS - {model_name}")
    print(f"{'='*70}")
    
    print(f"\n  Cluster Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"    Cluster {cluster_id}: {count:>4} samples ({percentage:>5.2f}%)")
    
    print(f"\n  Cluster Characteristics:")
    for c in range(k):
        mask = labels == c
        cluster_data = X[mask]
        print(f"\n    Cluster {c}:")
        print(f"      Size: {mask.sum()}")
        print(f"      Centroid: {centroids[c]}")
        if len(cluster_data) > 0:
            print(f"      Mean distance to centroid: {np.mean(np.linalg.norm(cluster_data - centroids[c], axis=1)):.4f}")
            print(f"      Max distance to centroid: {np.max(np.linalg.norm(cluster_data - centroids[c], axis=1)):.4f}")


def plot_elbow_curve(auto_k_stats, model_name, output_dir=None):
    """Plot elbow curve with silhouette scores."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    ks = auto_k_stats["ks"]
    inertias = auto_k_stats["inertias"]
    silhouettes = auto_k_stats["silhouettes"]
    
    ax1 = plt.gca()
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12)
    ax1.plot(ks, inertias, 'o-', color=color, linewidth=2, markersize=8, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
    ax2.plot(ks, silhouettes, 's--', color=color, linewidth=2, markersize=8, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Elbow Curve & Silhouette Analysis - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, 'elbow_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Elbow Curve saved → {output_path}")
    plt.close()


def plot_cluster_visualization(X_2d, labels, centroids_2d, model_name, save_name, output_dir=None):
    """Plot 2D cluster visualization."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    k = len(centroids_2d)
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    for c in range(k):
        mask = labels == c
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[colors[c]], s=50, alpha=0.6, label=f'Cluster {c}')
    
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               c='red', marker='*', s=500, edgecolors='black', 
               linewidths=2, label='Centroids', zorder=10)
    
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title(f'Cluster Visualization - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, save_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Cluster visualization saved → {save_name}")
    plt.close()


def plot_inertia_history(inertia_history, model_name, output_dir=None):
    """Plot inertia across restarts."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    restarts = range(1, len(inertia_history) + 1)
    plt.bar(restarts, inertia_history, alpha=0.7, color='#00e5ff', edgecolor='none')
    best_idx = np.argmin(inertia_history)
    plt.bar(best_idx + 1, inertia_history[best_idx], 
           color='#39ff14', alpha=1.0, edgecolor='none', label='Best restart')
    plt.axhline(min(inertia_history), color='green', linestyle='--', 
               linewidth=2, alpha=0.8, label='Best inertia')
    plt.xlabel('Restart #', fontsize=12)
    plt.ylabel('Inertia (WCSS)', fontsize=12)
    plt.title(f'Inertia Across Restarts - {model_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    if output_dir is None:
        output_dir = os.getcwd()
    output_path = os.path.join(output_dir, 'inertia_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Inertia history saved → inertia_history.png")
    plt.close()


def analyze_complexity_clustering(X, scratch_time, sklearn_time):
    """Analyze computational complexity for clustering."""
    n_samples, n_features = X.shape
    
    print(f"\n{'='*70}")
    print(f"  COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Dataset Characteristics:")
    print(f"  {'Samples':<25}: {n_samples:,}")
    print(f"  {'Features':<25}: {n_features:,}")
    
    print(f"\n  Time Complexity:")
    print(f"  {'Scratch Implementation':<25}: O(n_init × max_iter × n_samples × n_features)")
    print(f"  {'sklearn KMeans':<25}: O(n_init × max_iter × n_samples × n_features)")
    print(f"  {'Note':<25}: Both use KMeans++ initialization")
    
    print(f"\n  Training Time Comparison:")
    print(f"  {'Scratch':<25}: {scratch_time:.4f}s")
    print(f"  {'sklearn':<25}: {sklearn_time:.4f}s")
    print(f"  {'Speedup Factor':<25}: {scratch_time/sklearn_time:.2f}x slower")
    
    print(f"\n  Space Complexity:")
    print(f"  {'Scratch':<25}: O(n_features × k) - stores centroids")
    print(f"  {'sklearn':<25}: O(n_features × k) - stores cluster_centers_")
    print(f"  {'Memory Usage':<25}: Similar for both implementations")
    
    print(f"\n  Optimization Techniques:")
    print(f"  {'Scratch':<25}: Vectorized distance computation, KMeans++ init")
    print(f"  {'sklearn':<25}: Optimized BLAS/LAPACK, Cython acceleration")


def explain_metric_differences_clustering(scratch_metrics, sklearn_metrics):
    """Explain why clustering metrics differ between implementations."""
    print(f"\n{'='*70}")
    print(f"  METRIC DIFFERENCES ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Key Differences:")
    print(f"  {'Metric':<25} {'Scratch':<15} {'sklearn':<15} {'Difference':<15}")
    print(f"  {'-'*70}")
    for metric in ['Inertia', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Time (s)']:
        scratch_val = scratch_metrics.get(metric, np.nan)
        sklearn_val = sklearn_metrics.get(metric, np.nan)
        if not np.isnan(scratch_val) and not np.isnan(sklearn_val):
            diff = scratch_val - sklearn_val
            print(f"  {metric:<25} {scratch_val:<15.4f} {sklearn_val:<15.4f} {diff:+.4f}")
    
    print(f"\n  Possible Reasons for Differences:")
    print(f"  1. Initialization:")
    print(f"     - Both use KMeans++ but random seed differences affect starting points")
    print(f"     - Different random number generators may produce different initializations")
    
    print(f"\n  2. Convergence Criteria:")
    print(f"     - Scratch: Checks both label stability AND centroid drift")
    print(f"     - sklearn: Primarily checks centroid drift")
    print(f"     - Different convergence checks lead to different stopping points")
    
    print(f"\n  3. Empty Cluster Handling:")
    print(f"     - Scratch: Rescues from highest-error region")
    print(f"     - sklearn: Uses different empty cluster handling strategy")
    print(f"     - Different strategies affect final cluster assignments")
    
    print(f"\n  4. Numerical Precision:")
    print(f"     - Scratch: NumPy float64 operations")
    print(f"     - sklearn: Optimized BLAS/LAPACK with potential precision differences")
    print(f"     - Slight numerical differences accumulate over iterations")
    
    print(f"\n  5. Multi-restart Selection:")
    print(f"     - Both select best restart based on inertia")
    print(f"     - Different random seeds in restarts lead to different best solutions")
    
    print(f"\n  6. Optimization Level:")
    print(f"     - Scratch: Pure Python/NumPy implementation")
    print(f"     - sklearn: Cython-optimized with BLAS acceleration")
    print(f"     - sklearn's lower-level optimizations provide speed advantages")
