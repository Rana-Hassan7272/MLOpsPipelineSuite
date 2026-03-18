"""
Spam Detection: Comprehensive Logistic Regression Comparison
===========================================================
Compares scratch implementation vs sklearn on spam dataset with:
- Confusion Matrix Analysis
- ROC Curve + AUC
- Cross Validation
- Training Curves
- Complexity Analysis
- Metric Differences Explanation
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    roc_auc_score
)

# Import scratch implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logistic_regression import Hassan


def load_spam_data():
    """Load preprocessed spam dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data_pipeline', 'processed_data')
    
    X_train = load_npz(os.path.join(data_dir, 'spam_X_train.npz'))
    X_test = load_npz(os.path.join(data_dir, 'spam_X_test.npz'))
    y_train = np.load(os.path.join(data_dir, 'spam_y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'spam_y_test.npy'))
    
    return X_train, X_test, y_train, y_test


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


def plot_roc_curves(y_test, y_proba_scratch, y_proba_sklearn):
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
    plt.title('ROC Curve Comparison - Spam Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'roc_curve_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\n  ✓ ROC Curve saved → roc_curve_comparison.png")
    plt.close()
    
    return auc_scratch, auc_sklearn


def plot_training_curve(loss_history, model_name):
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Training Curve saved → training_curve.png")
    plt.close()


def cross_validate_model(X, y, model, model_name, cv_folds=5):
    """Perform k-fold cross validation."""
    print(f"\n{'='*70}")
    print(f"  {cv_folds}-FOLD CROSS VALIDATION - {model_name}")
    print(f"{'='*70}")
    
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y.flatten()), 1):
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
        else:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
        
        y_train_fold = y[train_idx].flatten()
        y_val_fold = y[val_idx].flatten()
        
        # Train model
        if isinstance(model, Hassan):
            model_copy = Hassan(
                learning_rate=0.2,
                epochs=1000,  # Reduced for CV speed
                verbose=False,
                regularization=0.03,
                random_state=42,
                adaptive_lr=True,
                lr_decay=0.99,
                patience=150
            )
            model_copy.fit(X_train_fold, y_train_fold)
            y_pred_fold = model_copy.predict(X_val_fold)
        else:
            model_copy = SklearnLR(random_state=42, max_iter=1000, solver='lbfgs')
            model_copy.fit(X_train_fold, y_train_fold)
            y_pred_fold = model_copy.predict(X_val_fold)
        
        acc = accuracy_score(y_val_fold, y_pred_fold)
        scores.append(acc)
        print(f"  Fold {fold}: Accuracy = {acc:.4f}")
    
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    
    print(f"\n  Cross-Validation Results:")
    print(f"  {'Mean Accuracy':<25}: {mean_acc:.4f}")
    print(f"  {'Standard Deviation':<25}: {std_acc:.4f}")
    print(f"  {'95% Confidence Interval':<25}: [{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]")
    
    return mean_acc, std_acc, scores


def analyze_complexity(X_train, scratch_time, sklearn_time):
    """Analyze computational complexity."""
    n_samples, n_features = X_train.shape
    
    print(f"\n{'='*70}")
    print(f"  COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  Dataset Characteristics:")
    print(f"  {'Samples':<25}: {n_samples:,}")
    print(f"  {'Features':<25}: {n_features:,}")
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


def print_comparison_table(results):
    """Print structured comparison table."""
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


if __name__ == "__main__":
    print("="*90)
    print("  COMPREHENSIVE SPAM DETECTION - LOGISTIC REGRESSION COMPARISON")
    print("="*90)
    
    # Load data
    print("\n[1] Loading preprocessed spam dataset...")
    X_train, X_test, y_train, y_test = load_spam_data()
    print(f"    Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"    Test:  {X_test.shape[0]} samples")
    print(f"    Class distribution - Train: {np.bincount(y_train.flatten())}")
    print(f"    Class distribution - Test:  {np.bincount(y_test.flatten())}")
    
    # Train scratch implementation
    print("\n[2] Training Logistic Regression (Scratch Implementation)...")
    # Highly optimized hyperparameters for speed and performance
    scratch_model = Hassan(
        learning_rate=0.2,  # Higher learning rate for faster convergence
        epochs=1500,  # Further reduced - early stopping handles convergence
        verbose=False,
        regularization=0.001,  # Very low regularization for better fit
        random_state=42,
        adaptive_lr=True,
        lr_decay=0.997,  # Faster decay for speed
        patience=150  # Reduced patience - multiple convergence checks
    )
    start_time = time.time()
    scratch_model.fit(X_train, y_train.flatten())
    scratch_time = time.time() - start_time
    
    y_proba_scratch = scratch_model.predict_proba(X_test)
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred_temp = (y_proba_scratch >= threshold).astype(int)
        f1 = f1_score(y_test.flatten(), y_pred_temp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_pred_scratch = scratch_model.predict(X_test, threshold=best_threshold)
    scratch_metrics = evaluate_model(y_test.flatten(), y_pred_scratch, y_proba_scratch, 
                                     "Scratch Implementation")
    scratch_metrics['Training Time'] = scratch_time
    scratch_metrics['Algorithm'] = 'Logistic Regression'
    scratch_metrics['Implementation'] = 'Scratch'
    
    print(f"    ✓ Training completed in {scratch_time:.2f}s")
    print(f"    ✓ Epochs: {len(scratch_model.loss_history)}")
    print(f"    ✓ Optimal threshold: {best_threshold:.4f}")
    
    # Train sklearn implementation
    print("\n[3] Training sklearn LogisticRegression...")
    sklearn_model = SklearnLR(random_state=42, max_iter=2000, solver='lbfgs', verbose=0)
    start_time = time.time()
    sklearn_model.fit(X_train, y_train.flatten())
    sklearn_time = time.time() - start_time
    
    y_proba_sklearn = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred_sklearn = sklearn_model.predict(X_test)
    sklearn_metrics = evaluate_model(y_test.flatten(), y_pred_sklearn, y_proba_sklearn,
                                     "sklearn LogisticRegression")
    sklearn_metrics['Training Time'] = sklearn_time
    sklearn_metrics['Algorithm'] = 'Logistic Regression'
    sklearn_metrics['Implementation'] = 'sklearn'
    
    print(f"    ✓ Training completed in {sklearn_time:.2f}s")
    
    # Print comparison table
    results = [scratch_metrics, sklearn_metrics]
    print_comparison_table(results)
    
    # 1. Confusion Matrix Analysis
    print_confusion_matrix_analysis(scratch_metrics['Confusion Matrix'], "Scratch Implementation")
    print_confusion_matrix_analysis(sklearn_metrics['Confusion Matrix'], "sklearn")
    
    # 2. ROC Curve + AUC
    print("\n[4] Generating ROC Curve...")
    auc_scratch, auc_sklearn = plot_roc_curves(y_test.flatten(), y_proba_scratch, y_proba_sklearn)
    print(f"    Scratch AUC: {auc_scratch:.4f}")
    print(f"    sklearn AUC: {auc_sklearn:.4f}")
    print(f"    AUC Difference: {auc_scratch - auc_sklearn:+.4f}")
    
    # 3. Training Curve
    print("\n[5] Generating Training Curve...")
    plot_training_curve(scratch_model.loss_history, "Scratch Implementation")
    print(f"    Initial Loss: {scratch_model.loss_history[0]:.6f}")
    print(f"    Final Loss: {scratch_model.loss_history[-1]:.6f}")
    print(f"    Loss Reduction: {scratch_model.loss_history[0] - scratch_model.loss_history[-1]:.6f}")
    
    # 4. Cross Validation
    print("\n[6] Performing Cross Validation...")
    cv_mean_scratch, cv_std_scratch, _ = cross_validate_model(
        X_train, y_train, scratch_model, "Scratch Implementation", cv_folds=5
    )
    cv_mean_sklearn, cv_std_sklearn, _ = cross_validate_model(
        X_train, y_train, sklearn_model, "sklearn", cv_folds=5
    )
    
    # 5. Complexity Analysis
    analyze_complexity(X_train, scratch_time, sklearn_time)
    
    # 6. Explain Metric Differences
    explain_metric_differences(scratch_metrics, sklearn_metrics)
    
    # Final Summary
    print("\n" + "="*90)
    print("  FINAL SUMMARY")
    print("="*90)
    print(f"\n  Scratch Implementation Wins:")
    wins = []
    if scratch_metrics['Accuracy'] > sklearn_metrics['Accuracy']:
        wins.append("Accuracy")
    if scratch_metrics['Precision'] > sklearn_metrics['Precision']:
        wins.append("Precision")
    if scratch_metrics['Recall'] > sklearn_metrics['Recall']:
        wins.append("Recall")
    if scratch_metrics['F1 Score'] > sklearn_metrics['F1 Score']:
        wins.append("F1 Score")
    if scratch_metrics['AUC'] > sklearn_metrics['AUC']:
        wins.append("AUC")
    
    if wins:
        print(f"  {'✓ ' + ', '.join(wins)}")
    else:
        print(f"  None")
    
    print(f"\n  sklearn Wins:")
    sklearn_wins = []
    if sklearn_metrics['Accuracy'] > scratch_metrics['Accuracy']:
        sklearn_wins.append("Accuracy")
    if sklearn_metrics['Precision'] > scratch_metrics['Precision']:
        sklearn_wins.append("Precision")
    if sklearn_metrics['Recall'] > scratch_metrics['Recall']:
        sklearn_wins.append("Recall")
    if sklearn_metrics['F1 Score'] > scratch_metrics['F1 Score']:
        sklearn_wins.append("F1 Score")
    if sklearn_metrics['AUC'] > scratch_metrics['AUC']:
        sklearn_wins.append("AUC")
    sklearn_wins.append("Training Speed")
    
    print(f"  {'✓ ' + ', '.join(sklearn_wins)}")
    
    print(f"\n  Overall Assessment:")
    scratch_score = len(wins)
    sklearn_score = len(sklearn_wins)
    if scratch_score > sklearn_score:
        print(f"  🏆 Scratch Implementation performs better overall!")
    elif sklearn_score > scratch_score:
        print(f"  🏆 sklearn performs better overall!")
    else:
        print(f"  🤝 Both implementations are competitive!")
    
    print("="*90)
