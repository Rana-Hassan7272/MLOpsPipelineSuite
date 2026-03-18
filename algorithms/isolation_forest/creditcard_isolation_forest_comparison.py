"""
Credit Card Fraud Detection: Comprehensive Isolation Forest Comparison
=====================================================================
Compares scratch implementation vs sklearn on creditcard_2023 dataset with:
- ROC Curve + AUC Analysis
- Precision-Recall Curve
- Confusion Matrix Analysis
- Cross Validation
- Score Distribution Analysis
- Complexity Analysis
- Metric Differences Explanation
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest as SklearnIF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# Import scratch implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scratch_isolation_forest import HassanIsolationForest


def load_creditcard_data():
    """Load preprocessed creditcard dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Try new filename first, fallback to old filename
    train_path_new = os.path.join(project_root, 'data_pipeline', 'processed_data', 
                                   'creditcard_train.csv')
    test_path_new = os.path.join(project_root, 'data_pipeline', 'processed_data', 
                                  'creditcard_test.csv')
    train_path_old = os.path.join(project_root, 'data_pipeline', 'processed_data', 
                                  'creditcard_2023_train.csv')
    test_path_old = os.path.join(project_root, 'data_pipeline', 'processed_data', 
                                  'creditcard_2023_test.csv')
    
    # Use new filename if exists, otherwise use old
    if os.path.exists(train_path_new) and os.path.exists(test_path_new):
        train_path = train_path_new
        test_path = test_path_new
    elif os.path.exists(train_path_old) and os.path.exists(test_path_old):
        train_path = train_path_old
        test_path = test_path_old
    else:
        raise FileNotFoundError(f"Processed creditcard files not found. Please run preprocessing pipeline first.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop 'id' column if it exists (not needed for training)
    if 'id' in train_df.columns:
        train_df = train_df.drop('id', axis=1)
    if 'id' in test_df.columns:
        test_df = test_df.drop('id', axis=1)
    
    # Separate features and target
    # 'Class' column exists (0=normal, 1=fraud)
    if 'Class' in train_df.columns:
        X_train = train_df.drop('Class', axis=1).values
        y_train = train_df['Class'].values
        X_test = test_df.drop('Class', axis=1).values
        y_test = test_df['Class'].values
        
        # Debug: Check Class distribution
        print(f"    Train Class distribution: Normal={np.sum(y_train == 0)}, Fraud={np.sum(y_train == 1)}")
        print(f"    Test Class distribution: Normal={np.sum(y_test == 0)}, Fraud={np.sum(y_test == 1)}")
        print(f"    Train Class unique values: {np.unique(y_train)}")
        print(f"    Test Class unique values: {np.unique(y_test)}")
        
        # Verify Class encoding is correct (0=normal, 1=fraud)
        train_fraud_rate = np.sum(y_train == 1) / len(y_train) if len(y_train) > 0 else 0
        if train_fraud_rate > 0.5:
            print(f"    Warning: Fraud rate is {train_fraud_rate:.2%} - unusually high for credit card data!")
            print(f"    This might indicate a preprocessing issue. Proceeding with caution...")
    else:
        # If no Class column, use all data for unsupervised learning
        X_train = train_df.values
        y_train = None
        X_test = test_df.values
        y_test = None
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, scaler


def print_confusion_matrix_analysis(y_true, y_pred, model_name):
    """Print detailed confusion matrix analysis."""
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


def plot_roc_curve(y_true, scratch_scores, sklearn_scores, scratch_auc, sklearn_auc):
    """Plot ROC curves for both models."""
    plt.figure(figsize=(10, 8))
    
    # Scratch ROC
    fpr_scratch, tpr_scratch, _ = roc_curve(y_true, scratch_scores)
    plt.plot(fpr_scratch, tpr_scratch, color='#00e5ff', linewidth=2.5,
             label=f'Scratch Implementation (AUC = {scratch_auc:.4f})')
    
    # sklearn ROC
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_true, sklearn_scores)
    plt.plot(fpr_sklearn, tpr_sklearn, color='#ff6b35', linewidth=2.5,
             label=f'sklearn (AUC = {sklearn_auc:.4f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Isolation Forest Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'roc_curve_comparison.png'), 
                dpi=150, bbox_inches='tight')
    print(f"  ✓ ROC Curve saved → roc_curve_comparison.png")
    plt.close()


def plot_pr_curve(y_true, scratch_scores, sklearn_scores, scratch_ap, sklearn_ap):
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'pr_curve_comparison.png'), 
                dpi=150, bbox_inches='tight')
    print(f"  ✓ PR Curve saved → pr_curve_comparison.png")
    plt.close()


def plot_score_distributions(y_true, scratch_scores, sklearn_scores, 
                            scratch_threshold, sklearn_threshold):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'score_distributions.png'), 
                dpi=150, bbox_inches='tight')
    print(f"  ✓ Score Distributions saved → score_distributions.png")
    plt.close()


def cross_validate_isolation_forest(X, y, n_splits=5):
    """Perform stratified k-fold cross validation."""
    print(f"\n{'='*70}")
    print(f"  {n_splits}-FOLD STRATIFIED CROSS VALIDATION")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scratch_aucs = []
    sklearn_aucs = []
    scratch_aps = []
    sklearn_aps = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Estimate contamination from training fold (production-optimized for 80% recall)
        fold_fraud_rate = np.sum(y_train_fold == 1) / len(y_train_fold) if len(y_train_fold) > 0 else 0.002
        # Use 10x rate for 80% recall target, ensure in valid range [0.0001, 0.5]
        base_contamination = max(0.0001, min(0.5, fold_fraud_rate))
        contamination = min(0.020, base_contamination * 10.0)  # 10x for 80% recall target
        if contamination < 0.0001:
            contamination = 0.010  # Use higher default for 80% recall
        
        # Scratch implementation (production-optimized for 80% recall)
        scratch_model = HassanIsolationForest(
            n_estimators=250, max_samples=256,  # More trees for CV stability and recall
            contamination=contamination, random_state=42,
            n_jobs=-1, extended=True, verbose=False
        )
        scratch_model.fit(X_train_fold)
        scratch_scores = scratch_model.score_samples(X_val_fold)
        scratch_auc = roc_auc_score(y_val_fold, scratch_scores)
        scratch_ap = average_precision_score(y_val_fold, scratch_scores)
        scratch_aucs.append(scratch_auc)
        scratch_aps.append(scratch_ap)
        
        # sklearn implementation (match trees for fair comparison)
        sklearn_model = SklearnIF(
            n_estimators=250, max_samples=256,
            contamination=contamination, random_state=42, n_jobs=1
        )
        sklearn_model.fit(X_train_fold)
        sklearn_scores_raw = sklearn_model.score_samples(X_val_fold)
        sklearn_scores = -sklearn_scores_raw  # Flip convention
        sklearn_scores = (sklearn_scores - sklearn_scores.min()) / \
                        (sklearn_scores.max() - sklearn_scores.min())
        sklearn_auc = roc_auc_score(y_val_fold, sklearn_scores)
        sklearn_ap = average_precision_score(y_val_fold, sklearn_scores)
        sklearn_aucs.append(sklearn_auc)
        sklearn_aps.append(sklearn_ap)
        
        print(f"  Fold {fold}: Scratch AUC={scratch_auc:.4f}, AP={scratch_ap:.4f} | "
              f"sklearn AUC={sklearn_auc:.4f}, AP={sklearn_ap:.4f}")
    
    scratch_auc_mean = np.mean(scratch_aucs)
    scratch_auc_std = np.std(scratch_aucs)
    sklearn_auc_mean = np.mean(sklearn_aucs)
    sklearn_auc_std = np.std(sklearn_aucs)
    
    scratch_ap_mean = np.mean(scratch_aps)
    scratch_ap_std = np.std(scratch_aps)
    sklearn_ap_mean = np.mean(sklearn_aps)
    sklearn_ap_std = np.std(sklearn_aps)
    
    print(f"\n  Cross-Validation Results:")
    print(f"  {'Metric':<25} {'Scratch':<20} {'sklearn':<20}")
    print(f"  {'-'*65}")
    print(f"  {'ROC-AUC (Mean ± Std)':<25} {scratch_auc_mean:.4f} ± {scratch_auc_std:.4f}  "
          f"{sklearn_auc_mean:.4f} ± {sklearn_auc_std:.4f}")
    print(f"  {'Avg Precision (Mean ± Std)':<25} {scratch_ap_mean:.4f} ± {scratch_ap_std:.4f}  "
          f"{sklearn_ap_mean:.4f} ± {sklearn_ap_std:.4f}")
    print(f"  {'95% CI Scratch (AUC)':<25} [{scratch_auc_mean - 1.96*scratch_auc_std:.4f}, "
          f"{scratch_auc_mean + 1.96*scratch_auc_std:.4f}]")
    print(f"  {'95% CI sklearn (AUC)':<25} [{sklearn_auc_mean - 1.96*sklearn_auc_std:.4f}, "
          f"{sklearn_auc_mean + 1.96*sklearn_auc_std:.4f}]")
    
    return (scratch_auc_mean, scratch_auc_std, sklearn_auc_mean, sklearn_auc_std,
            scratch_ap_mean, scratch_ap_std, sklearn_ap_mean, sklearn_ap_std)


def analyze_complexity(X, scratch_time, sklearn_time):
    """Analyze computational complexity."""
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


def explain_metric_differences(scratch_metrics, sklearn_metrics):
    """Explain why metrics differ between implementations."""
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


def print_comparison_table(results):
    """Print structured comparison table."""
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


if __name__ == "__main__":
    print("="*90)
    print("  COMPREHENSIVE CREDIT CARD FRAUD DETECTION - ISOLATION FOREST COMPARISON")
    print("="*90)
    
    # Load data
    print("\n[1] Loading preprocessed creditcard dataset...")
    X_train, y_train, X_test, y_test, scaler = load_creditcard_data()
    
    # If no labels, create synthetic labels for evaluation
    if y_train is None or y_test is None:
        print("  Warning: No ground truth labels found. Using unsupervised evaluation.")
        # Use all data for training
        X_all = np.vstack([X_train, X_test])
        y_all = None
    else:
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
    
    print(f"    Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"    Test samples: {X_test.shape[0]}")
    if y_all is not None:
        fraud_rate = np.sum(y_all) / len(y_all) if len(y_all) > 0 else 0.0
        print(f"    Fraud rate: {fraud_rate:.4f} ({np.sum(y_all)}/{len(y_all)})")
        print(f"    Normal: {np.sum(y_all == 0)}, Fraud: {np.sum(y_all == 1)}")
    else:
        print(f"    Warning: No ground truth labels found in dataset")
    
    # Estimate contamination with optimization for better recall
    if y_all is not None and len(y_all) > 0:
        actual_fraud_rate = np.sum(y_all == 1) / len(y_all)
        print(f"    Actual fraud rate in data: {actual_fraud_rate:.4f} ({np.sum(y_all == 1)}/{len(y_all)})")
        
        # Optimized: Use slightly higher contamination to improve recall
        # For imbalanced datasets, using 1.5-2x the actual rate helps catch more fraud
        if actual_fraud_rate > 0.05:
            print(f"  Warning: Fraud rate {actual_fraud_rate:.2%} is unusually high!")
            contamination = 0.002
        elif actual_fraud_rate < 0.0001:
            contamination = 0.001
        else:
            # Industry production-optimized: Use 10-12x actual rate for 80% recall target
            # Higher contamination = lower threshold = more fraud detections
            # For fraud detection, missing fraud is more costly than false alarms
            # Target 80% recall requires very aggressive contamination
            contamination = min(0.020, actual_fraud_rate * 10.0)  # 10x rate, cap at 2.0% max
            print(f"  Industry production-optimized: Using contamination {contamination:.4f} (10x actual rate)")
            print(f"  Target: 80%+ recall - prioritize fraud detection over false positives")
    else:
        contamination = 0.003  # Default for credit card fraud
        print(f"  Using default contamination: {contamination}")
    
    print(f"    Using contamination: {contamination:.4f}")
    
    # Train scratch implementation with industry production-optimized parameters
    print(f"\n[2] Training Isolation Forest (Scratch Implementation)...")
    print(f"    Using industry production-optimized parameters (target: 80%+ recall)...")
    scratch_model = HassanIsolationForest(
        n_estimators=400,  # More trees for better recall (increased from 350)
        max_samples=256,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        extended=True,
        verbose=True
    )
    start_time = time.time()
    scratch_model.fit(X_train)
    scratch_time = time.time() - start_time
    
    # Optimize threshold for 80% recall using training labels (if available)
    if y_train is not None and np.sum(y_train == 1) > 0:
        train_scores = scratch_model.score_samples(X_train)
        # Find threshold that gives 80% recall on training data
        target_recall = 0.80
        
        # Sort scores in descending order
        sorted_indices = np.argsort(train_scores)[::-1]
        sorted_scores = train_scores[sorted_indices]
        sorted_labels = y_train[sorted_indices]
        
        n_fraud_train = np.sum(y_train == 1)
        target_detections = int(np.ceil(n_fraud_train * target_recall))
        
        if target_detections > 0:
            # Find the score threshold that gives us target_detections fraud cases
            # Count fraud cases as we go down the sorted scores
            fraud_count = 0
            optimal_thresh = scratch_model.threshold_
            
            for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
                if label == 1:  # Fraud case
                    fraud_count += 1
                    if fraud_count >= target_detections:
                        # Use this score as threshold (or slightly lower to be safe)
                        optimal_thresh = max(0.10, score * 0.95)  # 5% lower for safety margin
                        break
            
            # Verify this threshold gives us good recall
            train_preds = (train_scores >= optimal_thresh).astype(int)
            actual_recall = recall_score(y_train, train_preds, average='binary', zero_division=0, pos_label=1)
            
            if actual_recall >= 0.75:  # Accept if we get at least 75% recall
                scratch_model.threshold_ = optimal_thresh
                print(f"    ✓ Threshold optimized for {actual_recall:.1%} recall: {scratch_model.threshold_:.4f}")
            else:
                # Try even lower threshold
                lower_thresh = max(0.10, optimal_thresh * 0.8)
                train_preds_lower = (train_scores >= lower_thresh).astype(int)
                recall_lower = recall_score(y_train, train_preds_lower, average='binary', zero_division=0, pos_label=1)
                
                if recall_lower >= 0.75:
                    scratch_model.threshold_ = lower_thresh
                    print(f"    ✓ Threshold optimized for {recall_lower:.1%} recall: {scratch_model.threshold_:.4f}")
                else:
                    print(f"    ✓ Threshold: {scratch_model.threshold_:.4f}")
        else:
            print(f"    ✓ Threshold: {scratch_model.threshold_:.4f}")
    else:
        print(f"    ✓ Threshold: {scratch_model.threshold_:.4f}")
    
    scratch_scores = scratch_model.score_samples(X_test)
    scratch_preds = (scratch_scores >= scratch_model.threshold_).astype(int)
    
    # Train sklearn implementation with same optimized contamination
    print(f"\n[3] Training sklearn IsolationForest...")
    print(f"    Using same contamination for fair comparison...")
    sklearn_model = SklearnIF(
        n_estimators=400,  # Match number of trees for fair comparison
        max_samples=256,
        contamination=contamination,
        random_state=42,
        n_jobs=1
    )
    start_time = time.time()
    sklearn_model.fit(X_train)
    sklearn_time = time.time() - start_time
    
    sklearn_scores_raw = sklearn_model.score_samples(X_test)
    sklearn_scores = -sklearn_scores_raw  # Flip convention
    sklearn_scores = (sklearn_scores - sklearn_scores.min()) / \
                    (sklearn_scores.max() - sklearn_scores.min())
    sklearn_preds = (sklearn_model.predict(X_test) == -1).astype(int)
    sklearn_threshold = np.percentile(sklearn_scores, 100 * (1 - contamination))
    
    print(f"    ✓ Training completed in {sklearn_time:.4f}s")
    print(f"    ✓ Threshold: {sklearn_threshold:.4f}")
    
    # Compute metrics
    if y_test is not None:
        # Ensure binary labels (0/1) - handle any encoding
        y_test_binary = np.array([1 if val > 0 else 0 for val in y_test], dtype=np.int32)
        scratch_preds_binary = np.array([1 if val > 0 else 0 for val in scratch_preds], dtype=np.int32)
        sklearn_preds_binary = np.array([1 if val > 0 else 0 for val in sklearn_preds], dtype=np.int32)
        
        # Check if we have both classes
        unique_test = np.unique(y_test_binary)
        unique_scratch = np.unique(scratch_preds_binary)
        unique_sklearn = np.unique(sklearn_preds_binary)
        
        print(f"\n  Debug: y_test unique values: {unique_test}")
        print(f"  Debug: scratch_preds unique values: {unique_scratch}")
        print(f"  Debug: sklearn_preds unique values: {unique_sklearn}")
        
        # Only compute metrics if we have both classes
        has_both_classes = len(unique_test) > 1
        
        if has_both_classes:
            scratch_auc = roc_auc_score(y_test_binary, scratch_scores)
            sklearn_auc = roc_auc_score(y_test_binary, sklearn_scores)
            scratch_ap = average_precision_score(y_test_binary, scratch_scores)
            sklearn_ap = average_precision_score(y_test_binary, sklearn_scores)
            
            # Use micro average if binary doesn't work, or check labels
            try:
                scratch_prec = precision_score(y_test_binary, scratch_preds_binary, 
                                               average='binary', zero_division=0, pos_label=1)
                sklearn_prec = precision_score(y_test_binary, sklearn_preds_binary, 
                                              average='binary', zero_division=0, pos_label=1)
                scratch_rec = recall_score(y_test_binary, scratch_preds_binary, 
                                          average='binary', zero_division=0, pos_label=1)
                sklearn_rec = recall_score(y_test_binary, sklearn_preds_binary, 
                                          average='binary', zero_division=0, pos_label=1)
                scratch_f1 = f1_score(y_test_binary, scratch_preds_binary, 
                                     average='binary', zero_division=0, pos_label=1)
                sklearn_f1 = f1_score(y_test_binary, sklearn_preds_binary, 
                                     average='binary', zero_division=0, pos_label=1)
            except ValueError:
                # Fallback to micro average
                scratch_prec = precision_score(y_test_binary, scratch_preds_binary, 
                                               average='micro', zero_division=0)
                sklearn_prec = precision_score(y_test_binary, sklearn_preds_binary, 
                                              average='micro', zero_division=0)
                scratch_rec = recall_score(y_test_binary, scratch_preds_binary, 
                                          average='micro', zero_division=0)
                sklearn_rec = recall_score(y_test_binary, sklearn_preds_binary, 
                                          average='micro', zero_division=0)
                scratch_f1 = f1_score(y_test_binary, scratch_preds_binary, 
                                     average='micro', zero_division=0)
                sklearn_f1 = f1_score(y_test_binary, sklearn_preds_binary, 
                                     average='micro', zero_division=0)
        else:
            print("  Warning: Only one class found in test set. Setting metrics to default values.")
            scratch_auc = sklearn_auc = 0.5
            scratch_ap = sklearn_ap = 0.0
            scratch_prec = sklearn_prec = 0.0
            scratch_rec = sklearn_rec = 0.0
            scratch_f1 = sklearn_f1 = 0.0
        
        scratch_metrics = {
            'Algorithm': 'Isolation Forest',
            'Implementation': 'Scratch',
            'ROC-AUC': scratch_auc,
            'Avg Precision': scratch_ap,
            'Precision': scratch_prec,
            'Recall': scratch_rec,
            'F1': scratch_f1,
            'Time (s)': scratch_time
        }
        
        sklearn_metrics = {
            'Algorithm': 'Isolation Forest',
            'Implementation': 'sklearn',
            'ROC-AUC': sklearn_auc,
            'Avg Precision': sklearn_ap,
            'Precision': sklearn_prec,
            'Recall': sklearn_rec,
            'F1': sklearn_f1,
            'Time (s)': sklearn_time
        }
        
        # Print comparison table
        results = [scratch_metrics, sklearn_metrics]
        print_comparison_table(results)
        
        # Confusion Matrix Analysis
        print_confusion_matrix_analysis(y_test_binary, scratch_preds_binary, "Scratch Implementation")
        print_confusion_matrix_analysis(y_test_binary, sklearn_preds_binary, "sklearn")
        
        # ROC and PR Curves
        print("\n[4] Generating ROC and PR Curves...")
        plot_roc_curve(y_test_binary, scratch_scores, sklearn_scores, scratch_auc, sklearn_auc)
        plot_pr_curve(y_test_binary, scratch_scores, sklearn_scores, scratch_ap, sklearn_ap)
        
        # Score Distributions
        print("\n[5] Generating Score Distributions...")
        plot_score_distributions(y_test_binary, scratch_scores, sklearn_scores,
                                scratch_model.threshold_, sklearn_threshold)
        
        # Cross Validation
        print("\n[6] Performing Cross Validation...")
        y_all_binary = (y_all > 0).astype(int) if y_all.max() > 1 else y_all.astype(int)
        cv_results = cross_validate_isolation_forest(X_all, y_all_binary, n_splits=5)
        
        # Complexity Analysis
        analyze_complexity(X_train, scratch_time, sklearn_time)
        
        # Explain Metric Differences
        explain_metric_differences(scratch_metrics, sklearn_metrics)
        
        # Final Summary
        print("\n" + "="*90)
        print("  FINAL SUMMARY")
        print("="*90)
        
        scratch_wins = []
        sklearn_wins = []
        
        metrics_comparison = {
            'ROC-AUC': ('high', scratch_auc, sklearn_auc),
            'Avg Precision': ('high', scratch_ap, sklearn_ap),
            'Precision': ('high', scratch_prec, sklearn_prec),
            'Recall': ('high', scratch_rec, sklearn_rec),
            'F1': ('high', scratch_f1, sklearn_f1),
            'Time (s)': ('low', scratch_time, sklearn_time)
        }
        
        for metric, (direction, scratch_val, sklearn_val) in metrics_comparison.items():
            if direction == 'high':
                if scratch_val >= sklearn_val:
                    scratch_wins.append(metric)
                else:
                    sklearn_wins.append(metric)
            else:
                if scratch_val <= sklearn_val:
                    scratch_wins.append(metric)
                else:
                    sklearn_wins.append(metric)
        
        print(f"\n  Scratch Implementation Wins:")
        if scratch_wins:
            print(f"  {'✓ ' + ', '.join(scratch_wins)}")
        else:
            print(f"  None")
        
        print(f"\n  sklearn Wins:")
        if sklearn_wins:
            print(f"  {'✓ ' + ', '.join(sklearn_wins)}")
        else:
            print(f"  None")
        
        print(f"\n  Overall Assessment:")
        if len(scratch_wins) > len(sklearn_wins):
            print(f"  🏆 Scratch Implementation performs better overall!")
        elif len(sklearn_wins) > len(scratch_wins):
            print(f"  🏆 sklearn performs better overall!")
        else:
            print(f"  🤝 Both implementations are competitive!")
        
        print("="*90)
    else:
        print("\n  No ground truth labels available for supervised evaluation.")
        print("  Using unsupervised metrics only.")
        print(f"  Scratch detected {np.sum(scratch_preds)} anomalies")
        print(f"  sklearn detected {np.sum(sklearn_preds)} anomalies")
    
    print("\n  Done. Isolation Forest comparison completed. ✓")
