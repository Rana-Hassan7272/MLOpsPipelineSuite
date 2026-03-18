"""
Experiment: Isolation Forest vs sklearn
=======================================
Comprehensive comparison of scratch implementation vs sklearn on creditcard dataset.
Uses production-ready structure with training pipeline and evaluation modules.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import yaml
from sklearn.ensemble import IsolationForest as SklearnIF
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import modules
from algorithms.isolation_forest.scratch_isolation_forest import HassanIsolationForest
from training.trainer import ModelTrainer
from evaluation.metrics import (
    print_confusion_matrix_analysis_fraud,
    plot_roc_curves,
    plot_pr_curve,
    plot_score_distributions,
    analyze_complexity_isolation_forest,
    explain_metric_differences_isolation_forest,
    print_comparison_table
)
from evaluation.cross_validation import cross_validate_isolation_forest
from tracking.mlflow_tracker import MLflowTracker
import mlflow
from mlflow.models import infer_signature


def load_creditcard_data():
    """Load preprocessed creditcard dataset."""
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


def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    print("="*90)
    print("  COMPREHENSIVE CREDIT CARD FRAUD DETECTION - ISOLATION FOREST COMPARISON")
    print("="*90)
    
    # Load configuration
    config = load_config()
    if_config = config.get('isolation_forest', {})
    hyperparams = if_config.get('hyperparameters', {})
    
    # Load data
    print("\n[1] Loading preprocessed creditcard dataset...")
    X_train, y_train, X_test, y_test, scaler = load_creditcard_data()
    
    # If no labels, create synthetic labels for evaluation
    if y_train is None or y_test is None:
        print("  Warning: No ground truth labels found. Using unsupervised evaluation.")
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
        
        if actual_fraud_rate > 0.05:
            print(f"  Warning: Fraud rate {actual_fraud_rate:.2%} is unusually high!")
            contamination = 0.002
        elif actual_fraud_rate < 0.0001:
            contamination = 0.001
        else:
            # Industry production-optimized: Use 10x actual rate for 80% recall target
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
        n_estimators=hyperparams.get('n_estimators', 400),
        max_samples=hyperparams.get('max_samples', 256),
        contamination=contamination,
        random_state=if_config.get('training', {}).get('random_state', 42),
        n_jobs=if_config.get('training', {}).get('n_jobs', -1),
        extended=hyperparams.get('extended', True),
        verbose=if_config.get('training', {}).get('verbose', True)
    )
    
    scratch_trainer = ModelTrainer(scratch_model, "Scratch Implementation")
    scratch_trainer.train(X_train)
    
    # Optimize threshold for 80% recall using training labels (if available)
    if y_train is not None and np.sum(y_train == 1) > 0:
        train_scores = scratch_model.score_samples(X_train)
        # Find threshold that gives 80% recall on training data
        target_recall = hyperparams.get('target_recall', 0.80)
        
        # Sort scores in descending order
        sorted_indices = np.argsort(train_scores)[::-1]
        sorted_scores = train_scores[sorted_indices]
        sorted_labels = y_train[sorted_indices]
        
        n_fraud_train = np.sum(y_train == 1)
        target_detections = int(np.ceil(n_fraud_train * target_recall))
        
        if target_detections > 0:
            # Find the score threshold that gives us target_detections fraud cases
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
        n_estimators=hyperparams.get('n_estimators', 400),
        max_samples=hyperparams.get('max_samples', 256),
        contamination=contamination,
        random_state=if_config.get('training', {}).get('random_state', 42),
        n_jobs=1
    )
    
    sklearn_trainer = ModelTrainer(sklearn_model, "sklearn IsolationForest")
    sklearn_trainer.train(X_train)
    
    sklearn_scores_raw = sklearn_model.score_samples(X_test)
    sklearn_scores = -sklearn_scores_raw  # Flip convention
    sklearn_scores = (sklearn_scores - sklearn_scores.min()) / \
                    (sklearn_scores.max() - sklearn_scores.min())
    sklearn_preds = (sklearn_model.predict(X_test) == -1).astype(int)
    sklearn_threshold = np.percentile(sklearn_scores, 100 * (1 - contamination))
    
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
            'Time (s)': scratch_trainer.get_training_time()
        }
        
        sklearn_metrics = {
            'Algorithm': 'Isolation Forest',
            'Implementation': 'sklearn',
            'ROC-AUC': sklearn_auc,
            'Avg Precision': sklearn_ap,
            'Precision': sklearn_prec,
            'Recall': sklearn_rec,
            'F1': sklearn_f1,
            'Time (s)': sklearn_trainer.get_training_time()
        }
        
        # Print comparison table
        results = [scratch_metrics, sklearn_metrics]
        print_comparison_table(results, algorithm_type='anomaly_detection')
        
        # Confusion Matrix Analysis
        print_confusion_matrix_analysis_fraud(y_test_binary, scratch_preds_binary, "Scratch Implementation")
        print_confusion_matrix_analysis_fraud(y_test_binary, sklearn_preds_binary, "sklearn")
        
        # ROC and PR Curves
        print("\n[4] Generating ROC and PR Curves...")
        output_dir = os.path.join(project_root, 'experiments')
        os.makedirs(output_dir, exist_ok=True)
        auc_scratch, auc_sklearn = plot_roc_curves(y_test_binary, scratch_scores, sklearn_scores, 
                                                   output_dir, "Isolation Forest Comparison")
        print(f"    Scratch AUC: {auc_scratch:.4f}")
        print(f"    sklearn AUC: {auc_sklearn:.4f}")
        print(f"    AUC Difference: {auc_scratch - auc_sklearn:+.4f}")
        plot_pr_curve(y_test_binary, scratch_scores, sklearn_scores, scratch_ap, sklearn_ap, output_dir)
        
        # Score Distributions
        print("\n[5] Generating Score Distributions...")
        plot_score_distributions(y_test_binary, scratch_scores, sklearn_scores,
                                scratch_model.threshold_, sklearn_threshold, output_dir)
        
        # Cross Validation
        print("\n[6] Performing Cross Validation...")
        y_all_binary = (y_all > 0).astype(int) if y_all.max() > 1 else y_all.astype(int)
        cv_results = cross_validate_isolation_forest(
            X_all, y_all_binary, 
            n_splits=if_config.get('evaluation', {}).get('cv_folds', 5),
            scratch_model_class=HassanIsolationForest
        )
        # Unpack CV results: (scratch_auc_mean, scratch_auc_std, sklearn_auc_mean, sklearn_auc_std,
        #                     scratch_ap_mean, scratch_ap_std, sklearn_ap_mean, sklearn_ap_std)
        cv_scratch_auc_mean, cv_scratch_auc_std, cv_sklearn_auc_mean, cv_sklearn_auc_std, \
        cv_scratch_ap_mean, cv_scratch_ap_std, cv_sklearn_ap_mean, cv_sklearn_ap_std = cv_results
        
        # Complexity Analysis
        analyze_complexity_isolation_forest(X_train, scratch_trainer.get_training_time(), 
                                          sklearn_trainer.get_training_time())
        
        # Explain Metric Differences
        explain_metric_differences_isolation_forest(scratch_metrics, sklearn_metrics)
        
        # MLflow Tracking
        print("\n[7] Logging to MLflow...")
        experiment_name = "Isolation_Forest_Comparison"
        
        # Track Scratch Implementation
        scratch_hyperparams = {
            "n_estimators": hyperparams.get('n_estimators', 400),
            "max_samples": hyperparams.get('max_samples', 256),
            "contamination": contamination,
            "max_features": hyperparams.get('max_features', 1.0),
            "extended": hyperparams.get('extended', True),
            "target_recall": hyperparams.get('target_recall', 0.80),
            "threshold": scratch_model.threshold_
        }
        scratch_mlflow_metrics = {
            "roc_auc": scratch_auc,
            "average_precision": scratch_ap,
            "precision": scratch_prec,
            "recall": scratch_rec,
            "f1": scratch_f1,
            "cv_mean_roc_auc": cv_scratch_auc_mean,
            "cv_std_roc_auc": cv_scratch_auc_std,
            "cv_mean_ap": cv_scratch_ap_mean,
            "cv_std_ap": cv_scratch_ap_std
        }
        
        with MLflowTracker(experiment_name) as tracker:
            tracker.start_run(run_name="Isolation_Forest_Scratch",
                             tags={"implementation": "scratch", "dataset": "creditcard",
                                   "model_type": "anomaly_detection", "status": "experiment"})
            tracker.log_params(scratch_hyperparams)
            tracker.log_metrics(scratch_mlflow_metrics)
            tracker.log_metrics({"training_time_seconds": scratch_metrics['Time (s)']})
            
            # Log dataset info
            tracker.log_dataset_info(X_train, y_train, "train")
            tracker.log_dataset_info(X_test, y_test, "test")
            
            # Log confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test_binary, scratch_preds_binary)
            tracker.log_confusion_matrix(cm, labels=["Normal", "Fraud"])
            
            # Log plots
            if os.path.exists(os.path.join(output_dir, "roc_curve_comparison.png")):
                tracker.log_plot(os.path.join(output_dir, "roc_curve_comparison.png"), "plots")
            if os.path.exists(os.path.join(output_dir, "pr_curve_comparison.png")):
                tracker.log_plot(os.path.join(output_dir, "pr_curve_comparison.png"), "plots")
            if os.path.exists(os.path.join(output_dir, "score_distributions.png")):
                tracker.log_plot(os.path.join(output_dir, "score_distributions.png"), "plots")
            if config:
                tracker.log_dict(config, "config")
            print("    ✓ Scratch implementation logged to MLflow")
        
        # Track sklearn Implementation
        sklearn_hyperparams = {
            "n_estimators": hyperparams.get('n_estimators', 400),
            "max_samples": hyperparams.get('max_samples', 256),
            "contamination": contamination,
            "max_features": hyperparams.get('max_features', 1.0),
            "random_state": if_config.get('training', {}).get('random_state', 42),
            "threshold": sklearn_threshold
        }
        sklearn_mlflow_metrics = {
            "roc_auc": sklearn_auc,
            "average_precision": sklearn_ap,
            "precision": sklearn_prec,
            "recall": sklearn_rec,
            "f1": sklearn_f1,
            "cv_mean_roc_auc": cv_sklearn_auc_mean,
            "cv_std_roc_auc": cv_sklearn_auc_std,
            "cv_mean_ap": cv_sklearn_ap_mean,
            "cv_std_ap": cv_sklearn_ap_std
        }
        
        with MLflowTracker(experiment_name) as tracker:
            tracker.start_run(run_name="Isolation_Forest_sklearn",
                             tags={"implementation": "sklearn", "dataset": "creditcard",
                                   "model_type": "anomaly_detection", "status": "experiment"})
            tracker.log_params(sklearn_hyperparams)
            tracker.log_metrics(sklearn_mlflow_metrics)
            tracker.log_metrics({"training_time_seconds": sklearn_metrics['Time (s)']})
            
            # Log dataset info
            tracker.log_dataset_info(X_train, y_train, "train")
            tracker.log_dataset_info(X_test, y_test, "test")
            
            # Log confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test_binary, sklearn_preds_binary)
            tracker.log_confusion_matrix(cm, labels=["Normal", "Fraud"])
            
            # Log model with signature
            try:
                signature = infer_signature(X_test[:10], sklearn_model.score_samples(X_test[:10]))
                tracker.log_model(sklearn_model, "model", signature=signature,
                                input_example=X_test[:5])
                
                # Register best model (based on F1 score for fraud detection)
                if sklearn_metrics['F1'] >= scratch_metrics['F1']:
                    model_uri = f"runs:/{tracker.active_run.info.run_id}/model"
                    mv = tracker.register_model(
                        model_name="IsolationForest_sklearn",
                        model_uri=model_uri,
                        stage="Staging",
                        tags={"algorithm": "isolation_forest", "best_metric": "f1",
                              "f1_score": str(sklearn_metrics['F1']),
                              "recall": str(sklearn_metrics['Recall'])}
                    )
                    if mv:
                        print(f"    ✓ Model registered: IsolationForest_sklearn v{mv.version} (Staging)")
            except Exception as e:
                print(f"    Note: Model logging/registration skipped: {e}")
            
            # Log plots
            if os.path.exists(os.path.join(output_dir, "roc_curve_comparison.png")):
                tracker.log_plot(os.path.join(output_dir, "roc_curve_comparison.png"), "plots")
            if os.path.exists(os.path.join(output_dir, "pr_curve_comparison.png")):
                tracker.log_plot(os.path.join(output_dir, "pr_curve_comparison.png"), "plots")
            if os.path.exists(os.path.join(output_dir, "score_distributions.png")):
                tracker.log_plot(os.path.join(output_dir, "score_distributions.png"), "plots")
            if config:
                tracker.log_dict(config, "config")
            print("    ✓ sklearn implementation logged to MLflow")
        
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
            'Time (s)': ('low', scratch_trainer.get_training_time(), sklearn_trainer.get_training_time())
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
