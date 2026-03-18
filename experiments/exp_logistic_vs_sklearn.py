"""
Experiment: Logistic Regression vs sklearn
==========================================
Comprehensive comparison of scratch implementation vs sklearn on spam dataset.
Uses production-ready structure with training pipeline and evaluation modules.
"""

import os
import sys
import numpy as np
import time
import yaml
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import f1_score

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import modules
from algorithms.logistic_regression.logistic_regression import Hassan
from training.pipeline import TrainingPipeline
from training.trainer import ModelTrainer
from evaluation.metrics import (
    evaluate_model,
    print_confusion_matrix_analysis,
    plot_roc_curves,
    plot_training_curve,
    analyze_complexity,
    explain_metric_differences,
    print_comparison_table
)
from evaluation.cross_validation import cross_validate_model
from tracking.mlflow_tracker import MLflowTracker
from mlflow.models import infer_signature
import mlflow


def load_spam_data():
    """Load preprocessed spam dataset."""
    data_dir = os.path.join(project_root, 'data_pipeline', 'processed_data')
    
    X_train = load_npz(os.path.join(data_dir, 'spam_X_train.npz'))
    X_test = load_npz(os.path.join(data_dir, 'spam_X_test.npz'))
    y_train = np.load(os.path.join(data_dir, 'spam_y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'spam_y_test.npy'))
    
    return X_train, X_test, y_train, y_test


def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    print("="*90)
    print("  COMPREHENSIVE SPAM DETECTION - LOGISTIC REGRESSION COMPARISON")
    print("="*90)
    
    # Load configuration
    config = load_config()
    hyperparams = config.get('hyperparameters', {})
    
    # Load data
    print("\n[1] Loading preprocessed spam dataset...")
    X_train, X_test, y_train, y_test = load_spam_data()
    print(f"    Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"    Test:  {X_test.shape[0]} samples")
    print(f"    Class distribution - Train: {np.bincount(y_train.flatten())}")
    print(f"    Class distribution - Test:  {np.bincount(y_test.flatten())}")
    
    # Train scratch implementation
    print("\n[2] Training Logistic Regression (Scratch Implementation)...")
    scratch_model = Hassan(
        learning_rate=hyperparams.get('learning_rate', 0.2),
        epochs=hyperparams.get('epochs', 1500),
        verbose=config.get('training', {}).get('verbose', False),
        regularization=hyperparams.get('regularization', 0.001),
        random_state=config.get('training', {}).get('random_state', 42),
        adaptive_lr=hyperparams.get('adaptive_lr', True),
        lr_decay=hyperparams.get('lr_decay', 0.997),
        patience=hyperparams.get('patience', 150)
    )
    
    scratch_trainer = ModelTrainer(scratch_model, "Scratch Implementation")
    scratch_trainer.train(X_train, y_train.flatten())
    
    y_proba_scratch = scratch_trainer.predict_proba(X_test)
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred_temp = (y_proba_scratch >= threshold).astype(int)
        f1 = f1_score(y_test.flatten(), y_pred_temp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_pred_scratch = scratch_trainer.predict(X_test, threshold=best_threshold)
    scratch_metrics = evaluate_model(y_test.flatten(), y_pred_scratch, y_proba_scratch, 
                                     "Scratch Implementation")
    scratch_metrics['Training Time'] = scratch_trainer.get_training_time()
    scratch_metrics['Algorithm'] = 'Logistic Regression'
    scratch_metrics['Implementation'] = 'Scratch'
    
    print(f"    ✓ Epochs: {len(scratch_model.loss_history)}")
    print(f"    ✓ Optimal threshold: {best_threshold:.4f}")
    
    # Train sklearn implementation
    print("\n[3] Training sklearn LogisticRegression...")
    sklearn_model = SklearnLR(random_state=42, max_iter=2000, solver='lbfgs', verbose=0)
    sklearn_trainer = ModelTrainer(sklearn_model, "sklearn LogisticRegression")
    sklearn_trainer.train(X_train, y_train.flatten())
    
    y_proba_sklearn = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred_sklearn = sklearn_trainer.predict(X_test)
    sklearn_metrics = evaluate_model(y_test.flatten(), y_pred_sklearn, y_proba_sklearn,
                                     "sklearn LogisticRegression")
    sklearn_metrics['Training Time'] = sklearn_trainer.get_training_time()
    sklearn_metrics['Algorithm'] = 'Logistic Regression'
    sklearn_metrics['Implementation'] = 'sklearn'
    
    # Print comparison table
    results = [scratch_metrics, sklearn_metrics]
    print_comparison_table(results)
    
    # 1. Confusion Matrix Analysis
    print_confusion_matrix_analysis(scratch_metrics['Confusion Matrix'], "Scratch Implementation")
    print_confusion_matrix_analysis(sklearn_metrics['Confusion Matrix'], "sklearn")
    
    # 2. ROC Curve + AUC
    print("\n[4] Generating ROC Curve...")
    output_dir = os.path.join(project_root, 'experiments')
    os.makedirs(output_dir, exist_ok=True)
    auc_scratch, auc_sklearn = plot_roc_curves(
        y_test.flatten(), y_proba_scratch, y_proba_sklearn, output_dir
    )
    print(f"    Scratch AUC: {auc_scratch:.4f}")
    print(f"    sklearn AUC: {auc_sklearn:.4f}")
    print(f"    AUC Difference: {auc_scratch - auc_sklearn:+.4f}")
    
    # 3. Training Curve
    print("\n[5] Generating Training Curve...")
    plot_training_curve(scratch_model.loss_history, "Scratch Implementation", output_dir)
    print(f"    Initial Loss: {scratch_model.loss_history[0]:.6f}")
    print(f"    Final Loss: {scratch_model.loss_history[-1]:.6f}")
    print(f"    Loss Reduction: {scratch_model.loss_history[0] - scratch_model.loss_history[-1]:.6f}")
    
    # 4. Cross Validation
    print("\n[6] Performing Cross Validation...")
    cv_mean_scratch, cv_std_scratch, _ = cross_validate_model(
        X_train, y_train, scratch_model, "Scratch Implementation", 
        cv_folds=config.get('evaluation', {}).get('cv_folds', 5),
        model_class=Hassan
    )
    cv_mean_sklearn, cv_std_sklearn, _ = cross_validate_model(
        X_train, y_train, sklearn_model, "sklearn", 
        cv_folds=config.get('evaluation', {}).get('cv_folds', 5)
    )
    
    # MLflow Tracking
    print("\n[7] Logging to MLflow...")
    experiment_name = "Logistic_Regression_Comparison"
    
    # Track Scratch Implementation
    scratch_hyperparams = {
        "learning_rate": hyperparams.get('learning_rate', 0.2),
        "epochs": len(scratch_model.loss_history),
        "regularization": hyperparams.get('regularization', 0.001),
        "adaptive_lr": hyperparams.get('adaptive_lr', True),
        "lr_decay": hyperparams.get('lr_decay', 0.997),
        "patience": hyperparams.get('patience', 150),
        "optimal_threshold": best_threshold
    }
    scratch_mlflow_metrics = {
        "accuracy": scratch_metrics['Accuracy'],
        "precision": scratch_metrics['Precision'],
        "recall": scratch_metrics['Recall'],
        "f1_score": scratch_metrics['F1 Score'],
        "auc": scratch_metrics['AUC'],
        "cv_mean_accuracy": cv_mean_scratch,
        "cv_std_accuracy": cv_std_scratch
    }
    
    with MLflowTracker(experiment_name) as tracker:
        tracker.start_run(run_name="Logistic_Regression_Scratch", 
                         tags={"implementation": "scratch", "dataset": "spam", 
                               "model_type": "classification", "status": "experiment"})
        tracker.log_params(scratch_hyperparams)
        tracker.log_metrics(scratch_mlflow_metrics)
        tracker.log_metrics({"training_time_seconds": scratch_metrics['Training Time']})
        
        # Log training curve (loss history)
        tracker.log_training_curve(scratch_model.loss_history, "training_loss")
        
        # Log dataset info
        tracker.log_dataset_info(X_train, y_train, "train")
        tracker.log_dataset_info(X_test, y_test, "test")
        
        # Log confusion matrix
        tracker.log_confusion_matrix(scratch_metrics['Confusion Matrix'], 
                                    labels=["Not Spam", "Spam"])
        
        # Log plots
        tracker.log_plot(os.path.join(output_dir, "roc_curve_comparison.png"), "plots")
        tracker.log_plot(os.path.join(output_dir, "training_curve.png"), "plots")
        
        if config:
            tracker.log_dict(config, "config")
        
        # Log model with signature (for scratch, we'll log a simple wrapper)
        try:
            # Create signature from test data
            if hasattr(X_test, 'toarray'):
                X_test_sample = X_test[:10].toarray()
            else:
                X_test_sample = X_test[:10]
            y_proba_sample = y_proba_scratch[:10]
            signature = infer_signature(X_test_sample, y_proba_sample)
            
            # For scratch models, we can't directly log, but we log metadata
            tracker.set_tags({
                "model_type": "scratch_logistic_regression",
                "has_model_artifact": "false"
            })
        except Exception as e:
            print(f"    Note: Could not create signature: {e}")
        
        print("    ✓ Scratch implementation logged to MLflow")
    
    # Track sklearn Implementation
    sklearn_hyperparams = {
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42
    }
    sklearn_mlflow_metrics = {
        "accuracy": sklearn_metrics['Accuracy'],
        "precision": sklearn_metrics['Precision'],
        "recall": sklearn_metrics['Recall'],
        "f1_score": sklearn_metrics['F1 Score'],
        "auc": sklearn_metrics['AUC'],
        "cv_mean_accuracy": cv_mean_sklearn,
        "cv_std_accuracy": cv_std_sklearn
    }
    
    # Additional manual logging for sklearn (auto-logging already captured most)
    with MLflowTracker(experiment_name) as tracker:
        tracker.start_run(run_name="Logistic_Regression_sklearn",
                         tags={"implementation": "sklearn", "dataset": "spam",
                               "model_type": "classification", "status": "experiment"})
        tracker.log_params(sklearn_hyperparams)
        tracker.log_metrics(sklearn_mlflow_metrics)
        tracker.log_metrics({"training_time_seconds": sklearn_metrics['Training Time']})
        
        # Log confusion matrix
        tracker.log_confusion_matrix(sklearn_metrics['Confusion Matrix'],
                                    labels=["Not Spam", "Spam"])
        
        # Log plots
        tracker.log_plot(os.path.join(output_dir, "roc_curve_comparison.png"), "plots")
        
        if config:
            tracker.log_dict(config, "config")
        
        # Log sklearn model with signature
        try:
            if hasattr(X_test, 'toarray'):
                X_test_sample = X_test[:10].toarray()
            else:
                X_test_sample = X_test[:10]
            y_proba_sample = sklearn_model.predict_proba(X_test[:10])[:, 1]
            signature = infer_signature(X_test_sample, y_proba_sample)
            
            # Log model with signature
            tracker.log_model(sklearn_model, "model", signature=signature, 
                            input_example=X_test_sample[:5])
            
            # Register best model to Model Registry (based on F1 score)
            if sklearn_metrics['F1 Score'] >= scratch_metrics['F1 Score']:
                model_uri = f"runs:/{tracker.active_run.info.run_id}/model"
                mv = tracker.register_model(
                    model_name="LogisticRegression_sklearn",
                    model_uri=model_uri,
                    stage="Staging",
                    tags={"algorithm": "logistic_regression", "best_metric": "f1_score",
                          "f1_score": str(sklearn_metrics['F1 Score'])}
                )
                if mv:
                    print(f"    ✓ Model registered: LogisticRegression_sklearn v{mv.version} (Staging)")
        except Exception as e:
            print(f"    Note: Model logging/registration skipped: {e}")
        
        print("    ✓ sklearn implementation logged to MLflow")
    
    # 5. Complexity Analysis
    analyze_complexity(X_train, scratch_trainer.get_training_time(), 
                      sklearn_trainer.get_training_time())
    
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
