"""
Cross-Validation Module
=======================
Provides k-fold cross-validation functionality.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def cross_validate_model(X, y, model, model_name, cv_folds=5, model_class=None):
    """
    Perform k-fold cross validation.
    
    Parameters
    ----------
    X : array-like
        Training features
    y : array-like
        Training labels
    model : object
        Trained model instance (for sklearn) or model class (for scratch)
    model_name : str
        Name of the model for display
    cv_folds : int
        Number of cross-validation folds (default: 5)
    model_class : class, optional
        Model class to instantiate for scratch implementations
    """
    print(f"\n{'='*70}")
    print(f"  {cv_folds}-FOLD CROSS VALIDATION - {model_name}")
    print(f"{'='*70}")
    
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    # Flatten y if needed
    y_flat = y.flatten() if hasattr(y, 'flatten') else y
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_flat), 1):
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
        else:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
        
        y_train_fold = y_flat[train_idx]
        y_val_fold = y_flat[val_idx]
        
        # Train model
        if model_class is not None:
            # For scratch implementations, create new instance
            model_copy = model_class(
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
            # For sklearn models, clone and fit
            from sklearn.base import clone
            model_copy = clone(model)
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


def cross_validate_kmeans(X, k, n_splits=5, scratch_model_class=None):
    """
    Perform k-fold cross validation for K-means clustering.
    
    Parameters
    ----------
    X : array-like
        Training features
    k : int
        Number of clusters
    n_splits : int
        Number of cross-validation folds (default: 5)
    scratch_model_class : class, optional
        Scratch model class to instantiate
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans as SklearnKMeans
    
    print(f"\n{'='*70}")
    print(f"  {n_splits}-FOLD CROSS VALIDATION")
    print(f"{'='*70}")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scratch_scores = []
    sklearn_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        
        # Scratch implementation
        if scratch_model_class is not None:
            scratch_model = scratch_model_class(n_clusters=k, n_init=5, random_state=42, verbose=False)
            scratch_model.fit(X_train_fold)
            scratch_labels = scratch_model.predict(X_val_fold)
        else:
            # Fallback if no model class provided
            scratch_labels = np.zeros(len(X_val_fold))
        
        scratch_sil = silhouette_score(X_val_fold, scratch_labels)
        scratch_scores.append(scratch_sil)
        
        # sklearn implementation
        sklearn_model = SklearnKMeans(n_clusters=k, n_init=5, random_state=42)
        sklearn_model.fit(X_train_fold)
        sklearn_labels = sklearn_model.predict(X_val_fold)
        sklearn_sil = silhouette_score(X_val_fold, sklearn_labels)
        sklearn_scores.append(sklearn_sil)
        
        print(f"  Fold {fold}: Scratch Silhouette = {scratch_sil:.4f}, "
              f"sklearn Silhouette = {sklearn_sil:.4f}")
    
    scratch_mean = np.mean(scratch_scores)
    scratch_std = np.std(scratch_scores)
    sklearn_mean = np.mean(sklearn_scores)
    sklearn_std = np.std(sklearn_scores)
    
    print(f"\n  Cross-Validation Results:")
    print(f"  {'Scratch - Mean':<25}: {scratch_mean:.4f} ± {scratch_std:.4f}")
    print(f"  {'sklearn - Mean':<25}: {sklearn_mean:.4f} ± {sklearn_std:.4f}")
    print(f"  {'95% CI Scratch':<25}: [{scratch_mean - 1.96*scratch_std:.4f}, "
          f"{scratch_mean + 1.96*scratch_std:.4f}]")
    print(f"  {'95% CI sklearn':<25}: [{sklearn_mean - 1.96*sklearn_std:.4f}, "
          f"{sklearn_mean + 1.96*sklearn_std:.4f}]")
    
    return scratch_mean, scratch_std, sklearn_mean, sklearn_std


def cross_validate_isolation_forest(X, y, n_splits=5, scratch_model_class=None):
    """
    Perform stratified k-fold cross validation for Isolation Forest.
    
    Parameters
    ----------
    X : array-like
        Training features
    y : array-like
        Training labels (for stratified split)
    n_splits : int
        Number of cross-validation folds (default: 5)
    scratch_model_class : class, optional
        Scratch model class to instantiate
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import IsolationForest as SklearnIF
    from sklearn.metrics import roc_auc_score, average_precision_score
    
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
        if scratch_model_class is not None:
            scratch_model = scratch_model_class(
                n_estimators=250, max_samples=256,  # More trees for CV stability and recall
                contamination=contamination, random_state=42,
                n_jobs=-1, extended=True, verbose=False
            )
            scratch_model.fit(X_train_fold)
            scratch_scores = scratch_model.score_samples(X_val_fold)
        else:
            scratch_scores = np.random.random(len(X_val_fold))  # Fallback
        
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
