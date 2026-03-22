"""
Experiment: K-Means vs sklearn
===============================
Comprehensive comparison of scratch implementation vs sklearn on store_customers dataset.
Uses production-ready structure with training pipeline and evaluation modules.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import yaml
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import modules
from algorithms.kmean.hassan_kmeans import HassanKMeans, full_evaluation, hassan_auto_k
from training.trainer import ModelTrainer
from evaluation.metrics import (
    print_cluster_analysis,
    plot_elbow_curve,
    plot_cluster_visualization,
    plot_inertia_history,
    analyze_complexity_clustering,
    explain_metric_differences_clustering,
    print_comparison_table
)
from evaluation.cross_validation import cross_validate_kmeans
from tracking.mlflow_tracker import MLflowTracker
import mlflow
from mlflow.models import infer_signature


def load_store_customers_data():
    """Load preprocessed store_customers dataset."""
    data_path = os.path.join(project_root, 'data_pipeline', 'processed_data', 
                             'store_customers_processed.csv')
    
    df = pd.read_csv(data_path)
    X = df.values.astype(np.float64)
    
    return X, df


def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    print("="*90)
    print("  COMPREHENSIVE STORE CUSTOMERS CLUSTERING - K-MEANS COMPARISON")
    print("="*90)
    
    # Load configuration
    config = load_config()
    kmeans_config = config.get('kmeans', {})
    hyperparams = kmeans_config.get('hyperparameters', {})
    
    # Load data
    print("\n[1] Loading preprocessed store_customers dataset...")
    X, df = load_store_customers_data()
    print(f"    Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"    Feature names: {list(df.columns)}")
    
    # Auto-k selection
    print("\n[2] Auto-k selection via elbow method and gap statistic...")
    if hyperparams.get('auto_k', True):
        k_range = range(hyperparams.get('k_range', [2, 11])[0], 
                        hyperparams.get('k_range', [2, 11])[1])
        n_refs = hyperparams.get('n_refs', 3)
        best_k, auto_k_stats = hassan_auto_k(X, k_range=k_range, n_refs=n_refs, 
                                            random_state=kmeans_config.get('training', {}).get('random_state', 42))
        print(f"    Recommended k = {best_k} (gap statistic)")
    else:
        best_k = hyperparams.get('n_clusters', 5)
        auto_k_stats = None
    
    k = hyperparams.get('n_clusters', 5)
    print(f"    Using k = {k} for comparison")
    
    # Train scratch implementation
    print(f"\n[3] Training K-Means (Scratch Implementation)...")
    scratch_model = HassanKMeans(
        n_clusters=k,
        n_init=hyperparams.get('n_init', 10),
        max_iter=hyperparams.get('max_iter', 300),
        tol=hyperparams.get('tol', 1e-4),
        random_state=kmeans_config.get('training', {}).get('random_state', 42),
        verbose=kmeans_config.get('training', {}).get('verbose', True)
    )
    
    scratch_trainer = ModelTrainer(scratch_model, "Scratch Implementation")
    scratch_trainer.train(X)
    
    scratch_metrics = full_evaluation(
        "Scratch Implementation", X, scratch_model.labels_,
        scratch_model.cluster_centers_, scratch_model.inertia_,
        scratch_trainer.get_training_time(), y_true=None
    )
    scratch_metrics['Algorithm'] = 'K-Means'
    scratch_metrics['Implementation'] = 'Scratch'
    
    print(f"    [OK] Best inertia: {scratch_model.inertia_:.2f}")
    print(f"    [OK] Iterations: {scratch_model.n_iter_}")
    
    # Train sklearn implementation
    print(f"\n[4] Training sklearn KMeans...")
    sklearn_model = SklearnKMeans(
        n_clusters=k,
        n_init=hyperparams.get('n_init', 10),
        max_iter=hyperparams.get('max_iter', 300),
        tol=hyperparams.get('tol', 1e-4),
        random_state=kmeans_config.get('training', {}).get('random_state', 42),
        algorithm='lloyd'
    )
    
    sklearn_trainer = ModelTrainer(sklearn_model, "sklearn KMeans")
    sklearn_trainer.train(X)
    
    sklearn_metrics = full_evaluation(
        "sklearn KMeans", X, sklearn_model.labels_,
        sklearn_model.cluster_centers_, sklearn_model.inertia_,
        sklearn_trainer.get_training_time(), y_true=None
    )
    sklearn_metrics['Algorithm'] = 'K-Means'
    sklearn_metrics['Implementation'] = 'sklearn'
    
    print(f"    [OK] Best inertia: {sklearn_model.inertia_:.2f}")
    print(f"    [OK] Iterations: {sklearn_model.n_iter_}")
    
    # Print comparison table
    results = [scratch_metrics, sklearn_metrics]
    print_comparison_table(results, algorithm_type='clustering')
    
    # Cluster Analysis
    print_cluster_analysis(scratch_model.labels_, scratch_model.cluster_centers_, 
                          "Scratch Implementation", X)
    print_cluster_analysis(sklearn_model.labels_, sklearn_model.cluster_centers_,
                          "sklearn", X)
    
    # Ensure output directory exists even when `auto_k_stats` is None.
    output_dir = os.path.join(project_root, 'experiments')
    os.makedirs(output_dir, exist_ok=True)
    
    # Elbow Curve
    if auto_k_stats is not None:
        print("\n[5] Generating Elbow Curve...")
        plot_elbow_curve(auto_k_stats, "Store Customers", output_dir)
    
    # Cluster Visualizations
    print("\n[6] Generating Cluster Visualizations...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    scratch_centroids_2d = pca.transform(scratch_model.cluster_centers_)
    sklearn_centroids_2d = pca.transform(sklearn_model.cluster_centers_)
    
    plot_cluster_visualization(X_2d, scratch_model.labels_, scratch_centroids_2d,
                               "Scratch Implementation", "scratch_clusters.png", output_dir)
    plot_cluster_visualization(X_2d, sklearn_model.labels_, sklearn_centroids_2d,
                               "sklearn", "sklearn_clusters.png", output_dir)
    
    # Inertia History
    print("\n[7] Generating Inertia History...")
    plot_inertia_history(scratch_model.inertia_history_, "Scratch Implementation", output_dir)
    
    # Cross Validation
    print("\n[8] Performing Cross Validation...")
    cv_scratch_mean, cv_scratch_std, cv_sklearn_mean, cv_sklearn_std = cross_validate_kmeans(
        X, k, n_splits=kmeans_config.get('evaluation', {}).get('cv_folds', 5),
        scratch_model_class=HassanKMeans
    )
    
    # Complexity Analysis
    analyze_complexity_clustering(X, scratch_trainer.get_training_time(), 
                                 sklearn_trainer.get_training_time())
    
    # Explain Metric Differences
    explain_metric_differences_clustering(scratch_metrics, sklearn_metrics)
    
    # MLflow Tracking
    print("\n[9] Logging to MLflow...")
    experiment_name = "KMeans_Comparison"
    
    # Track Scratch Implementation
    scratch_hyperparams = {
        "n_clusters": k,
        "n_init": hyperparams.get('n_init', 10),
        "max_iter": scratch_model.n_iter_,
        "tol": hyperparams.get('tol', 1e-4),
        "random_state": kmeans_config.get('training', {}).get('random_state', 42),
        "auto_k": hyperparams.get('auto_k', True),
        "recommended_k": best_k if auto_k_stats else k
    }
    scratch_mlflow_metrics = {
        "inertia": scratch_metrics.get('Inertia', 0),
        "silhouette": scratch_metrics.get('Silhouette', 0),
        "davies_bouldin": scratch_metrics.get('Davies-Bouldin', 0),
        "calinski_harabasz": scratch_metrics.get('Calinski-Harabasz', 0),
        "cv_mean_silhouette": cv_scratch_mean,
        "cv_std_silhouette": cv_scratch_std
    }
    
    with MLflowTracker(experiment_name) as tracker:
        tracker.start_run(run_name="KMeans_Scratch",
                         tags={"implementation": "scratch", "dataset": "store_customers",
                               "model_type": "clustering", "status": "experiment"})
        tracker.log_params(scratch_hyperparams)
        tracker.log_metrics(scratch_mlflow_metrics)
        tracker.log_metrics({"training_time_seconds": scratch_metrics.get('Time (s)', 0)})
        
        # Log training curve (inertia history)
        if hasattr(scratch_model, 'inertia_history_') and scratch_model.inertia_history_:
            tracker.log_training_curve(scratch_model.inertia_history_, "inertia")
        
        # Log dataset info
        tracker.log_dataset_info(X, None, "dataset")
        
        # Log plots
        if os.path.exists(os.path.join(output_dir, "elbow_curve.png")):
            tracker.log_plot(os.path.join(output_dir, "elbow_curve.png"), "plots")
        if os.path.exists(os.path.join(output_dir, "scratch_clusters.png")):
            tracker.log_plot(os.path.join(output_dir, "scratch_clusters.png"), "plots")
        if os.path.exists(os.path.join(output_dir, "inertia_history.png")):
            tracker.log_plot(os.path.join(output_dir, "inertia_history.png"), "plots")
        if config:
            tracker.log_dict(config, "config")
        print("    [OK] Scratch implementation logged to MLflow")
    
    # Track sklearn Implementation
    sklearn_hyperparams = {
        "n_clusters": k,
        "n_init": hyperparams.get('n_init', 10),
        "max_iter": sklearn_model.n_iter_,
        "tol": hyperparams.get('tol', 1e-4),
        "random_state": kmeans_config.get('training', {}).get('random_state', 42),
        "algorithm": "lloyd"
    }
    sklearn_mlflow_metrics = {
        "inertia": sklearn_metrics.get('Inertia', 0),
        "silhouette": sklearn_metrics.get('Silhouette', 0),
        "davies_bouldin": sklearn_metrics.get('Davies-Bouldin', 0),
        "calinski_harabasz": sklearn_metrics.get('Calinski-Harabasz', 0),
        "cv_mean_silhouette": cv_sklearn_mean,
        "cv_std_silhouette": cv_sklearn_std
    }
    
    with MLflowTracker(experiment_name) as tracker:
        tracker.start_run(run_name="KMeans_sklearn",
                         tags={"implementation": "sklearn", "dataset": "store_customers",
                               "model_type": "clustering", "status": "experiment"})
        tracker.log_params(sklearn_hyperparams)
        tracker.log_metrics(sklearn_mlflow_metrics)
        tracker.log_metrics({"training_time_seconds": sklearn_metrics.get('Time (s)', 0)})
        
        # Log dataset info
        tracker.log_dataset_info(X, None, "dataset")
        
        # Log model with signature and register if best
        try:
            signature = infer_signature(X[:10], sklearn_model.predict(X[:10]))
            # Log model first
            tracker.log_model(sklearn_model, "model", signature=signature,
                            input_example=X[:5])
            
            # Register best model (based on silhouette score) - after logging
            if sklearn_metrics.get('Silhouette', 0) >= scratch_metrics.get('Silhouette', 0):
                # Model URI is relative to the current run
                model_uri = f"runs:/{tracker.active_run.info.run_id}/model"
                mv = tracker.register_model(
                    model_name="KMeans_sklearn",
                    model_uri=model_uri,
                    stage="Staging",
                    tags={"algorithm": "kmeans", "best_metric": "silhouette",
                          "silhouette": str(sklearn_metrics.get('Silhouette', 0))}
                )
                if mv:
                    print(f"    [OK] Model registered: KMeans_sklearn v{mv.version} (Staging)")
        except Exception as e:
            print(f"    Note: Model logging/registration skipped: {e}")
        
        # Log plots
        if os.path.exists(os.path.join(output_dir, "elbow_curve.png")):
            tracker.log_plot(os.path.join(output_dir, "elbow_curve.png"), "plots")
        if os.path.exists(os.path.join(output_dir, "sklearn_clusters.png")):
            tracker.log_plot(os.path.join(output_dir, "sklearn_clusters.png"), "plots")
        if config:
            tracker.log_dict(config, "config")
        print("    [OK] sklearn implementation logged to MLflow")
    
    # Final Summary
    print("\n" + "="*90)
    print("  FINAL SUMMARY")
    print("="*90)
    
    scratch_wins = []
    sklearn_wins = []
    
    metrics_comparison = {
        'Inertia': ('low', scratch_metrics.get('Inertia'), sklearn_metrics.get('Inertia')),
        'Silhouette': ('high', scratch_metrics.get('Silhouette'), sklearn_metrics.get('Silhouette')),
        'Davies-Bouldin': ('low', scratch_metrics.get('Davies-Bouldin'), sklearn_metrics.get('Davies-Bouldin')),
        'Calinski-Harabasz': ('high', scratch_metrics.get('Calinski-Harabasz'), sklearn_metrics.get('Calinski-Harabasz')),
        'Time (s)': ('low', scratch_metrics.get('Time (s)'), sklearn_metrics.get('Time (s)'))
    }
    
    for metric, (direction, scratch_val, sklearn_val) in metrics_comparison.items():
        if not np.isnan(scratch_val) and not np.isnan(sklearn_val):
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
        print(f"  {'[OK] ' + ', '.join(scratch_wins)}")
    else:
        print(f"  None")
    
    print(f"\n  sklearn Wins:")
    if sklearn_wins:
        print(f"  {'[OK] ' + ', '.join(sklearn_wins)}")
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
