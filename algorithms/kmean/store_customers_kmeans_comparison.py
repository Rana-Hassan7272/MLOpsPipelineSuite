"""
Store Customers Clustering: Comprehensive K-Means Comparison
==========================================================
Compares scratch implementation vs sklearn on store_customers dataset with:
- Cluster Analysis
- Elbow Curve + Silhouette Analysis
- Cross Validation
- Training Curves (Inertia)
- Complexity Analysis
- Metric Differences Explanation
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.decomposition import PCA

# Import scratch implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hassan_kmeans import HassanKMeans, full_evaluation, hassan_auto_k


def load_store_customers_data():
    """Load preprocessed store_customers dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, 'data_pipeline', 'processed_data', 
                             'store_customers_processed.csv')
    
    df = pd.read_csv(data_path)
    X = df.values.astype(np.float64)
    
    return X, df


def print_cluster_analysis(labels, centroids, model_name, X):
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


def plot_elbow_curve(auto_k_stats, model_name):
    """Plot elbow curve with silhouette scores."""
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'elbow_curve.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Elbow Curve saved → elbow_curve.png")
    plt.close()


def plot_cluster_visualization(X_2d, labels, centroids_2d, model_name, save_name):
    """Plot 2D cluster visualization."""
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, save_name), dpi=150, bbox_inches='tight')
    print(f"  ✓ Cluster visualization saved → {save_name}")
    plt.close()


def plot_inertia_history(inertia_history, model_name):
    """Plot inertia across restarts."""
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'inertia_history.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Inertia history saved → inertia_history.png")
    plt.close()


def cross_validate_kmeans(X, k, n_splits=5):
    """Perform k-fold cross validation for K-means."""
    print(f"\n{'='*70}")
    print(f"  {n_splits}-FOLD CROSS VALIDATION")
    print(f"{'='*70}")
    
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scratch_scores = []
    sklearn_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        
        # Scratch implementation
        scratch_model = HassanKMeans(n_clusters=k, n_init=5, random_state=42, verbose=False)
        scratch_model.fit(X_train_fold)
        scratch_labels = scratch_model.predict(X_val_fold)
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


def explain_metric_differences(scratch_metrics, sklearn_metrics):
    """Explain why metrics differ between implementations."""
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


def print_comparison_table(results):
    """Print structured comparison table."""
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


if __name__ == "__main__":
    print("="*90)
    print("  COMPREHENSIVE STORE CUSTOMERS CLUSTERING - K-MEANS COMPARISON")
    print("="*90)
    
    # Load data
    print("\n[1] Loading preprocessed store_customers dataset...")
    X, df = load_store_customers_data()
    print(f"    Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"    Feature names: {list(df.columns)}")
    
    # Auto-k selection
    print("\n[2] Auto-k selection via elbow method and gap statistic...")
    best_k, auto_k_stats = hassan_auto_k(X, k_range=range(2, 11), n_refs=3, random_state=42)
    print(f"    Recommended k = {best_k} (gap statistic)")
    print(f"    Using k = 5 for comparison (common choice for customer segmentation)")
    k = 5
    
    # Train scratch implementation
    print(f"\n[3] Training K-Means (Scratch Implementation)...")
    scratch_model = HassanKMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        verbose=True
    )
    start_time = time.time()
    scratch_model.fit(X)
    scratch_time = time.time() - start_time
    
    scratch_metrics = full_evaluation(
        "Scratch Implementation", X, scratch_model.labels_,
        scratch_model.cluster_centers_, scratch_model.inertia_,
        scratch_time, y_true=None
    )
    scratch_metrics['Algorithm'] = 'K-Means'
    scratch_metrics['Implementation'] = 'Scratch'
    
    print(f"    ✓ Training completed in {scratch_time:.4f}s")
    print(f"    ✓ Best inertia: {scratch_model.inertia_:.2f}")
    print(f"    ✓ Iterations: {scratch_model.n_iter_}")
    
    # Train sklearn implementation
    print(f"\n[4] Training sklearn KMeans...")
    sklearn_model = SklearnKMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        algorithm='lloyd'
    )
    start_time = time.time()
    sklearn_model.fit(X)
    sklearn_time = time.time() - start_time
    
    sklearn_metrics = full_evaluation(
        "sklearn KMeans", X, sklearn_model.labels_,
        sklearn_model.cluster_centers_, sklearn_model.inertia_,
        sklearn_time, y_true=None
    )
    sklearn_metrics['Algorithm'] = 'K-Means'
    sklearn_metrics['Implementation'] = 'sklearn'
    
    print(f"    ✓ Training completed in {sklearn_time:.4f}s")
    print(f"    ✓ Best inertia: {sklearn_model.inertia_:.2f}")
    print(f"    ✓ Iterations: {sklearn_model.n_iter_}")
    
    # Print comparison table
    results = [scratch_metrics, sklearn_metrics]
    print_comparison_table(results)
    
    # Cluster Analysis
    print_cluster_analysis(scratch_model.labels_, scratch_model.cluster_centers_, 
                          "Scratch Implementation", X)
    print_cluster_analysis(sklearn_model.labels_, sklearn_model.cluster_centers_,
                          "sklearn", X)
    
    # Elbow Curve
    print("\n[5] Generating Elbow Curve...")
    plot_elbow_curve(auto_k_stats, "Store Customers")
    
    # Cluster Visualizations
    print("\n[6] Generating Cluster Visualizations...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    scratch_centroids_2d = pca.transform(scratch_model.cluster_centers_)
    sklearn_centroids_2d = pca.transform(sklearn_model.cluster_centers_)
    
    plot_cluster_visualization(X_2d, scratch_model.labels_, scratch_centroids_2d,
                               "Scratch Implementation", "scratch_clusters.png")
    plot_cluster_visualization(X_2d, sklearn_model.labels_, sklearn_centroids_2d,
                               "sklearn", "sklearn_clusters.png")
    
    # Inertia History
    print("\n[7] Generating Inertia History...")
    plot_inertia_history(scratch_model.inertia_history_, "Scratch Implementation")
    
    # Cross Validation
    print("\n[8] Performing Cross Validation...")
    cv_scratch_mean, cv_scratch_std, cv_sklearn_mean, cv_sklearn_std = cross_validate_kmeans(X, k)
    
    # Complexity Analysis
    analyze_complexity(X, scratch_time, sklearn_time)
    
    # Explain Metric Differences
    explain_metric_differences(scratch_metrics, sklearn_metrics)
    
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
