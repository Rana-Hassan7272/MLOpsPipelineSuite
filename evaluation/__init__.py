"""Evaluation module for model metrics and cross-validation."""

from .metrics import (
    evaluate_model,
    print_confusion_matrix_analysis,
    plot_roc_curves,
    plot_training_curve,
    analyze_complexity,
    explain_metric_differences,
    print_comparison_table,
    # Clustering metrics
    print_cluster_analysis,
    plot_elbow_curve,
    plot_cluster_visualization,
    plot_inertia_history,
    analyze_complexity_clustering,
    explain_metric_differences_clustering,
    # Anomaly detection metrics
    plot_pr_curve,
    plot_score_distributions,
    print_confusion_matrix_analysis_fraud,
    analyze_complexity_isolation_forest,
    explain_metric_differences_isolation_forest
)

from .cross_validation import cross_validate_model, cross_validate_kmeans, cross_validate_isolation_forest

__all__ = [
    'evaluate_model',
    'print_confusion_matrix_analysis',
    'plot_roc_curves',
    'plot_training_curve',
    'analyze_complexity',
    'explain_metric_differences',
    'print_comparison_table',
    'cross_validate_model',
    # Clustering
    'print_cluster_analysis',
    'plot_elbow_curve',
    'plot_cluster_visualization',
    'plot_inertia_history',
    'analyze_complexity_clustering',
    'explain_metric_differences_clustering',
    'cross_validate_kmeans',
    # Anomaly detection
    'plot_pr_curve',
    'plot_score_distributions',
    'print_confusion_matrix_analysis_fraud',
    'analyze_complexity_isolation_forest',
    'explain_metric_differences_isolation_forest',
    'cross_validate_isolation_forest'
]
