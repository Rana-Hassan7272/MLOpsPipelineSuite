"""
MLflow Tracking Module
======================
Production-ready MLflow tracking for all experiments.
Features:
- Model Registry integration
- Auto-logging for sklearn models
- Detailed metric tracking (training curves, confusion matrices)
- Model signatures and metadata
- Dataset information logging
"""

import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import json
import tempfile


class MLflowTracker:
    """MLflow experiment tracker for MLOps infrastructure."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Optional tracking URI.
                If None, we default to a SQLite DB at project_root/mlflow.db
                so that experiments and the `mlflow ui` process share the same backend.
        """
        # Ensure we always use the same backend as `mlflow ui` (sqlite:///mlflow.db in project root)
        if tracking_uri is None:
            env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if env_tracking_uri:
                tracking_uri = env_tracking_uri
            else:
            # Get project root: tracking/ is one level down from project root
                project_root = Path(__file__).resolve().parent.parent
                db_path = project_root / 'mlflow.db'
                # Use absolute path with forward slashes for SQLite URI
                tracking_uri = f"sqlite:///{str(db_path).replace(chr(92), '/')}"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.active_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags or {})
        return self.active_run
    
    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        # Convert all values to strings for MLflow compatibility
        params_str = {k: str(v) for k, v in params.items()}
        mlflow.log_params(params_str)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(key, float(value), step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as artifact."""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log a directory as artifacts."""
        if os.path.exists(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_model(self, model: Any, artifact_path: str = "model", 
                   signature: Optional[Any] = None, 
                   input_example: Optional[Any] = None,
                   registered_model_name: Optional[str] = None):
        """
        Log a model artifact with optional signature and registration.
        
        Args:
            model: Model object to log
            artifact_path: Path to store the model
            signature: MLflow model signature (input/output schema)
            input_example: Example input for the model
            registered_model_name: If provided, register model in Model Registry
        """
        # MLflow sklearn.log_model uses artifact_path as positional argument
        # The warning about 'name' is for other MLflow functions, not sklearn.log_model
        if signature is not None:
            mlflow.sklearn.log_model(
                model, artifact_path, 
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            mlflow.sklearn.log_model(
                model, artifact_path,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
    
    def register_model(self, model_name: str, model_uri: Optional[str] = None, 
                       stage: str = "None", tags: Optional[Dict[str, str]] = None):
        """
        Register a model in the Model Registry.
        
        Args:
            model_name: Name for the registered model
            model_uri: URI of the model artifact (defaults to current run's model)
            stage: Model stage (None, Staging, Production, Archived)
            tags: Tags to add to the registered model
        """
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        if model_uri is None:
            if not self.active_run:
                print("    Warning: No active run for model registration")
                return None
            # Use the current run's model
            model_uri = f"runs:/{self.active_run.info.run_id}/model"
        
        try:
            # Check if model already exists, if not create it
            try:
                existing_model = client.get_registered_model(model_name)
            except Exception:
                # Model doesn't exist, create it
                client.create_registered_model(model_name)
            
            # Create model version
            mv = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=self.active_run.info.run_id if self.active_run else None
            )
            
            if stage != "None":
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage=stage
                    )
                except Exception as e:
                    print(f"    Note: Could not transition to {stage}: {e}")
            
            if tags:
                for key, value in tags.items():
                    try:
                        client.set_model_version_tag(model_name, mv.version, key, str(value))
                    except Exception:
                        pass  # Tag setting is optional
            
            return mv
        except Exception as e:
            print(f"    Note: Model registration skipped: {e}")
            return None
    
    def log_training_curve(self, history: List[float], metric_name: str = "loss"):
        """
        Log training curve (e.g., loss over epochs).
        
        Args:
            history: List of metric values over training steps
            metric_name: Name of the metric (e.g., "loss", "accuracy")
        """
        for step, value in enumerate(history):
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(metric_name, float(value), step=step)
    
    def log_confusion_matrix(self, cm: np.ndarray, labels: Optional[List[str]] = None):
        """
        Log confusion matrix as artifact.
        
        Args:
            cm: Confusion matrix array
            labels: Optional class labels
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        try:
            mlflow.log_artifact(temp_path, "confusion_matrix")
        finally:
            os.unlink(temp_path)
    
    def log_dataset_info(self, X: Any, y: Optional[Any] = None, 
                        dataset_name: str = "dataset"):
        """
        Log dataset information (size, shape, class distribution).
        
        Args:
            X: Feature matrix
            y: Optional target vector
            dataset_name: Name for the dataset
        """
        dataset_info = {
            "n_samples": X.shape[0] if hasattr(X, 'shape') else len(X),
            "n_features": X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else None,
            "sparse": hasattr(X, 'toarray') if hasattr(X, 'toarray') else False
        }
        
        if y is not None:
            y_array = y.flatten() if hasattr(y, 'flatten') else np.array(y)
            unique, counts = np.unique(y_array, return_counts=True)
            class_dist = dict(zip([str(u) for u in unique], counts.tolist()))
            dataset_info["class_distribution"] = class_dist
            dataset_info["n_classes"] = len(unique)
        
        mlflow.log_dict(dataset_info, f"{dataset_name}_info.json")
        
        # Also log as tags for easy filtering
        mlflow.set_tag(f"{dataset_name}_n_samples", str(dataset_info["n_samples"]))
        if dataset_info["n_features"]:
            mlflow.set_tag(f"{dataset_name}_n_features", str(dataset_info["n_features"]))
    
    def log_model_signature(self, input_schema: Any, output_schema: Any):
        """
        Log model signature (input/output schema).
        
        Args:
            input_schema: MLflow schema for inputs
            output_schema: MLflow schema for outputs
        """
        from mlflow.models import ModelSignature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        return signature
    
    def log_plot(self, plot_path: str, artifact_path: str = "plots"):
        """Log a plot image."""
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the current run."""
        mlflow.set_tags(tags)
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str = "config"):
        """Log a dictionary as YAML artifact."""
        import yaml
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(dictionary, f, default_flow_style=False)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, artifact_path)
        finally:
            os.unlink(temp_path)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
    
    @staticmethod
    def enable_sklearn_autolog():
        """Enable MLflow auto-logging for sklearn models."""
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True,
            log_datasets=True
        )
    
    @staticmethod
    def disable_sklearn_autolog():
        """Disable MLflow auto-logging for sklearn models."""
        mlflow.sklearn.autolog(disable=True)


def track_experiment(
    experiment_name: str,
    model_name: str,
    implementation: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    training_time: float,
    plots_dir: Optional[str] = None,
    model_artifact: Optional[Any] = None,
    additional_tags: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to track a complete experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        model_name: Name of the model (e.g., "Logistic Regression")
        implementation: "Scratch" or "sklearn"
        hyperparameters: Dictionary of hyperparameters
        metrics: Dictionary of evaluation metrics
        training_time: Training time in seconds
        plots_dir: Directory containing plot files to log
        model_artifact: Model object to log (optional)
        additional_tags: Additional tags to log
        config: Full configuration dictionary to log
    """
    tags = {
        "model": model_name,
        "implementation": implementation,
        "algorithm": model_name
    }
    if additional_tags:
        tags.update(additional_tags)
    
    with MLflowTracker(experiment_name) as tracker:
        run_name = f"{model_name}_{implementation}"
        tracker.start_run(run_name=run_name, tags=tags)
        
        # Log hyperparameters
        tracker.log_params(hyperparameters)
        
        # Log metrics
        tracker.log_metrics(metrics)
        tracker.log_metric("training_time_seconds", training_time)
        
        # Log configuration if provided
        if config:
            tracker.log_dict(config, "config")
        
        # Log plots if directory provided
        if plots_dir and os.path.exists(plots_dir):
            for plot_file in os.listdir(plots_dir):
                if plot_file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    tracker.log_plot(
                        os.path.join(plots_dir, plot_file),
                        "plots"
                    )
        
        # Log model artifact if provided
        if model_artifact:
            tracker.log_model(model_artifact, "model")
        
        return tracker.active_run.info.run_id
