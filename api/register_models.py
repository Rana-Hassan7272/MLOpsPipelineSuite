"""
Script to register and promote models to Model Registry.
Run this after training models to make them available for the API.
"""

import os
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set MLflow tracking URI
mlflow_tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"
mlflow.set_tracking_uri(mlflow_tracking_uri)

def register_model_from_run(experiment_name: str, run_name: str, model_name: str, stage: str = "Staging"):
    """
    Register a model from an experiment run to Model Registry.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the run containing the model
        model_name: Name for the registered model
        stage: Stage to promote to (Staging, Production, etc.)
    """
    client = MlflowClient()
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return False
        
        # Search for runs with the specified name
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=10,  # Try multiple runs
            order_by=["start_time DESC"]
        )
        
        if len(runs) == 0:
            print(f"❌ Run '{run_name}' not found in experiment '{experiment_name}'")
            return False
        
        # Try each run until we find one with a model
        run_id = None
        for idx, run in runs.iterrows():
            test_run_id = run['run_id']
            model_uri = f"runs:/{test_run_id}/model"
            
            # Check if model exists in this run
            try:
                mlflow.sklearn.load_model(model_uri)
                run_id = test_run_id
                print(f"  Found model in run {test_run_id[:8]}...")
                break
            except Exception:
                continue  # Try next run
        
        if run_id is None:
            print(f"❌ No model found in any '{run_name}' runs")
            return False
        
        model_uri = f"runs:/{run_id}/model"
        
        # Create registered model if it doesn't exist
        try:
            client.get_registered_model(model_name)
            print(f"✓ Registered model '{model_name}' already exists")
        except Exception:
            client.create_registered_model(model_name)
            print(f"✓ Created registered model '{model_name}'")
        
        # Create model version
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        print(f"✓ Created model version {mv.version} for '{model_name}'")
        
        # Transition to stage
        if stage != "None":
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=stage
                )
                print(f"✓ Promoted '{model_name}' v{mv.version} to {stage}")
            except Exception as e:
                print(f"⚠ Could not promote to {stage}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return False


def list_registered_models():
    """List all registered models."""
    client = MlflowClient()
    try:
        models = client.search_registered_models()
        if len(models) == 0:
            print("No registered models found.")
            return
        
        print("\n" + "="*60)
        print("Registered Models:")
        print("="*60)
        for model in models:
            print(f"\n📦 {model.name}")
            for version in model.latest_versions:
                print(f"   Version {version.version}: {version.current_stage}")
    except Exception as e:
        print(f"Error listing models: {e}")


if __name__ == "__main__":
    print("="*60)
    print("MLflow Model Registry - Register Models")
    print("="*60)
    
    # Register Logistic Regression
    print("\n[1] Registering LogisticRegression_sklearn...")
    register_model_from_run(
        experiment_name="Logistic_Regression_Comparison",
        run_name="Logistic_Regression_sklearn",
        model_name="LogisticRegression_sklearn",
        stage="Staging"
    )
    
    # Register K-Means
    print("\n[2] Registering KMeans_sklearn...")
    register_model_from_run(
        experiment_name="KMeans_Comparison",
        run_name="KMeans_sklearn",
        model_name="KMeans_sklearn",
        stage="Staging"
    )
    
    # Register Isolation Forest
    print("\n[3] Registering IsolationForest_sklearn...")
    # Try to find any sklearn run with a model in Isolation Forest experiment
    success = register_model_from_run(
        experiment_name="Isolation_Forest_Comparison",
        run_name="Isolation_Forest_sklearn",
        model_name="IsolationForest_sklearn",
        stage="Staging"
    )
    
    # If that failed, try to find any run with sklearn implementation
    if not success:
        print("  Trying alternative: searching for any sklearn run...")
        try:
            experiment = mlflow.get_experiment_by_name("Isolation_Forest_Comparison")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.implementation = 'sklearn'",
                    max_results=10,
                    order_by=["start_time DESC"]
                )
                
                for idx, run in runs.iterrows():
                    run_id = run['run_id']
                    model_uri = f"runs:/{run_id}/model"
                    try:
                        mlflow.sklearn.load_model(model_uri)
                        # Found a model, register it
                        client = MlflowClient()
                        try:
                            client.get_registered_model("IsolationForest_sklearn")
                        except:
                            client.create_registered_model("IsolationForest_sklearn")
                        
                        mv = client.create_model_version(
                            name="IsolationForest_sklearn",
                            source=model_uri,
                            run_id=run_id
                        )
                        print(f"✓ Registered IsolationForest_sklearn v{mv.version} from run {run_id[:8]}...")
                        break
                    except:
                        continue
        except Exception as e:
            print(f"  ⚠ Could not find alternative run: {e}")
    
    # List all registered models
    print("\n" + "="*60)
    list_registered_models()
    print("\n" + "="*60)
    print("✓ Done! Models are now registered and available for the API.")
    print("="*60)
