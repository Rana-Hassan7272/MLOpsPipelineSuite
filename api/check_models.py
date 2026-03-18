"""
Quick script to check what models are available in experiments and registry.
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

def check_experiments():
    """Check what experiments and runs exist."""
    print("="*60)
    print("Available Experiments and Runs:")
    print("="*60)
    
    experiments = [
        "Logistic_Regression_Comparison",
        "KMeans_Comparison",
        "Isolation_Forest_Comparison"
    ]
    
    client = MlflowClient()
    
    for exp_name in experiments:
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if not experiment:
                print(f"\n❌ {exp_name}: Not found")
                continue
            
            print(f"\n✓ {exp_name} (ID: {experiment.experiment_id})")
            
            # Search for sklearn runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.implementation = 'sklearn'",
                max_results=5
            )
            
            if len(runs) > 0:
                for idx, run in runs.iterrows():
                    run_name = run.get('tags.mlflow.runName', run['run_id'])
                    print(f"   - Run: {run_name}")
                    
                    # Check if model exists
                    model_uri = f"runs:/{run['run_id']}/model"
                    try:
                        mlflow.sklearn.load_model(model_uri)
                        print(f"     ✓ Model available")
                    except:
                        print(f"     ❌ Model not found")
            else:
                print("   No sklearn runs found")
                
        except Exception as e:
            print(f"\n❌ {exp_name}: Error - {e}")

def check_registry():
    """Check what models are in the registry."""
    print("\n" + "="*60)
    print("Model Registry:")
    print("="*60)
    
    client = MlflowClient()
    
    try:
        models = client.search_registered_models()
        if len(models) == 0:
            print("No models registered yet.")
            print("\n💡 Run: python api/register_models.py")
        else:
            for model in models:
                print(f"\n📦 {model.name}")
                for version in model.latest_versions:
                    print(f"   Version {version.version}: {version.current_stage}")
    except Exception as e:
        print(f"Error checking registry: {e}")

if __name__ == "__main__":
    check_experiments()
    check_registry()
    print("\n" + "="*60)
