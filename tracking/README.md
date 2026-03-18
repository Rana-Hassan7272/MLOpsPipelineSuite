# MLflow Tracking - Production Features

## Features Implemented

### 1. **Model Registry Integration**
- Register models with versioning
- Model staging (None, Staging, Production, Archived)
- Model metadata and tags
- Automatic best model selection and registration

### 2. **Auto-Logging for sklearn**
- Enable/disable sklearn auto-logging
- Automatic parameter, metric, and model logging
- Model signatures and input examples

### 3. **Enhanced Metric Tracking**
- **Training Curves**: Log loss/metrics over epochs/steps
- **Confusion Matrices**: Visual confusion matrices as artifacts
- **Dataset Info**: Log dataset size, shape, class distribution
- **Cross-Validation Metrics**: Mean and std for CV results

### 4. **Model Signatures**
- Input/output schema inference
- Type validation
- Example inputs for testing

### 5. **Production-Ready Features**
- Comprehensive tags for filtering
- Organized artifact structure
- Model versioning
- Best model tracking

## Usage

### Basic Tracking
```python
from tracking.mlflow_tracker import MLflowTracker

with MLflowTracker("MyExperiment") as tracker:
    tracker.start_run(run_name="my_run", tags={"model": "logistic"})
    tracker.log_params({"learning_rate": 0.1})
    tracker.log_metrics({"accuracy": 0.95})
```

### Training Curves
```python
tracker.log_training_curve(loss_history, "training_loss")
```

### Confusion Matrix
```python
tracker.log_confusion_matrix(cm, labels=["Class 0", "Class 1"])
```

### Dataset Info
```python
tracker.log_dataset_info(X_train, y_train, "train")
```

### Model Registration
```python
tracker.log_model(model, "model", signature=signature)
mv = tracker.register_model(
    model_name="MyModel",
    stage="Production",
    tags={"version": "1.0"}
)
```

### Auto-Logging (sklearn)
```python
MLflowTracker.enable_sklearn_autolog()
# Train sklearn model - automatically logged
model.fit(X, y)
MLflowTracker.disable_sklearn_autolog()
```

## Viewing in MLflow UI

1. Start MLflow UI:
```bash
mlflow ui --backend-store-uri "sqlite:///path/to/mlflow.db"
```

2. Access at: `http://127.0.0.1:5000`

3. View:
   - **Experiments**: All experiment runs
   - **Models**: Registered models in Model Registry
   - **Artifacts**: Plots, configs, confusion matrices
   - **Metrics**: Training curves, evaluation metrics

## Model Registry

Registered models can be:
- **Staged**: None → Staging → Production → Archived
- **Versioned**: Automatic versioning on registration
- **Tagged**: Custom metadata for filtering
- **Deployed**: Load models by name and version
