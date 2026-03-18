"""
Training Pipeline
=================
Complete training pipeline for model training and evaluation.
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.trainer import ModelTrainer


class TrainingPipeline:
    """
    Complete training pipeline.
    
    Responsibilities:
    - Load dataset
    - Train model
    - Evaluate model
    - Save model
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize training pipeline.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}
        self.trainer = None
        self.model = None
    
    def load_data(self, data_loader_func):
        """
        Load dataset using provided loader function.
        
        Parameters
        ----------
        data_loader_func : callable
            Function that returns (X_train, X_test, y_train, y_test)
        """
        print("\n[1] Loading dataset...")
        X_train, X_test, y_train, y_test = data_loader_func()
        print(f"    Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"    Test:  {X_test.shape[0]} samples")
        print(f"    Class distribution - Train: {np.bincount(y_train.flatten())}")
        print(f"    Class distribution - Test:  {np.bincount(y_test.flatten())}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, model, X_train, y_train, model_name: str = "Model", **kwargs):
        """
        Train model using trainer.
        
        Parameters
        ----------
        model : object
            Model instance to train
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        model_name : str
            Name of the model
        **kwargs : dict
            Additional training parameters
        """
        self.trainer = ModelTrainer(model, model_name)
        self.model = self.trainer.train(X_train, y_train, **kwargs)
        return self.model
    
    def evaluate(self, X_test, y_test, evaluation_func):
        """
        Evaluate model.
        
        Parameters
        ----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        evaluation_func : callable
            Evaluation function that takes (y_true, y_pred, y_proba, model_name)
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.trainer.predict(X_test)
        y_proba = self.trainer.predict_proba(X_test)
        
        return evaluation_func(y_test.flatten(), y_pred, y_proba, self.trainer.model_name)
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        import pickle
        
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"  ✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load saved model.
        
        Parameters
        ----------
        filepath : str
            Path to saved model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"  ✓ Model loaded from {filepath}")
        return self.model
