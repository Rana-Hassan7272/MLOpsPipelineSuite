"""
Model Trainer
=============
Wrapper for training models with consistent interface.
"""

import time
import numpy as np
from typing import Optional, Dict, Any


class ModelTrainer:
    """
    Trainer class for model training with consistent interface.
    
    Responsibilities:
    - Load dataset
    - Train model
    - Evaluate model
    - Save model
    """
    
    def __init__(self, model, model_name: str = "Model"):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : object
            Model instance to train
        model_name : str
            Name of the model for logging
        """
        self.model = model
        self.model_name = model_name
        self.training_time = 0.0
        self.training_history = {}
    
    def train(self, X_train, y_train=None, **kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like, optional
            Training labels (required for supervised learning, optional for unsupervised)
        **kwargs : dict
            Additional training parameters
        """
        print(f"\n[Training {self.model_name}...]")
        start_time = time.time()
        
        # Train model - handle both supervised and unsupervised cases
        if y_train is not None:
            # Supervised learning (classification/regression)
            y_train_flat = y_train.flatten() if hasattr(y_train, 'flatten') else y_train
            self.model.fit(X_train, y_train_flat, **kwargs)
        else:
            # Unsupervised learning (clustering)
            self.model.fit(X_train, **kwargs)
        
        self.training_time = time.time() - start_time
        print(f"  ✓ Training completed in {self.training_time:.4f}s")
        
        # Store training history if available
        if hasattr(self.model, 'loss_history'):
            self.training_history['loss'] = self.model.loss_history
        
        return self.model
    
    def predict(self, X_test, **kwargs):
        """
        Make predictions.
        
        Parameters
        ----------
        X_test : array-like
            Test features
        **kwargs : dict
            Additional prediction parameters
        """
        return self.model.predict(X_test, **kwargs)
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities.
        
        Parameters
        ----------
        X_test : array-like
            Test features
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        else:
            raise AttributeError(f"{self.model_name} does not support predict_proba")
    
    def get_training_time(self) -> float:
        """Get training time in seconds."""
        return self.training_time
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.training_history
