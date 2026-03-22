==========
  COMPREHENSIVE SPAM DETECTION - LOGISTIC REGRESSION COMPARISON
================================================================================
==========

[1] Loading preprocessed spam dataset...
    Train: 4736 samples, 5000 features
    Test:  836 samples
    Class distribution - Train: [4097  639]
    Class distribution - Test:  [728 108]

[2] Training Logistic Regression (Scratch Implementation)...
    ✓ Training completed in 0.48s
    ✓ Epochs: 151
    ✓ Optimal threshold: 0.5400

[3] Training sklearn LogisticRegression...
    ✓ Training completed in 0.04s

================================================================================
==========
  LOGISTIC REGRESSION COMPARISON - SPAM DETECTION
================================================================================
==========

Algorithm                 Implementation  Accuracy     Precision    Recall
 F1 Score     AUC          Training Time
--------------------------------------------------------------------------------
----------
Logistic Regression       Scratch         0.9725       0.9208       0.8611
 0.8900       0.9867       0.48           s
Logistic Regression       sklearn         0.9617       0.9634       0.7315
 0.8316       0.9922       0.04           s
================================================================================
==========

======================================================================
  CONFUSION MATRIX ANALYSIS - Scratch Implementation
======================================================================

  Confusion Matrix:
                      Predicted 0     Predicted 1
         Actual 0             720               8
         Actual 1              15              93

  Detailed Breakdown:
  True Negatives (TN)           :   720 - Correctly predicted non-spam
  False Positives (FP)          :     8 - Non-spam misclassified as spam (Type I
 Error)
  False Negatives (FN)          :    15 - Spam misclassified as non-spam (Type I
I Error)
  True Positives (TP)           :    93 - Correctly predicted spam

  Error Analysis:
  False Positive Rate           : 0.0110 (8/728)
  False Negative Rate           : 0.1389 (15/108)
  Total Errors                  : 23/836 (2.75%)

======================================================================
  CONFUSION MATRIX ANALYSIS - sklearn
======================================================================

  Confusion Matrix:
                      Predicted 0     Predicted 1
         Actual 0             725               3
         Actual 1              29              79

  Detailed Breakdown:
  True Negatives (TN)           :   725 - Correctly predicted non-spam
  False Positives (FP)          :     3 - Non-spam misclassified as spam (Type I
 Error)
  False Negatives (FN)          :    29 - Spam misclassified as non-spam (Type I
I Error)
  True Positives (TP)           :    79 - Correctly predicted spam

  Error Analysis:
  False Positive Rate           : 0.0041 (3/728)
  False Negative Rate           : 0.2685 (29/108)
  Total Errors                  : 32/836 (3.83%)

[4] Generating ROC Curve...

  ✓ ROC Curve saved → roc_curve_comparison.png
    Scratch AUC: 0.9867
    sklearn AUC: 0.9922
    AUC Difference: -0.0055

[5] Generating Training Curve...
  ✓ Training Curve saved → training_curve.png
    Initial Loss: 1.071868
    Final Loss: 0.461700
    Loss Reduction: 0.610168

[6] Performing Cross Validation...

======================================================================
  5-FOLD CROSS VALIDATION - Scratch Implementation
======================================================================
  Fold 1: Accuracy = 0.9768
  Fold 2: Accuracy = 0.9694
  Fold 3: Accuracy = 0.9725
  Fold 4: Accuracy = 0.9715
  Fold 5: Accuracy = 0.9768

  Cross-Validation Results:
  Mean Accuracy            : 0.9734
  Standard Deviation       : 0.0029
  95% Confidence Interval  : [0.9676, 0.9792]

======================================================================
  5-FOLD CROSS VALIDATION - sklearn
======================================================================
  Fold 1: Accuracy = 0.9578
  Fold 2: Accuracy = 0.9504
  Fold 3: Accuracy = 0.9514
  Fold 4: Accuracy = 0.9504
  Fold 5: Accuracy = 0.9567

  Cross-Validation Results:
  Mean Accuracy            : 0.9533
  Standard Deviation       : 0.0032
  95% Confidence Interval  : [0.9470, 0.9597]

======================================================================
  COMPLEXITY ANALYSIS
======================================================================

  Dataset Characteristics:
  Samples                  : 4,736
  Features                 : 5,000
  Sparsity                 : 99.86%

  Time Complexity:
  Scratch Implementation   : O(epochs × n_samples × n_features)
  sklearn (LBFGS)          : O(iterations × n_samples × n_features)
  Note                     : sklearn uses optimized BLAS/LAPACK libraries

  Training Time Comparison:
  Scratch                  : 0.4787s
  sklearn                  : 0.0428s
  Speedup Factor           : 11.18x slower

  Space Complexity:
  Scratch                  : O(n_features) - stores weights + bias
  sklearn                  : O(n_features) - stores coefficients + intercept
  Memory Usage             : Similar for both implementations

======================================================================
  METRIC DIFFERENCES ANALYSIS
======================================================================

  Key Differences:
  Metric               Scratch         sklearn         Difference
  -----------------------------------------------------------------
  Accuracy             0.9725          0.9617          +0.0108
  Precision            0.9208          0.9634          -0.0426
  Recall               0.8611          0.7315          +0.1296
  F1 Score             0.8900          0.8316          +0.0584
  AUC                  0.9867          0.9922          -0.0055

  Possible Reasons for Differences:
  1. Regularization:
     - Scratch uses L2 regularization (λ = 0.05)
     - sklearn uses default regularization (C = 1.0, λ = 1/C = 1.0)
     - Different regularization strengths affect model complexity

  2. Optimization Method:
     - Scratch: Batch Gradient Descent with adaptive learning rate
     - sklearn: LBFGS (Limited-memory BFGS) - quasi-Newton method
     - LBFGS converges faster but may find different local minima

  3. Convergence Tolerance:
     - Scratch: Early stopping with patience (150 epochs)
     - sklearn: Convergence tolerance based on gradient norm
     - Different stopping criteria lead to different convergence points

  4. Numerical Precision:
     - Scratch: NumPy float64 operations
     - sklearn: Optimized BLAS/LAPACK with potential precision differences
     - Sparse matrix operations may have slight numerical differences

  5. Class Weight Handling:
     - Scratch: Automatic inverse frequency weighting
     - sklearn: Default balanced='auto' or equal weights
     - Different class weighting affects decision boundaries

  6. Initialization:
     - Scratch: Xavier/Glorot initialization with bias adjustment
     - sklearn: Default initialization (usually zeros)
     - Better initialization can lead to better convergence

================================================================================
==========
  FINAL SUMMARY
================================================================================
==========

  Scratch Implementation Wins:
  ✓ Accuracy, Recall, F1 Score

  sklearn Wins:
  ✓ Precision, AUC, Training Speed

  Overall Assessment:
  🤝 Both implementations are competitive!
================================================================================
==========
