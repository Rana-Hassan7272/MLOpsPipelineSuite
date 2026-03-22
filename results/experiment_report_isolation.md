================================================================================
==========
  COMPREHENSIVE CREDIT CARD FRAUD DETECTION - ISOLATION FOREST COMPARISON
================================================================================
==========

[1] Loading preprocessed creditcard dataset...
    Train Class distribution: Normal=226602, Fraud=378
    Test Class distribution: Normal=56651, Fraud=95
    Train Class unique values: [0 1]
    Test Class unique values: [0 1]
    Training samples: 226980, Features: 30
    Test samples: 56746
    Fraud rate: 0.0017 (473/283726)
    Normal: 283253, Fraud: 473
    Actual fraud rate in data: 0.0017 (473/283726)
  Industry production-optimized: Using contamination 0.0167 (10x actual rate)
  Target: 80%+ recall - prioritize fraud detection over false positives
    Using contamination: 0.0167

[2] Training Isolation Forest (Scratch Implementation)...
    Using industry production-optimized parameters (target: 80%+ recall)...
  HassanIsolationForest | trees=400 | sub=256 | depth≤9 | feats=30/30 | extended
=True
  ✓ 400 trees built
    ✓ Threshold optimized for 84.4% recall: 0.4897

[3] Training sklearn IsolationForest...
    Using same contamination for fair comparison...
    ✓ Training completed in 36.2151s
    ✓ Threshold: 0.4724

  Debug: y_test unique values: [0 1]
  Debug: scratch_preds unique values: [0 1]
  Debug: sklearn_preds unique values: [0 1]

================================================================================
==========
  ISOLATION FOREST COMPARISON - CREDIT CARD FRAUD DETECTION
================================================================================
==========

Algorithm                 Implementation  ROC-AUC      Avg Precision   Precision
    Recall       F1           Training Time
--------------------------------------------------------------------------------
----------
Isolation Forest          Scratch         0.9409       0.1705          0.0356
    0.8105       0.0682       10.9714        s
Isolation Forest          sklearn         0.9364       0.1224          0.0628
    0.6632       0.1148       36.2151        s
================================================================================
==========

======================================================================
  CONFUSION MATRIX ANALYSIS - Scratch Implementation
======================================================================

  Confusion Matrix:
                           Predicted Normal      Predicted Fraud
         Actual Normal                54564                 2087
          Actual Fraud                   18                   77

  Detailed Breakdown:
  True Negatives (TN)           :  54564 - Correctly predicted normal transactio
ns
  False Positives (FP)          :   2087 - Normal transactions flagged as fraud
(Type I Error)
  False Negatives (FN)          :     18 - Fraud transactions missed (Type II Er
ror)
  True Positives (TP)           :     77 - Correctly detected fraud

  Error Analysis:
  False Positive Rate           : 0.0368 (2087/56651)
  False Negative Rate           : 0.1895 (18/95)
  Total Errors                  :   2105/56746 (3.71%)
  Accuracy                      : 0.9629

======================================================================
  CONFUSION MATRIX ANALYSIS - sklearn
======================================================================

  Confusion Matrix:
                           Predicted Normal      Predicted Fraud
         Actual Normal                55711                  940
          Actual Fraud                   32                   63

  Detailed Breakdown:
  True Negatives (TN)           :  55711 - Correctly predicted normal transactio
ns
  False Positives (FP)          :    940 - Normal transactions flagged as fraud
(Type I Error)
  False Negatives (FN)          :     32 - Fraud transactions missed (Type II Er
ror)
  True Positives (TP)           :     63 - Correctly detected fraud

  Error Analysis:
  False Positive Rate           : 0.0166 (940/56651)
  False Negative Rate           : 0.3368 (32/95)
  Total Errors                  :    972/56746 (1.71%)
  Accuracy                      : 0.9829

[4] Generating ROC and PR Curves...
  ✓ ROC Curve saved → roc_curve_comparison.png
  ✓ PR Curve saved → pr_curve_comparison.png

[5] Generating Score Distributions...
  ✓ Score Distributions saved → score_distributions.png

[6] Performing Cross Validation...

======================================================================
  5-FOLD STRATIFIED CROSS VALIDATION
======================================================================
  Fold 1: Scratch AUC=0.9543, AP=0.1940 | sklearn AUC=0.9478, AP=0.1373
  Fold 2: Scratch AUC=0.9642, AP=0.2004 | sklearn AUC=0.9609, AP=0.1230
  Fold 3: Scratch AUC=0.9394, AP=0.1378 | sklearn AUC=0.9351, AP=0.0987
  Fold 4: Scratch AUC=0.9288, AP=0.1076 | sklearn AUC=0.9250, AP=0.0771
  Fold 5: Scratch AUC=0.9739, AP=0.2807 | sklearn AUC=0.9687, AP=0.1787

  Cross-Validation Results:
  Metric                    Scratch              sklearn
  -----------------------------------------------------------------
  ROC-AUC (Mean ± Std)      0.9521 ± 0.0163  0.9475 ± 0.0161
  Avg Precision (Mean ± Std) 0.1841 ± 0.0595  0.1230 ± 0.0347
  95% CI Scratch (AUC)      [0.9202, 0.9841]
  95% CI sklearn (AUC)      [0.9160, 0.9790]

======================================================================
  COMPLEXITY ANALYSIS
======================================================================

  Dataset Characteristics:
  Samples                  : 226,980
  Features                 : 30

  Time Complexity:
  Scratch Implementation   : O(n_estimators × max_samples × log(max_samples))
  sklearn IsolationForest  : O(n_estimators × max_samples × log(max_samples))
  Note                     : Both use Isolation Tree construction

  Training Time Comparison:
  Scratch                  : 10.9714s
  sklearn                  : 36.2151s
  Speedup Factor           : 3.30x faster

  Space Complexity:
  Scratch                  : O(n_estimators × max_nodes × features)
  sklearn                  : O(n_estimators × max_nodes × features)
  Memory Usage             : Similar for both implementations

  Optimization Techniques:
  Scratch                  : Parallel tree building, vectorized scoring, flat tr
ee storage
  sklearn                  : Cython-optimized, BLAS acceleration, single-threade
d by default

======================================================================
  METRIC DIFFERENCES ANALYSIS
======================================================================

  Key Differences:
  Metric                    Scratch         sklearn         Difference
  ----------------------------------------------------------------------
  ROC-AUC                   0.9409          0.9364          +0.0045
  Avg Precision             0.1705          0.1224          +0.0481
  Precision                 0.0356          0.0628          -0.0272
  Recall                    0.8105          0.6632          +0.1474
  F1                        0.0682          0.1148          -0.0466
  Time (s)                  10.9714         36.2151         -25.2437

  Possible Reasons for Differences:
  1. Split Selection:
     - Scratch: Extended Isolation Forest (hyperplane cuts)
     - sklearn: Classic axis-aligned splits
     - Extended splits reduce axis-aligned bias, improving detection

  2. Parallelization:
     - Scratch: Parallel tree building via joblib
     - sklearn: Single-threaded by default
     - Parallelization provides speed advantages

  3. Scoring Method:
     - Scratch: Vectorized batch scoring
     - sklearn: Optimized Cython implementation
     - Different implementations may converge to slightly different scores

  4. Threshold Selection:
     - Scratch: Auto-threshold with valley detection
     - sklearn: Percentile-based threshold
     - Different threshold methods affect precision/recall tradeoff

  5. Numerical Precision:
     - Scratch: NumPy float64 operations
     - sklearn: Cython with potential precision differences
     - Slight numerical differences in tree construction

================================================================================
==========
  FINAL SUMMARY
================================================================================
==========

  Scratch Implementation Wins:
  ✓ ROC-AUC, Avg Precision, Recall, Time (s)

  sklearn Wins:
  ✓ Precision, F1

  Overall Assessment:
  🏆 Scratch Implementation performs better overall!
================================================================================
==========

  Done. Isolation Forest comparison completed. ✓
