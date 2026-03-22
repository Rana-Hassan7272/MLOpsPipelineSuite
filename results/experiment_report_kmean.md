==========
  COMPREHENSIVE STORE CUSTOMERS CLUSTERING - K-MEANS COMPARISON
================================================================================
==========

[1] Loading preprocessed store_customers dataset...
    Samples: 1000, Features: 4
    Feature names: ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-10
0)']

[2] Auto-k selection via elbow method and gap statistic...
    Recommended k = 3 (gap statistic)
    Using k = 5 for comparison (common choice for customer segmentation)

[3] Training K-Means (Scratch Implementation)...
  Hassan KMeans | k=5 | restarts=10 | max_iter=300
    restart  1/10  │  inertia=      604.31  │  iters=  26 ← best
    restart  2/10  │  inertia=      637.95  │  iters=  12
    restart  3/10  │  inertia=      668.91  │  iters=  10
    restart  4/10  │  inertia=      637.77  │  iters=  22
    restart  5/10  │  inertia=      601.36  │  iters=  28 ← best
    restart  6/10  │  inertia=      604.27  │  iters=  15
    restart  7/10  │  inertia=      601.36  │  iters=  32
    restart  8/10  │  inertia=      605.23  │  iters=  16
    restart  9/10  │  inertia=      645.51  │  iters=  22
    restart 10/10  │  inertia=      668.99  │  iters=  13
    ✓ Training completed in 0.1719s
    ✓ Best inertia: 601.36
    ✓ Iterations: 28

[4] Training sklearn KMeans...
    ✓ Training completed in 8.1511s
    ✓ Best inertia: 600.56
    ✓ Iterations: 7

================================================================================
==========
  K-MEANS CLUSTERING COMPARISON - STORE CUSTOMERS
================================================================================
==========

Algorithm                 Implementation  Inertia         Silhouette      Davies
-Bouldin     Calinski-Harabasz    Training Time
--------------------------------------------------------------------------------
----------
K-Means                   Scratch         601.36          0.3063          1.1079
             1095.33              0.1719         s
K-Means                   sklearn         600.56          0.3076          1.1098
             1097.14              8.1511         s
================================================================================
==========

======================================================================
  CLUSTER ANALYSIS - Scratch Implementation
======================================================================

  Cluster Distribution:
    Cluster 0:  223 samples (22.30%)
    Cluster 1:  119 samples (11.90%)
    Cluster 2:  228 samples (22.80%)
    Cluster 3:  250 samples (25.00%)
    Cluster 4:  180 samples (18.00%)

  Cluster Characteristics:

    Cluster 0:
      Size: 223
      Centroid: [ 0.         -0.78932675 -0.85523636  0.78591864]
      Mean distance to centroid: 0.6547
      Max distance to centroid: 1.4211

    Cluster 1:
      Size: 119
      Centroid: [ 0.45378151  2.1499539   1.79319413 -1.90977168]
      Mean distance to centroid: 0.8723
      Max distance to centroid: 1.5453

    Cluster 2:
      Size: 228
      Centroid: [ 1.         -0.68353784 -0.82875431  0.76203697]
      Mean distance to centroid: 0.6437
      Max distance to centroid: 1.9727

    Cluster 3:
      Size: 250
      Centroid: [ 0.424      -0.1082755  -0.02603857  0.05830109]
      Mean distance to centroid: 0.7540
      Max distance to centroid: 1.4996

    Cluster 4:
      Size: 180
      Centroid: [ 0.48333333  0.57272696  0.95996241 -0.75731515]
      Mean distance to centroid: 0.8158
      Max distance to centroid: 1.9482

======================================================================
  CLUSTER ANALYSIS - sklearn
======================================================================

  Cluster Distribution:
    Cluster 0:  245 samples (24.50%)
    Cluster 1:  218 samples (21.80%)
    Cluster 2:  126 samples (12.60%)
    Cluster 3:  185 samples (18.50%)
    Cluster 4:  226 samples (22.60%)

  Cluster Characteristics:

    Cluster 0:
      Size: 245
      Centroid: [ 0.4122449  -0.13078208 -0.07347452  0.07015852]
      Mean distance to centroid: 0.7421
      Max distance to centroid: 1.4699

    Cluster 1:
      Size: 218
      Centroid: [-4.99600361e-16 -7.98984323e-01 -8.66037397e-01  7.93494140e-01
]
      Mean distance to centroid: 0.6513
      Max distance to centroid: 1.4174

    Cluster 2:
      Size: 126
      Centroid: [ 0.47619048  2.09562969  1.77023352 -1.88952126]
      Mean distance to centroid: 0.8979
      Max distance to centroid: 1.5838

    Cluster 3:
      Size: 185
      Centroid: [ 0.47567568  0.5246288   0.92790785 -0.68001387]
      Mean distance to centroid: 0.8058
      Max distance to centroid: 1.9890

    Cluster 4:
      Size: 226
      Centroid: [ 1.         -0.68533397 -0.83148215  0.76863577]
      Mean distance to centroid: 0.6425
      Max distance to centroid: 1.9656

[5] Generating Elbow Curve...
  ✓ Elbow Curve saved → elbow_curve.png

[6] Generating Cluster Visualizations...
  ✓ Cluster visualization saved → scratch_clusters.png
  ✓ Cluster visualization saved → sklearn_clusters.png

[7] Generating Inertia History...
  ✓ Inertia history saved → inertia_history.png

[8] Performing Cross Validation...

======================================================================
  5-FOLD CROSS VALIDATION
======================================================================
  Fold 1: Scratch Silhouette = 0.3029, sklearn Silhouette = 0.2878
  Fold 2: Scratch Silhouette = 0.3076, sklearn Silhouette = 0.3095
  Fold 3: Scratch Silhouette = 0.2941, sklearn Silhouette = 0.2958
  Fold 4: Scratch Silhouette = 0.3102, sklearn Silhouette = 0.3102
  Fold 5: Scratch Silhouette = 0.3077, sklearn Silhouette = 0.3077

  Cross-Validation Results:
  Scratch - Mean           : 0.3045 ± 0.0057
  sklearn - Mean           : 0.3022 ± 0.0089
  95% CI Scratch           : [0.2933, 0.3157]
  95% CI sklearn           : [0.2848, 0.3196]

======================================================================
  COMPLEXITY ANALYSIS
======================================================================

  Dataset Characteristics:
  Samples                  : 1,000
  Features                 : 4

  Time Complexity:
  Scratch Implementation   : O(n_init × max_iter × n_samples × n_features)
  sklearn KMeans           : O(n_init × max_iter × n_samples × n_features)
  Note                     : Both use KMeans++ initialization

  Training Time Comparison:
  Scratch                  : 0.1719s
  sklearn                  : 8.1511s
  Speedup Factor           : 0.02x slower

  Space Complexity:
  Scratch                  : O(n_features × k) - stores centroids
  sklearn                  : O(n_features × k) - stores cluster_centers_
  Memory Usage             : Similar for both implementations

  Optimization Techniques:
  Scratch                  : Vectorized distance computation, KMeans++ init
  sklearn                  : Optimized BLAS/LAPACK, Cython acceleration

======================================================================
  METRIC DIFFERENCES ANALYSIS
======================================================================

  Key Differences:
  Metric                    Scratch         sklearn         Difference
  ----------------------------------------------------------------------
  Inertia                   601.3644        600.5564        +0.8080
  Silhouette                0.3063          0.3076          -0.0013
  Davies-Bouldin            1.1079          1.1098          -0.0019
  Calinski-Harabasz         1095.3303       1097.1387       -1.8083
  Time (s)                  0.1719          8.1511          -7.9791

  Possible Reasons for Differences:
  1. Initialization:
     - Both use KMeans++ but random seed differences affect starting points
     - Different random number generators may produce different initializations

  2. Convergence Criteria:
     - Scratch: Checks both label stability AND centroid drift
     - sklearn: Primarily checks centroid drift
     - Different convergence checks lead to different stopping points

  3. Empty Cluster Handling:
     - Scratch: Rescues from highest-error region
     - sklearn: Uses different empty cluster handling strategy
     - Different strategies affect final cluster assignments

  4. Numerical Precision:
     - Scratch: NumPy float64 operations
     - sklearn: Optimized BLAS/LAPACK with potential precision differences
     - Slight numerical differences accumulate over iterations

  5. Multi-restart Selection:
     - Both select best restart based on inertia
     - Different random seeds in restarts lead to different best solutions

  6. Optimization Level:
     - Scratch: Pure Python/NumPy implementation
     - sklearn: Cython-optimized with BLAS acceleration
     - sklearn's lower-level optimizations provide speed advantages

================================================================================
==========
  FINAL SUMMARY
================================================================================
==========

  Scratch Implementation Wins:
  ✓ Davies-Bouldin, Time (s)

  sklearn Wins:
  ✓ Inertia, Silhouette, Calinski-Harabasz

  Overall Assessment:
  🏆 sklearn performs better overall!
================================================================================
==========
