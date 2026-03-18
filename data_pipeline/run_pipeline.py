import os
import numpy as np
import pandas as pd
from ingestion import load_data
from cleaning import clean_data
from feature_engineering import engineer_features, encode_labels, vectorize_text, encode_gender
from cleaning import scale_specific_features
from split_data import split_train_test

def process_spam_dataset(file_path, output_dir='processed_data'):
    """Process spam dataset for logistic regression."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, file_path)
    print(f"\nProcessing: {file_path}")
    
    # Step 1: Ingestion with proper encoding
    df = load_data(file_path, encoding='latin-1')
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Drop unnecessary columns
    drop_cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    df = df.drop(columns=drop_cols, axis=1, errors='ignore')
    df = df.dropna()
    print(f"  After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 3: Prepare X and Y
    X = df['v2']
    Y = df['v1']
    
    # Step 4: Encode labels
    Y_encoded, label_encoder = encode_labels(Y)
    print(f"  Labels encoded: {len(np.unique(Y_encoded))} classes")
    
    # Step 5: Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_encoded, test_size=0.15, random_state=42
    )
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    # Step 6: Vectorize text using TF-IDF
    max_features = 5000
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(
        X_train, X_test, max_features=max_features
    )
    print(f"  Vectorized: max_features={max_features}")
    print(f"  Features shape - Train: {X_train_vectorized.shape}, Test: {X_test_vectorized.shape}")
    
    # Step 7: Save processed data
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.basename(file_path).replace('.csv', '')
    
    from scipy.sparse import save_npz
    save_npz(os.path.join(output_path, f"{file_name}_X_train.npz"), X_train_vectorized)
    save_npz(os.path.join(output_path, f"{file_name}_X_test.npz"), X_test_vectorized)
    np.save(os.path.join(output_path, f"{file_name}_y_train.npy"), Y_train)
    np.save(os.path.join(output_path, f"{file_name}_y_test.npy"), Y_test)
    print(f"  Saved to {output_path}/")

def process_dataset(file_path, output_dir='processed_data', exclude_cols=None, target_col=None, drop_cols=None):
    """Process a single dataset through the pipeline."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, file_path)
    print(f"\nProcessing: {file_path}")
    
    # Step 1: Ingestion
    df = load_data(file_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Cleaning
    df, scaler = clean_data(df, exclude_cols=exclude_cols, missing_strategy='drop', drop_cols=drop_cols)
    print(f"  After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 3: Feature Engineering
    df = engineer_features(df)
    print(f"  After feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 4: Train/Test Split
    if target_col and target_col in df.columns:
        X_train, X_test, y_train, y_test = split_train_test(df, target_col=target_col)
        print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
        
        output_path = os.path.join(script_dir, output_dir)
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.basename(file_path).replace('.csv', '')
        X_train.to_csv(os.path.join(output_path, f"{file_name}_X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_path, f"{file_name}_X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_path, f"{file_name}_y_train.csv"), index=False)
        y_test.to_csv(os.path.join(output_path, f"{file_name}_y_test.csv"), index=False)
        print(f"  Saved to {output_path}/")
    else:
        train_df, test_df = split_train_test(df)
        print(f"  Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")
        
        output_path = os.path.join(script_dir, output_dir)
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.basename(file_path).replace('.csv', '')
        train_df.to_csv(os.path.join(output_path, f"{file_name}_train.csv"), index=False)
        test_df.to_csv(os.path.join(output_path, f"{file_name}_test.csv"), index=False)
        print(f"  Saved to {output_path}/")

def process_store_customers_dataset(file_path, output_dir='processed_data'):
    """Process store_customers dataset for K-means clustering."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, file_path)
    print(f"\nProcessing: {file_path}")
    
    # Step 1: Ingestion
    df = load_data(file_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Make a copy
    df_processed = df.copy()
    
    # Step 3: Check for duplicates
    duplicates = df_processed.duplicated().sum()
    print(f"  Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df_processed = df_processed.drop_duplicates()
        print(f"  Removed {duplicates} duplicate rows")
    
    # Step 4: Handle missing values
    if df_processed.isnull().sum().sum() > 0:
        from cleaning import handle_missing_values
        df_processed = handle_missing_values(df_processed, strategy='median_mode')
        print(f"  Handled missing values")
    else:
        print(f"  No missing values found")
    
    # Step 5: Encode Gender (0=Female, 1=Male)
    df_processed, label_encoder = encode_gender(df_processed)
    print(f"  Encoded Gender column")
    
    # Step 6: Drop CustomerID
    df_processed = df_processed.drop('CustomerID', axis=1)
    print(f"  Dropped CustomerID column")
    print(f"  After preprocessing: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
    
    # Step 7: Feature scaling (only specific features)
    features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    df_scaled, scaler = scale_specific_features(df_processed, features_to_scale)
    print(f"  Scaled features: {features_to_scale}")
    
    # Step 8: Save processed data
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.basename(file_path).replace('.csv', '')
    df_scaled.to_csv(os.path.join(output_path, f"{file_name}_processed.csv"), index=False)
    print(f"  Saved to {output_path}/")

def process_creditcard_dataset(file_path, output_dir='processed_data'):
    """Process creditcard.csv dataset for Isolation Forest with best practices."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, file_path)
    print(f"\nProcessing: {file_path}")
    
    # Step 1: Load data
    df = pd.read_csv(file_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Check for 'id' column (drop if exists)
    if 'id' in df.columns:
        df = df.drop(columns=['id'], axis=1)
        print(f"  Dropped 'id' column")
    
    # Step 3: Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate rows")
    else:
        print(f"  No duplicates found")
    
    # Step 4: Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values, dropping rows...")
        df = df.dropna()
        print(f"  After dropping missing: {df.shape[0]} rows")
    else:
        print(f"  No missing values found")
    
    # Step 5: Separate features and target
    target_col = 'Class'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")
    
    # Define predictors: Time, V1-V28, Amount
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Feature names: {feature_cols[:5]}... (showing first 5)")
    
    # Step 6: Check Class column encoding (should already be 0/1)
    print(f"\n  Class Distribution:")
    class_counts = y.value_counts().to_dict()
    print(f"    {class_counts}")
    unique_classes = sorted(y.unique())
    print(f"    Unique values: {unique_classes}")
    
    # Verify Class is binary (0/1)
    if set(unique_classes) == {-1, 1}:
        print(f"  Converting Class: -1 -> 0 (normal), 1 -> 1 (fraud)")
        y = y.replace({-1: 0, 1: 1})
    elif set(unique_classes) == {0, 1}:
        print(f"  Class already in correct format: 0 (normal), 1 (fraud)")
    else:
        print(f"  Warning: Unexpected Class values: {unique_classes}")
    
    fraud_count = int(y.sum())
    normal_count = int(len(y) - fraud_count)
    fraud_rate = fraud_count / len(y)
    print(f"  Fraud: {fraud_count} ({fraud_rate:.4%})")
    print(f"  Normal: {normal_count} ({1-fraud_rate:.4%})")
    
    # Step 7: Normalize features using Z-score (mean=0, std=1)
    # Important for Isolation Forest to work well with different feature scales
    print(f"\n  Normalizing features using Z-score...")
    X_normalized = (X - X.mean()) / (X.std() + 1e-8)  # Small epsilon for numerical stability
    
    # Verify normalization
    print(f"  Normalized feature statistics:")
    print(f"    Mean (should be ~0): {X_normalized.mean().mean():.6f}")
    print(f"    Std (should be ~1): {X_normalized.std().mean():.6f}")
    
    # Step 8: Convert to numpy arrays
    X_array = X_normalized.values.astype(np.float64)
    y_array = y.values.astype(np.int32)
    
    # Step 9: Train/Test Split (80/20) with stratification
    # Stratification is crucial for imbalanced datasets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y_array  # Maintain class distribution in both sets
    )
    
    print(f"\n  Dataset Split (80/20):")
    train_fraud = int(np.sum(y_train))
    train_normal = int(np.sum(y_train == 0))
    test_fraud = int(np.sum(y_test))
    test_normal = int(np.sum(y_test == 0))
    
    print(f"    Train: {X_train.shape[0]:,} samples")
    print(f"            - Normal: {train_normal:,} ({100*train_normal/len(y_train):.2f}%)")
    print(f"            - Fraud:  {train_fraud:,} ({100*train_fraud/len(y_train):.4f}%)")
    print(f"    Test:  {X_test.shape[0]:,} samples")
    print(f"            - Normal: {test_normal:,} ({100*test_normal/len(y_test):.2f}%)")
    print(f"            - Fraud:  {test_fraud:,} ({100*test_fraud/len(y_test):.4f}%)")
    
    # Step 10: Create DataFrames for saving
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df['Class'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['Class'] = y_test
    
    # Step 11: Save processed data
    output_path = os.path.join(script_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.basename(file_path).replace('.csv', '')
    
    train_file = os.path.join(output_path, f"{file_name}_train.csv")
    test_file = os.path.join(output_path, f"{file_name}_test.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n  ✓ Saved processed data:")
    print(f"    Train: {train_file}")
    print(f"    Test:  {test_file}")
    print(f"  ✓ Preprocessing completed successfully!")

if __name__ == "__main__":
    # Process creditcard dataset (new file: creditcard.csv)
    process_creditcard_dataset("datasets/creditcard.csv")
    
    # Process spam dataset for logistic regression
    process_spam_dataset("datasets/spam.csv")
    
    # Process store_customers dataset for K-means
    process_store_customers_dataset("datasets/store_customers.csv")
    
    print("\nPipeline completed successfully!")
