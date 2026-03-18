import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()

def handle_missing_values(df, strategy='drop'):
    """Handle missing values."""
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'median_mode':
        df = df.copy()
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].mode().shape[0] > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def normalize_features(df, exclude_cols=None):
    """Normalize numeric features using StandardScaler."""
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

def scale_specific_features(df, features_to_scale):
    """Scale specific features using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df_scaled, scaler

def drop_columns(df, columns):
    """Drop specified columns."""
    return df.drop(columns=columns, axis=1, errors='ignore')

def clean_data(df, exclude_cols=None, missing_strategy='drop', drop_cols=None):
    """Apply all cleaning steps."""
    if drop_cols:
        df = drop_columns(df, drop_cols)
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    df, scaler = normalize_features(df, exclude_cols=exclude_cols)
    return df, scaler
