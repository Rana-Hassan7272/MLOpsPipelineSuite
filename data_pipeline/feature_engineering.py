import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def encode_categorical(df, columns=None):
    """Encode categorical variables."""
    if columns is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    else:
        categorical_cols = columns
    
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    return df_encoded

def encode_labels(y):
    """Encode labels using LabelEncoder."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def encode_gender(df):
    """Encode Gender column using LabelEncoder (0=Female, 1=Male)."""
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])
    return df_encoded, le

def vectorize_text(X_train, X_test, max_features=5000):
    """Vectorize text data using TF-IDF for logistic regression."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer

def engineer_features(df):
    """Apply feature engineering steps."""
    df = encode_categorical(df)
    return df
