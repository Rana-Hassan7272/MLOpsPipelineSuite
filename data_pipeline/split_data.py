import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df, target_col=None, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df, test_df
