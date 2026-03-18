import pandas as pd
import os

def load_data(file_path, encoding='utf-8', delimiter=','):
    """Load CSV data from file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
