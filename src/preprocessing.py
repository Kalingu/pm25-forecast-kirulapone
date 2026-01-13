import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clip_data(df, column, min_val, max_val):
    """Clip extreme values in a column."""
    df[column] = df[column].clip(min_val, max_val)
    return df

def log_transform(df, column):
    """Apply log transformation to reduce skewness."""
    df[column] = np.log1p(df[column])
    return df

def scale_data(df, column):
    """Min-max scale a column between 0 and 1."""
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler
