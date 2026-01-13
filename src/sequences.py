import numpy as np

def create_sequences(data, seq_length):
    """
    Convert time-series data into sequences for model input.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
