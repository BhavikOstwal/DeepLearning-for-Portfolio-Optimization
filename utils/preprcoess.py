import pandas as pd
import numpy as np

def load_data(file_paths):
    """Load CSV files and extract Close prices, aligning by date."""
    dfs = [pd.read_csv(fp, parse_dates=['Date'], index_col='Date')['Close'] for fp in file_paths]
    data = pd.concat(dfs, axis=1).dropna()
    data.columns = ['VTI', 'AGG', 'DBC', 'VIX']
    return data

def calculate_returns(data):
    """Calculate daily returns from close prices."""
    returns = data.pct_change().dropna()
    return returns

def standardize_data(train_data, test_data):
    """Standardize prices and returns using training data statistics."""
    price_means = train_data.mean()
    price_stds = train_data.std()
    train_prices_std = (train_data - price_means) / price_stds
    test_prices_std = (test_data - price_means) / price_stds
    
    return_means = train_prices_std.pct_change().dropna().mean()
    return_stds = train_prices_std.pct_change().dropna().std()
    train_returns = train_prices_std.pct_change().dropna()
    test_returns = test_prices_std.pct_change().dropna()
    train_returns_std = (train_returns - return_means) / return_stds
    test_returns_std = (test_returns - return_means) / return_stds
    
    return train_prices_std, test_prices_std, train_returns_std, test_returns_std

def create_sequences(prices, returns, lookback=50):
    """Create input sequences and next-day returns for training/testing."""
    sequences = []
    for t in range(lookback, len(prices) - 1):
        price_seq = prices.iloc[t - lookback:t].values  # (50, 4)
        return_seq = returns.iloc[t - lookback:t].values  # (50, 4)
        x = np.hstack([price_seq, return_seq])  # (50, 8)
        r_next = returns.iloc[t].values  # (4,)
        sequences.append((x, r_next))
    return sequences