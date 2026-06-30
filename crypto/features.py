import numpy as np
import pandas as pd

def calculate_rsi(prices, period=14):
    """
    Relative Strength Index (RSI) indicator.
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / (down + 1e-10)
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / (down + 1e-10)
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def calculate_indicators(df):
    """
    Computes stationary financial indicators from candlestick data.
    """
    df = df.copy()
    
    # 1. Price SMAs and EMA
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    
    # 2. Stationary Price features (normalized relative to current price/averages)
    df["close_to_sma10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
    df["close_to_ema10"] = (df["close"] - df["ema_10"]) / df["ema_10"]
    df["sma10_to_sma30"] = (df["sma_10"] - df["sma_30"]) / df["sma_30"]
    
    # 3. Bollinger Bands & position
    rolling_std = df["close"].rolling(20).std()
    rolling_mean = df["close"].rolling(20).mean()
    df["bb_upper"] = rolling_mean + 2 * rolling_std
    df["bb_lower"] = rolling_mean - 2 * rolling_std
    # Normalized position between Bollinger bands (0 to 1)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    
    # 4. RSI (Relative Strength Index)
    df["rsi"] = calculate_rsi(df["close"].values, period=14) / 100.0 # scale to 0-1
    
    # 5. Returns and Volatility
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(10).std()
    
    # 6. Volume relative change
    df["volume_change"] = df["volume"].pct_change()
    
    # Target label: 1.0 if the next close price is higher than current close, 0.0 otherwise
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(np.float32)
    
    # Drop rows with NaN (from rolling features)
    df = df.dropna().reset_index(drop=True)
    return df

def generate_sequences(df, window_size=20):
    """
    Generates sequence features for LSTM training.
    """
    feature_cols = [
        "close_to_sma10", "close_to_ema10", "sma10_to_sma30",
        "bb_position", "rsi", "log_return", "volatility", "volume_change"
    ]
    
    features_matrix = df[feature_cols].values
    targets = df["target"].values
    
    X = []
    y = []
    
    for i in range(len(df) - window_size):
        X.append(features_matrix[i : i + window_size])
        # Target for index `i + window_size - 1` is whether candle `i + window_size` closed UP
        y.append(targets[i + window_size - 1])
        
    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y, dtype=np.float32).reshape(-1, 1),
        "datetimes": df.iloc[window_size:]["datetime"].values.tolist(),
        "closes": df.iloc[window_size:]["close"].values.tolist()
    }
