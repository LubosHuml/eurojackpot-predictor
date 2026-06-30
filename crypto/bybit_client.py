import requests
import pandas as pd
import time

def fetch_historical_klines(symbol="BTCUSDT", interval="60", limit=1000):
    """
    Fetches historical candlestick (K-line) data from Bybit V5 API.
    
    Parameters:
        symbol (str): Trading pair, e.g. "BTCUSDT"
        interval (str): K-line interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, "D", "W", "M"
        limit (int): Number of K-lines to fetch (max 1000)
        
    Returns:
        pd.DataFrame: Candlestick data sorted chronologically (ascending time)
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") != 0:
            raise Exception(f"Bybit API error: {data.get('retMsg')}")
            
        list_data = data["result"]["list"]
        
        # Columns in Bybit V5 K-line list:
        # [0] startTime, [1] openPrice, [2] highPrice, [3] lowPrice, [4] closePrice, [5] volume, [6] turnover
        df = pd.DataFrame(list_data, columns=["startTime", "open", "high", "low", "close", "volume", "turnover"])
        
        # Convert types
        df["startTime"] = pd.to_numeric(df["startTime"])
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["turnover"] = pd.to_numeric(df["turnover"])
        
        # Convert timestamp to datetime (startTime is in milliseconds)
        df["datetime"] = pd.to_datetime(df["startTime"], unit="ms")
        
        # Sort chronologically (ascending time)
        df = df.sort_values("startTime").reset_index(drop=True)
        return df
        
    except Exception as e:
        print(f"Failed to fetch K-lines from Bybit: {e}")
        return None

if __name__ == "__main__":
    # Quick test execution
    print("Testing Bybit API fetch...")
    df = fetch_historical_klines()
    if df is not None:
        print(f"Successfully fetched {len(df)} K-lines.")
        print(df.tail(3))
