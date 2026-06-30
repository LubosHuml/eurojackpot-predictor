import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import time
from datetime import datetime

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
sys.path.append(os.path.join(project_path, "crypto"))

import bybit_client
import features
import train

def generate_live_prediction():
    crypto_dir = os.path.dirname(__file__)
    model_path = os.path.join(crypto_dir, train.MODEL_PATH)
    scaler_path = os.path.join(crypto_dir, train.SCALER_PATH)
    backtest_path = os.path.join(crypto_dir, "crypto_backtest_results.json")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Train the model first.")
        return False
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load backtest win rate if available
    win_rate = 50.0
    if os.path.exists(backtest_path):
        try:
            with open(backtest_path, "r") as f:
                bt_data = json.load(f)
                win_rate = bt_data.get("win_rate", 50.0)
        except Exception:
            pass
            
    # Fetch 60 candles (plenty to compute sliding indicators of window size 20 + 30)
    print("Fetching live data from Bybit...")
    df = bybit_client.fetch_historical_klines(symbol="BTCUSDT", interval="60", limit=60)
    if df is None or len(df) < 40:
        print("Failed to fetch sufficient data.")
        return False
        
    df_indicators = features.calculate_indicators(df)
    
    window_size = 20
    data_dict = features.generate_sequences(df_indicators, window_size=window_size)
    
    # Get the very last sequence
    X_last = data_dict["X"][-1:] # shape (1, 20, 8)
    last_close = data_dict["closes"][-1]
    last_date = data_dict["datetimes"][-1]
    
    # Calculate ATR (Average True Range) for Stop Loss and Take Profit
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    
    # Scale input
    samples, w, num_feats = X_last.shape
    X_last_flat = X_last.reshape(-1, num_feats)
    X_last_scaled_flat = scaler.transform(X_last_flat)
    X_last_scaled = X_last_scaled_flat.reshape(samples, w, num_feats)
    
    # Predict
    prob = float(model.predict(X_last_scaled, verbose=0)[0, 0])
    
    prediction = "UP" if prob >= 0.50 else "DOWN"
    confidence = prob if prob >= 0.50 else (1.0 - prob)
    confidence_pct = confidence * 100
    
    # Determine actionable trading advice
    # Confidence threshold of 54.5% to filter out noisy trades
    if confidence_pct < 54.5:
        action = "WAIT / NEUTRAL"
        stop_loss = 0.0
        take_profit = 0.0
    else:
        action = "BUY / LONG" if prediction == "UP" else "SELL / SHORT"
        if prediction == "UP":
            stop_loss = last_close - (1.5 * atr)
            take_profit = last_close + (2.0 * atr)
        else:
            stop_loss = last_close + (1.5 * atr)
            take_profit = last_close - (2.0 * atr)
            
    output = {
        "datetime": str(last_date),
        "price": float(last_close),
        "prediction": prediction,
        "probability": round(confidence_pct, 1),
        "raw_prob": prob,
        "win_rate": round(win_rate, 1),
        "action": action,
        "stop_loss": round(stop_loss, 1) if stop_loss > 0 else "N/A",
        "take_profit": round(take_profit, 1) if take_profit > 0 else "N/A",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_path = os.path.join(crypto_dir, "crypto_live_prediction.json")
    with open(output_path, "w") as f:
        json.dump(output, f)
        
    print(f"Generated live prediction saved to {output_path}")
    print(output)
    return True

if __name__ == "__main__":
    generate_live_prediction()
