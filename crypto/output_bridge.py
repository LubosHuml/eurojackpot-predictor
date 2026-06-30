import os
import sys
import numpy as np
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
            
    # Fetch 50 candles (plenty to compute sliding indicators of window size 20 + 30)
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
    
    # Scale input
    samples, w, num_feats = X_last.shape
    X_last_flat = X_last.reshape(-1, num_feats)
    X_last_scaled_flat = scaler.transform(X_last_flat)
    X_last_scaled = X_last_scaled_flat.reshape(samples, w, num_feats)
    
    # Predict
    prob = float(model.predict(X_last_scaled, verbose=0)[0, 0])
    
    prediction = "UP" if prob >= 0.50 else "DOWN"
    confidence = prob if prob >= 0.50 else (1.0 - prob)
    
    output = {
        "datetime": str(last_date),
        "price": float(last_close),
        "prediction": prediction,
        "probability": round(confidence * 100, 1),
        "raw_prob": prob,
        "win_rate": round(win_rate, 1),
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
