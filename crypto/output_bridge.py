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

def get_prediction_for_symbol(symbol):
    crypto_dir = os.path.dirname(__file__)
    sym_lower = symbol.lower().replace("/", "")
    
    model_path = os.path.join(crypto_dir, f"crypto_lstm_model_{sym_lower}.keras")
    scaler_path = os.path.join(crypto_dir, f"crypto_scaler_{sym_lower}.joblib")
    backtest_path = os.path.join(crypto_dir, "crypto_backtest_results.json")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler not found for {symbol}.")
        return None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load backtest win rate if available
    win_rate = 50.0
    if os.path.exists(backtest_path):
        try:
            with open(backtest_path, "r") as f:
                bt_data = json.load(f)
                win_rate = bt_data.get(sym_lower, {}).get("win_rate", 50.0)
        except Exception:
            pass
            
    # Fetch data
    print(f"Fetching live data for {symbol}...")
    df = bybit_client.fetch_historical_klines(symbol=symbol, interval="60", limit=60)
    if df is None or len(df) < 40:
        print(f"Failed to fetch sufficient data for {symbol}.")
        return None
        
    df_indicators = features.calculate_indicators(df)
    window_size = 20
    data_dict = features.generate_sequences(df_indicators, window_size=window_size)
    
    X_last = data_dict["X"][-1:] # (1, 20, 8)
    last_close = data_dict["closes"][-1]
    last_date = data_dict["datetimes"][-1]
    
    # Calculate ATR
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    
    # Scale
    samples, w, num_feats = X_last.shape
    X_last_flat = X_last.reshape(-1, num_feats)
    X_last_scaled_flat = scaler.transform(X_last_flat)
    X_last_scaled = X_last_scaled_flat.reshape(samples, w, num_feats)
    
    # Predict
    prob = float(model.predict(X_last_scaled, verbose=0)[0, 0])
    
    prediction = "UP" if prob >= 0.50 else "DOWN"
    confidence = prob if prob >= 0.50 else (1.0 - prob)
    confidence_pct = confidence * 100
    
    # Action and levels
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
            
    return {
        "datetime": str(last_date),
        "price": float(last_close),
        "prediction": prediction,
        "probability": round(confidence_pct, 1),
        "raw_prob": prob,
        "win_rate": round(win_rate, 1),
        "action": action,
        "stop_loss": round(stop_loss, 2) if stop_loss > 0 else "N/A",
        "take_profit": round(take_profit, 2) if take_profit > 0 else "N/A"
    }

def generate_live_predictions():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    output = {}
    
    for sym in symbols:
        pred = get_prediction_for_symbol(sym)
        if pred is not None:
            output[sym.lower()] = pred
            
    if not output:
        print("No predictions successfully generated.")
        return False
        
    output["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    crypto_dir = os.path.dirname(__file__)
    output_path = os.path.join(crypto_dir, "crypto_live_prediction.json")
    
    # Check for changes to trigger alerts
    old_actions = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                old_data = json.load(f)
                for key in old_data.keys():
                    if key != "updated_at":
                        old_actions[key] = old_data[key].get("action")
        except Exception:
            pass
            
    # Save output
    with open(output_path, "w") as f:
        json.dump(output, f)
        
    print(f"Generated multi-token live predictions saved to {output_path}")
    
    # Check for shifts and send alerts
    for key in output.keys():
        if key == "updated_at":
            continue
            
        sym_name = key.upper()
        old_act = old_actions.get(key)
        new_act = output[key].get("action")
        
        if old_act is not None and old_act != new_act:
            print(f"[{sym_name}] Action changed from {old_act} to {new_act}. Sending email...")
            # We trigger the alert email via a helper similar to output_bridge.py
            # Since the alerts are built in executor or output_bridge, we can write a helper here:
            from executor import send_alert
            message = (
                f"=== {sym_name} TREND ALERT ===\n\n"
                f"Action Shift: {old_act} -> {new_act}\n"
                f"Current Price: {output[key]['price']:,.2f} USDT\n"
                f"Confidence: {output[key]['probability']:.1f}%\n"
                f"Stop Loss: {output[key]['stop_loss']}\n"
                f"Take Profit: {output[key]['take_profit']}"
            )
            send_alert(f"[AI Bot] {sym_name} Trend Alert: {new_act}", message)
            
    return True

if __name__ == "__main__":
    generate_live_predictions()
