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

def send_email_alert(new_action, price, confidence, sl, tp):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    email_to = os.environ.get("EMAIL_TO", "lubos8huml@gmail.com")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print("SMTP_USER or SMTP_PASSWORD not set. Skipping email alert.")
        return False

    subject = f"[AI Bot] BTC/USDT Trade Alert: {new_action}"
    
    body = f"""
    <h3>BTC/USDT AI Prediction Alert</h3>
    <p>The neural forecasting model has detected a change in trading action:</p>
    <ul>
        <li><b>New Action:</b> <span style="color: {'#06b6d4' if 'LONG' in new_action else '#ef4444' if 'SHORT' in new_action else '#6b7280'}; font-weight: bold;">{new_action}</span></li>
        <li><b>Trigger Price:</b> {price:,.1f} USDT</li>
        <li><b>Model Confidence:</b> {confidence:.1f}%</li>
        <li><b>Stop Loss (SL):</b> {sl}</li>
        <li><b>Take Profit (TP):</b> {tp}</li>
    </ul>
    <br>
    <p><i>This is an automated message from your Eurojackpot & Crypto Predictor dashboard.</i></p>
    """

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = email_to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, email_to, msg.as_string())
        server.close()
        print(f"Email alert successfully sent to {email_to}")
        return True
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return False

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
    
    # Check if prediction changed
    old_action = None
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                old_data = json.load(f)
                old_action = old_data.get("action")
        except Exception:
            pass
            
    # Save first
    with open(output_path, "w") as f:
        json.dump(output, f)
        
    print(f"Generated live prediction saved to {output_path}")
    print(output)
    
    # Send email if action changed
    if old_action is not None and old_action != action:
        print(f"Action changed from {old_action} to {action}. Sending email...")
        send_email_alert(
            action,
            output["price"],
            output["probability"],
            output["stop_loss"],
            output["take_profit"]
        )
        
    return True

if __name__ == "__main__":
    generate_live_prediction()
