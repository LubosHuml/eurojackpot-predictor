import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(script_dir)
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import crypto.features as features
import crypto.quantum_lotto as quantum_lotto

def build_lstm_model(window_size=20, num_features=32, lstm_units=32):
    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(
            lstm_units,
            dropout=0.2,
            kernel_regularizer=regularizers.l2(1e-4),
            recurrent_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def fetch_mt5_data(symbol, limit=3000):
    print(f"Fetching {limit} hourly bars for {symbol} from MT5...")
    # Select the symbol in Market Watch
    mt5.symbol_select(symbol, True)
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, limit)
    if rates is None or len(rates) == 0:
        print(f"[ERROR] Could not fetch rates for {symbol}. Error: {mt5.last_error()}")
        return None
        
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    })
    return df

def train_mt5_model(symbol):
    df = fetch_mt5_data(symbol, limit=3000)
    if df is None or len(df) < 500:
        print(f"[ERROR] Insufficient data for {symbol}")
        return
        
    # Calculate indicators
    df_indicators = features.calculate_indicators(df)
    
    # Scale indicators for Quantum Reservoir
    close_sma10 = df_indicators["close_to_sma10"].values
    close_ema10 = df_indicators["close_to_ema10"].values
    sma10_sma30 = df_indicators["sma10_to_sma30"].values
    bb_pos = df_indicators["bb_position"].values
    rsi_vals = df_indicators["rsi"].values
    log_ret = df_indicators["log_return"].values
    vol = df_indicators["volatility"].values
    vol_chg = df_indicators["volume_change"].values
    
    u_res_a = np.column_stack([
        np.clip(close_sma10 * 20.0, -1.0, 1.0),
        np.clip(close_ema10 * 20.0, -1.0, 1.0),
        np.clip(sma10_sma30 * 20.0, -1.0, 1.0),
        np.clip(2.0 * bb_pos - 1.0, -1.0, 1.0)
    ])
    
    u_res_b = np.column_stack([
        np.clip(2.0 * rsi_vals - 1.0, -1.0, 1.0),
        np.clip(log_ret * 50.0, -1.0, 1.0),
        np.clip(vol * 200.0 - 1.0, -1.0, 1.0),
        np.clip(np.tanh(vol_chg), -1.0, 1.0)
    ])
    
    # Simulate Dual Reservoirs
    res_a = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    res_b = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    
    q_feats_a = []
    q_feats_b = []
    for idx in range(len(df_indicators)):
        q_feats_a.append(res_a.step(u_res_a[idx]))
        q_feats_b.append(res_b.step(u_res_b[idx]))
        
    q_feats_a = np.array(q_feats_a)
    q_feats_b = np.array(q_feats_b)
    
    classical_feats = df_indicators[[
        "close_to_sma10", "close_to_ema10", "sma10_to_sma30",
        "bb_position", "rsi", "log_return", "volatility", "volume_change"
    ]].values
    
    total_feats = np.column_stack([classical_feats, q_feats_a, q_feats_b])
    
    window_size = 20
    X = []
    y = []
    targets = df_indicators["target"].values
    for i in range(len(df_indicators) - window_size):
        X.append(total_feats[i : i + window_size])
        y.append(targets[i + window_size - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # Scale features
    scaler = StandardScaler()
    samples, w, num_feats = X.shape
    X_flat = X.reshape(-1, num_feats)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(samples, w, num_feats)
    
    # Save scaler and model
    sym_lower = symbol.lower()
    crypto_dir = os.path.join(project_path, "crypto")
    
    scaler_path = os.path.join(crypto_dir, f"crypto_scaler_{sym_lower}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    model = build_lstm_model(window_size=window_size, num_features=num_feats)
    print(f"Training LSTM model for {symbol} (30 epochs)...")
    model.fit(
        X_scaled,
        y,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    model_path = os.path.join(crypto_dir, f"crypto_lstm_model_{sym_lower}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
        
    symbols = ["USDJPY", "NAS100", "XAUUSD"]
    for s in symbols:
        print(f"\n==================================================")
        print(f"        STARTING MT5 TRAINING FOR: {s}")
        print(f"==================================================")
        try:
            train_mt5_model(s)
        except Exception as e:
            print(f"Training failed for {s}: {e}")
            
    mt5.shutdown()
    print("\nAll MT5 asset models trained and ready!")

if __name__ == "__main__":
    main()
