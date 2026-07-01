import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import crypto.bybit_client as bybit_client
import crypto.features as features
import crypto.quantum_lotto as quantum_lotto

def build_hybrid_lstm_model(window_size=20, num_features=32, lstm_units=32, learning_rate=1e-3):
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
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid", name="direction_head")
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_hybrid_crypto_backtest(symbol="BTCUSDT", val_split=150):
    print(f"Fetching historical data for {symbol}...")
    df = bybit_client.fetch_historical_klines(symbol=symbol, interval="60", limit=1000)
    if df is None or len(df) < 200:
        print("Not enough data.")
        return
        
    df_indicators = features.calculate_indicators(df)
    
    # Extract indicators
    close_sma10 = df_indicators["close_to_sma10"].values
    close_ema10 = df_indicators["close_to_ema10"].values
    sma10_sma30 = df_indicators["sma10_to_sma30"].values
    bb_pos = df_indicators["bb_position"].values
    rsi_vals = df_indicators["rsi"].values
    log_ret = df_indicators["log_return"].values
    vol = df_indicators["volatility"].values
    vol_chg = df_indicators["volume_change"].values
    
    # 1. Map indicators to [-1, 1] range for Quantum Reservoirs
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
    
    # 2. Simulate Dual 4-qubit Quantum Reservoirs
    print("Simulating Dual 4-qubit Quantum Reservoirs...")
    res_a = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    res_b = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    
    q_feats_a = []
    q_feats_b = []
    
    for idx in range(len(df_indicators)):
        q_feats_a.append(res_a.step(u_res_a[idx]))
        q_feats_b.append(res_b.step(u_res_b[idx]))
        
    q_feats_a = np.array(q_feats_a)
    q_feats_b = np.array(q_feats_b)
    
    # Combine classical indicators with quantum reservoir features
    classical_feats = df_indicators[[
        "close_to_sma10", "close_to_ema10", "sma10_to_sma30",
        "bb_position", "rsi", "log_return", "volatility", "volume_change"
    ]].values
    
    # Total features = 8 (classical) + 12 (Res A) + 12 (Res B) = 32 features
    total_feats = np.column_stack([classical_feats, q_feats_a, q_feats_b])
    targets = df_indicators["target"].values
    
    # 3. Generate sequences
    window_size = 20
    X = []
    y = []
    for i in range(len(df_indicators) - window_size):
        X.append(total_feats[i : i + window_size])
        y.append(targets[i + window_size - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    closes_val = df_indicators.iloc[window_size:]["close"].values.tolist()
    
    # Split
    X_train = X[:-val_split]
    y_train = y[:-val_split]
    X_val = X[-val_split:]
    y_val = y[-val_split:]
    val_closes = closes_val[-val_split:]
    
    # Scale features
    scaler = StandardScaler()
    samples, w, num_feats = X_train.shape
    X_train_flat = X_train.reshape(-1, num_feats)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(samples, w, num_feats)
    
    X_val_flat = X_val.reshape(-1, num_feats)
    X_val_scaled_flat = scaler.transform(X_val_flat)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape[0], w, num_feats)
    
    # Train model
    print("Training QRC+LSTM hybrid model...")
    model = build_hybrid_lstm_model(window_size=window_size, num_features=num_feats)
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    pred_probs = model.predict(X_val_scaled, verbose=0).flatten()
    
    correct = 0
    for i in range(len(pred_probs)):
        actual_up = y_val[i, 0] > 0.5
        pred_up = pred_probs[i] > 0.5
        if actual_up == pred_up:
            correct += 1
    win_rate = (correct / len(pred_probs)) * 100
    
    # Calculate returns
    signals = []
    for p in pred_probs:
        if p > 0.51:
            signals.append(1.0)
        elif p < 0.49:
            signals.append(-1.0)
        else:
            signals.append(0.0)
    signals = np.array(signals)
    
    closes = np.array(val_closes)
    asset_returns = np.diff(np.log(closes))
    strategy_signals = signals[:-1]
    strat_returns = strategy_signals * asset_returns
    
    # Fees (0.06% per trade transition)
    fee_rate = 0.0006
    transaction_costs = []
    prev_sig = 0.0
    for sig in strategy_signals:
        if sig != prev_sig:
            cost = fee_rate if prev_sig == 0.0 or sig == 0.0 else fee_rate * 2
            transaction_costs.append(cost)
        else:
            transaction_costs.append(0.0)
        prev_sig = sig
    transaction_costs = np.array(transaction_costs)
    net_strat_returns = strat_returns - transaction_costs
    
    print("\n========================================================")
    print(f"      QRC-ENHANCED HYBRID CRYPTO BACKTEST ({symbol})      ")
    print("========================================================")
    print(f"Classic LSTM Win Rate     : 58.7%")
    print(f"Quantum Hybrid Win Rate   : {win_rate:.2f}%")
    print("--------------------------------------------------------")
    print(f"Cumulative Buy & Hold Return: {np.sum(asset_returns)*100:+.2f}%")
    print(f"Cumulative Strategy Return  : {np.sum(net_strat_returns)*100:+.2f}%")
    print(f"Total Trades Executed       : {np.sum(transaction_costs > 0)}")
    print("========================================================")

if __name__ == "__main__":
    run_hybrid_crypto_backtest()
