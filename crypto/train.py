import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib
from sklearn.preprocessing import StandardScaler

# Add project path dynamically to sys.path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_path, "crypto"))

import bybit_client
import features
import quantum_lotto

def build_lstm_model(window_size=20, num_features=32, lstm_units=32, learning_rate=1e-3):
    """
    Builds the LSTM binary classifier for price direction.
    """
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

def train_crypto_model(symbol="BTCUSDT", verbose=0):
    # 1. Fetch data
    print(f"Fetching Bybit historical K-lines for {symbol}...")
    df = bybit_client.fetch_historical_klines(symbol=symbol, interval="60", limit=1000)
    if df is None or len(df) < 200:
        raise Exception(f"Could not fetch sufficient data for {symbol}.")
        
    # Discard the last (current incomplete) candle to match the inference pipeline
    df = df.iloc[:-1].reset_index(drop=True)
        
    # 2. Compute features
    print(f"Calculating technical indicators for {symbol}...")
    df_indicators = features.calculate_indicators(df)
    
    # Scale technical indicators to [-1, 1] for Dual 4-qubit Quantum Reservoirs
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
    
    # Simulate Dual 4-qubit Quantum Reservoirs
    print(f"Simulating Dual 4-qubit Quantum Reservoirs for {symbol}...")
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
    
    # 8 classical features + 12 Res A + 12 Res B = 32 features
    total_feats = np.column_stack([classical_feats, q_feats_a, q_feats_b])
    
    # Save a temporary copy of indicators to keep targets and features aligned
    df_temp = df_indicators.copy()
    
    # 3. Generate sequences
    window_size = 20
    X = []
    y = []
    targets = df_temp["target"].values
    for i in range(len(df_temp) - window_size):
        X.append(total_feats[i : i + window_size])
        y.append(targets[i + window_size - 1])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # Validation split: hold out the last 150 hours for backtesting
    val_split = 150
    X_train = X[:-val_split]
    y_train = y[:-val_split]
    
    # 4. Scale features
    scaler = StandardScaler()
    samples, w, num_feats = X_train.shape
    X_train_flat = X_train.reshape(-1, num_feats)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(samples, w, num_feats)
    
    # Save scaler in local folder with symbol suffix
    sym_lower = symbol.lower().replace("/", "")
    scaler_name = f"crypto_scaler_{sym_lower}.joblib"
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), scaler_name))
    print(f"Scaler saved to {scaler_name}")
    
    # 5. Build and train model
    model = build_lstm_model(window_size=window_size, num_features=num_feats)
    
    print(f"Training LSTM price predictor for {symbol} (30 epochs)...")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=verbose
    )
    
    # Save model in local folder with symbol suffix
    model_name = f"crypto_lstm_model_{sym_lower}.keras"
    model_save_path = os.path.join(os.path.dirname(__file__), model_name)
    model.save(model_save_path)
    print(f"Model saved to {model_name}")
    return model, scaler

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for sym in symbols:
        print(f"\n==================================================")
        print(f"          STARTING TRAINING FOR: {sym}")
        print(f"==================================================")
        try:
            train_crypto_model(symbol=sym, verbose=0)
        except Exception as e:
            print(f"Training failed for {sym}: {e}")
