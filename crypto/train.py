import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib
from sklearn.preprocessing import StandardScaler

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
sys.path.append(os.path.join(project_path, "crypto"))

import bybit_client
import features

def build_lstm_model(window_size=20, num_features=8, lstm_units=32, learning_rate=1e-3):
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
    if df is None or len(df) < 50:
        raise Exception(f"Could not fetch sufficient data for {symbol}.")
        
    # 2. Compute features
    print(f"Calculating technical indicators for {symbol}...")
    df_indicators = features.calculate_indicators(df)
    
    # 3. Generate sequences
    window_size = 20
    data_dict = features.generate_sequences(df_indicators, window_size=window_size)
    
    X = data_dict["X"]
    y = data_dict["y"]
    
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
    
    print(f"Training LSTM price predictor for {symbol}...")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=20,
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
