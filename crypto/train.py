import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib
from sklearn.preprocessing import StandardScaler
import bybit_client
import features

SCALER_PATH = "crypto_scaler.joblib"
MODEL_PATH = "crypto_lstm_model.keras"

def build_lstm_model(window_size=20, num_features=8, lstm_units=32, learning_rate=1e-3):
    """
    Builds the LSTM binary classifier for BTC/USDT price direction.
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

def train_crypto_model(verbose=1):
    # 1. Fetch data
    print("Fetching Bybit historical K-lines...")
    df = bybit_client.fetch_historical_klines(symbol="BTCUSDT", interval="60", limit=1000)
    if df is None:
        raise Exception("Could not fetch data.")
        
    # 2. Compute features
    print("Calculating technical indicators...")
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
    
    X_val = X[-val_split:]
    y_val = y[-val_split:]
    
    # 4. Scale features
    # Since X is 3D (samples, window_size, features), we scale along the feature axis
    scaler = StandardScaler()
    samples, w, num_feats = X_train.shape
    X_train_flat = X_train.reshape(-1, num_feats)
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled_flat.reshape(samples, w, num_feats)
    
    # Save scaler in local folder
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), SCALER_PATH))
    print(f"Scaler saved to {SCALER_PATH}")
    
    # 5. Build and train model
    model = build_lstm_model(window_size=window_size, num_features=num_feats)
    
    print("Training LSTM price predictor...")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        verbose=verbose
    )
    
    # Save model in local folder
    model_save_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model.save(model_save_path)
    print(f"Model saved to {MODEL_PATH}")
    return model, scaler

if __name__ == "__main__":
    train_crypto_model()
