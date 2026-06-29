import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
import sportka_database
import sportka_features
import sportka_models

SCALER_X_PATH = "sportka_scaler_x.joblib"
SCALER_SUM_PATH = "sportka_scaler_sum.joblib"
SCALER_COUNTS_PATH = "sportka_scaler_counts.joblib"
MODEL_PATH = "sportka_lstm_model.keras"

def fit_transform_3d(scaler, X):
    samples, w, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    return X_scaled_flat.reshape(samples, w, num_features)

def transform_3d(scaler, X):
    samples, w, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    X_scaled_flat = scaler.transform(X_flat)
    return X_scaled_flat.reshape(samples, w, num_features)

def train_and_prune(data_dict, window_size=10, lstm_units=64, learning_rate=1e-3, epochs=25, verbose=1):
    """
    Trains the multi-modal LSTM for Sportka, performs Iterative Magnitude Pruning (IMP),
    and saves the model.
    """
    # 1. Scale data
    scaler_x = StandardScaler()
    scaler_sum = StandardScaler()
    scaler_counts = StandardScaler()
    
    X_num_scaled = fit_transform_3d(scaler_x, data_dict['X_num'])
    y_sum_scaled = scaler_sum.fit_transform(data_dict['y_sum'])
    y_counts_scaled = scaler_counts.fit_transform(data_dict['y_counts'])
    
    # Save scalers
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_sum, SCALER_SUM_PATH)
    joblib.dump(scaler_counts, SCALER_COUNTS_PATH)
    
    X_main = data_dict['X_main']
    y_main_logits = data_dict['y_main_logits']
    
    inputs = [X_num_scaled, X_main]
    targets = {
        "sum_head": y_sum_scaled,
        "counts_head": y_counts_scaled,
        "main_logits_head": y_main_logits
    }
    
    # 2. Build model and capture initial weights
    model = sportka_models.build_sportka_model(
        window_size=window_size,
        num_features=data_dict['X_num'].shape[2],
        lstm_units=lstm_units,
        learning_rate=learning_rate
    )
    
    initial_layer_weights = {}
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            initial_layer_weights[layer.name] = [np.copy(x) for x in w]
            
    print("Stage 1: Training initial Sportka model...")
    model.fit(inputs, targets, epochs=epochs, batch_size=64, verbose=verbose)
    
    # 3. Compute magnitude masks (80% sparsity for prediction heads)
    prune_layers = ["sum_head", "counts_head", "main_logits_head"]
    masks = {}
    
    for name in prune_layers:
        layer = model.get_layer(name)
        weights = layer.get_weights()
        W = weights[0]
        
        # Determine threshold for 80% lowest magnitudes
        flat_abs = np.abs(W.flatten())
        threshold = np.percentile(flat_abs, 80)
        
        # Mask: 1.0 to keep, 0.0 to prune
        mask = (np.abs(W) >= threshold).astype(np.float32)
        masks[name] = mask
        print(f"Layer '{name}' sparsity: {100.0 - (np.sum(mask)/mask.size*100.0):.1f}%")
        
    # 4. Rewind to initial weights
    for name in initial_layer_weights:
        layer = model.get_layer(name)
        layer.set_weights(initial_layer_weights[name])
        
    # Apply initial mask directly to the rewound weights
    for name in prune_layers:
        layer = model.get_layer(name)
        w = layer.get_weights()
        w[0] = w[0] * masks[name]
        layer.set_weights(w)
        
    # Custom training loop callback to enforce mask during training
    class EnforceMaskCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            for name in prune_layers:
                layer = self.model.get_layer(name)
                w = layer.get_weights()
                w[0] = w[0] * masks[name]
                layer.set_weights(w)
                
    print("Stage 2: Retraining pruned Sportka model (enforcing sparsity)...")
    model.fit(
        inputs,
        targets,
        epochs=epochs,
        batch_size=64,
        verbose=verbose,
        callbacks=[EnforceMaskCallback()]
    )
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Sportka model saved to {MODEL_PATH}")
    return model

if __name__ == "__main__":
    sportka_database.init_db()
    draws = sportka_database.get_all_draws()
    df = sportka_features.compute_draw_features(draws)
    data_dict = sportka_features.generate_sequences(df, window_size=10)
    train_and_prune(data_dict)
