import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.preprocessing import StandardScaler
import models

# Paths to save scaler objects
SCALER_X_PATH = "scaler_x.joblib"
SCALER_SUM_PATH = "scaler_sum.joblib"
SCALER_COUNTS_PATH = "scaler_counts.joblib"

class PruningCallback(tf.keras.callbacks.Callback):
    """
    Enforces the pruning masks by setting pruned weights to 0
    at the end of each training batch.
    """
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        for layer_name, mask in self.masks.items():
            layer = self.model.get_layer(layer_name)
            weights = layer.get_weights()
            if weights:
                # weights[0] is kernel
                weights[0] = weights[0] * mask
                layer.set_weights(weights)

def fit_transform_3d(scaler, X):
    """Fits and transforms a 3D input sequence array using a 2D scaler."""
    samples, w, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    return X_scaled_flat.reshape(samples, w, num_features)

def transform_3d(scaler, X):
    """Transforms a 3D input sequence array using a pre-fitted 2D scaler."""
    samples, w, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    X_scaled_flat = scaler.transform(X_flat)
    return X_scaled_flat.reshape(samples, w, num_features)

def train_and_prune(data_dict, window_size=10, lstm_units=64, learning_rate=1e-3, epochs=40, verbose=0):
    """
    Trains the multi-modal LSTM, performs Iterative Magnitude Pruning (IMP), 
    rewinds weights to their initial state, and retrains the model.
    """
    # 1. Scale data
    scaler_x = StandardScaler()
    scaler_sum = StandardScaler()
    scaler_counts = StandardScaler()
    
    # Scale X_num
    X_num_scaled = fit_transform_3d(scaler_x, data_dict['X_num'])
    # Scale y_sum
    y_sum_scaled = scaler_sum.fit_transform(data_dict['y_sum'])
    # Scale y_counts
    y_counts_scaled = scaler_counts.fit_transform(data_dict['y_counts'])
    
    # Save scalers
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_sum, SCALER_SUM_PATH)
    joblib.dump(scaler_counts, SCALER_COUNTS_PATH)
    
    X_main = data_dict['X_main']
    X_euro = data_dict['X_euro']
    y_main_logits = data_dict['y_main_logits']
    y_euro_logits = data_dict['y_euro_logits']
    
    # Inputs list for keras fit
    inputs = [X_num_scaled, X_main, X_euro]
    targets = {
        "sum_head": y_sum_scaled,
        "counts_head": y_counts_scaled,
        "main_logits_head": y_main_logits,
        "euro_logits_head": y_euro_logits
    }
    
    # 2. Build model and capture initial weights layer-by-layer
    model = models.build_multimodal_model(
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
    
    print("Stage 1: Training initial model...")
    # Train stage 1
    model.fit(
        inputs,
        targets,
        epochs=epochs,
        batch_size=32,
        verbose=verbose
    )
    
    # 3. Compute magnitude masks for Dense layers
    prune_layers = ["sum_head", "counts_head", "main_logits_head", "euro_logits_head"]
    masks = {}
    
    for name in prune_layers:
        layer = model.get_layer(name)
        weights = layer.get_weights()
        if weights:
            kernel = weights[0]
            # Prune lowest 80% by absolute magnitude
            threshold = np.percentile(np.abs(kernel), 80)
            mask = (np.abs(kernel) >= threshold).astype(np.float32)
            masks[name] = mask
            
            # Print sparsity details
            total_elements = mask.size
            kept_elements = np.sum(mask)
            print(f"Layer {name} Pruning: Kept {kept_elements}/{total_elements} weights ({kept_elements/total_elements:.1%})")
            
    # 4. Rewind to original initialization and apply mask layer-by-layer
    print("Stage 2: Retraining the sparse pruned subnetwork (Lottery Ticket Hypothesis)...")
    for layer in model.layers:
        if layer.name in initial_layer_weights:
            w_init = [np.copy(x) for x in initial_layer_weights[layer.name]]
            if layer.name in masks:
                # Apply mask to the kernel (w_init[0])
                w_init[0] = w_init[0] * masks[layer.name]
            layer.set_weights(w_init)
            
    # 5. Retrain model with PruningCallback
    pruning_callback = PruningCallback(masks)
    
    model.fit(
        inputs,
        targets,
        epochs=epochs,
        batch_size=32,
        callbacks=[pruning_callback],
        verbose=verbose
    )
    
    # Verify that pruned weights are indeed 0
    for name in prune_layers:
        layer = model.get_layer(name)
        kernel = layer.get_weights()[0]
        sparsity = np.mean(kernel == 0)
        print(f"Post-Retraining {name} Sparsity (zeros fraction): {sparsity:.1%}")
        
    return model
