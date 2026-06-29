import numpy as np
import features
import models
import train
import backtest
from sklearn.preprocessing import StandardScaler

def autotune_hyperparameters(df, val_split=50, verbose=0):
    """
    Performs grid search tuning over lookback window, LSTM units, and learning rate.
    Uses 15 epochs during grid search to keep search time reasonable.
    """
    print("\n--- Initiating Hyperparameter Autotuning Loop ---")
    
    # Grid search space
    window_sizes = [5, 10, 15]
    lstm_densities = [32, 64]
    learning_rates = [1e-3, 5e-3]
    
    best_score = float('inf')
    best_params = {
        'window_size': 10,
        'lstm_units': 64,
        'learning_rate': 1e-3
    }
    
    # We will split the df into train and val
    # To do this cleanly, we generate sequences on the full df, and then slice the last val_split for validation
    
    for w in window_sizes:
        # Generate sequences with window size w
        data = features.generate_sequences(df, window_size=w)
        
        # Split train/val
        # The last val_split draws are reserved for validation
        n_samples = len(data['X_num'])
        if n_samples <= val_split + 5:
            # Not enough samples to split, skip this window size or reduce split
            continue
            
        train_idx = n_samples - val_split
        
        # Train split
        X_num_train = data['X_num'][:train_idx]
        X_main_train = data['X_main'][:train_idx]
        X_euro_train = data['X_euro'][:train_idx]
        y_sum_train = data['y_sum'][:train_idx]
        y_counts_train = data['y_counts'][:train_idx]
        y_main_logits_train = data['y_main_logits'][:train_idx]
        y_euro_logits_train = data['y_euro_logits'][:train_idx]
        
        # Val split
        X_num_val = data['X_num'][train_idx:]
        X_main_val = data['X_main'][train_idx:]
        X_euro_val = data['X_euro'][train_idx:]
        y_sum_val = data['y_sum'][train_idx:]
        y_counts_val = data['y_counts'][train_idx:]
        y_main_logits_val = data['y_main_logits'][train_idx:]
        y_euro_logits_val = data['y_euro_logits'][train_idx:]
        
        # Fit train scalers
        scaler_x = StandardScaler()
        scaler_sum = StandardScaler()
        scaler_counts = StandardScaler()
        
        X_num_train_scaled = train.fit_transform_3d(scaler_x, X_num_train)
        y_sum_train_scaled = scaler_sum.fit_transform(y_sum_train)
        y_counts_train_scaled = scaler_counts.fit_transform(y_counts_train)
        
        train_inputs = [X_num_train_scaled, X_main_train, X_euro_train]
        train_targets = {
            "sum_head": y_sum_train_scaled,
            "counts_head": y_counts_train_scaled,
            "main_logits_head": y_main_logits_train,
            "euro_logits_head": y_euro_logits_train
        }
        
        for units in lstm_densities:
            for lr in learning_rates:
                print(f"Testing: w={w}, lstm_units={units}, lr={lr}...")
                
                # Build model
                model = models.build_multimodal_model(
                    window_size=w,
                    num_features=X_num_train.shape[2],
                    lstm_units=units,
                    learning_rate=lr
                )
                
                # Train for 15 epochs (fast tuning run)
                model.fit(
                    train_inputs,
                    train_targets,
                    epochs=15,
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate on validation split
                metrics = backtest.evaluate_model(
                    model, X_num_val, X_main_val, X_euro_val,
                    y_sum_val, y_counts_val, y_main_logits_val, y_euro_logits_val,
                    scaler_x, scaler_sum, scaler_counts
                )
                
                # Compute hyperparameter score:
                # Score combines Sum MAE and Top-10 Classification Error (1.0 - main_top10)
                # Lower is better
                sum_err = metrics['sum_mae']
                class_err = 100.0 * (1.0 - metrics['main_top10'])
                score = sum_err + class_err
                
                print(f" -> Val Sum MAE: {sum_err:.2f}, Top-10 Acc: {metrics['main_top10']:.1%}, Score: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'window_size': w,
                        'lstm_units': units,
                        'learning_rate': lr
                    }
                    
    print(f"\nTuning finished. Best params: {best_params} (Score: {best_score:.4f})")
    return best_params
