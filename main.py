import argparse
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import database
import scraper
import features
import models
import train
import backtest
import generator
import tuner

def get_last_window_inputs(df, window_size):
    """
    Extracts the very last sequence window of size w to predict the next draw.
    """
    feature_cols = [
        'mean', 'std', 'median', 'sum', 'product_diff',
        'even_count', 'odd_count', 'low_count', 'high_count',
        'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
    ]
    
    num_features = df[feature_cols].values
    main_nums = df[['num1', 'num2', 'num3', 'num4', 'num5']].values
    euro_nums = df[['euro1', 'euro2']].values
    
    X_num = num_features[-window_size:]
    X_main = main_nums[-window_size:]
    X_euro = euro_nums[-window_size:]
    
    # Add batch dimension: [1, w, features]
    return {
        'X_num': np.expand_dims(X_num, axis=0).astype(np.float32),
        'X_main': np.expand_dims(X_main, axis=0).astype(np.int32),
        'X_euro': np.expand_dims(X_euro, axis=0).astype(np.int32)
    }

def main():
    parser = argparse.ArgumentParser(description="Eurojackpot AI Prediction & Validation Engine")
    parser.add_argument("--force-scrape", action="store_true", help="Force scraping of all historical draw data from 2012")
    parser.add_argument("--force-tune", action="store_true", help="Force hyperparameter autotuning loop before training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs per IMP stage")
    parser.add_argument("--bets-count", type=int, default=5, help="Number of combinations to generate per temperature setting")
    args = parser.parse_args()
    
    print("==========================================================")
    print("   EUROJACKPOT AI FORECASTING & VALIDATION ENGINE")
    print("==========================================================")
    
    # 1. Database and Scraping
    print("\n[Step 1/6] Synchronizing Database and Scraping Draw History...")
    scraper.update_database(force_all=args.force_scrape)
    
    draws = database.get_all_draws()
    print(f"Total draws currently in database: {len(draws)}")
    if len(draws) < 50:
        print("Error: Database has insufficient draws for model training. Need at least 50 draws.")
        return
        
    # 2. Feature Engineering
    print("\n[Step 2/6] Performing Advanced Feature Engineering...")
    df_features = features.compute_draw_features(draws)
    print(f"Features computed. Shape of feature matrix: {df_features.shape}")
    
    # 3. Hyperparameter Tuning
    window_size = 10
    lstm_units = 64
    learning_rate = 1e-3
    
    if args.force_tune:
        print("\n[Step 3/6] Running Autotuning Loop...")
        best_params = tuner.autotune_hyperparameters(df_features, val_split=50)
        window_size = best_params['window_size']
        lstm_units = best_params['lstm_units']
        learning_rate = best_params['learning_rate']
    else:
        print("\n[Step 3/6] Using default hyperparameters (lookback w=10, lstm=64, lr=0.001)")
        
    # 4. Generate sequences and train model with IMP
    print(f"\n[Step 4/6] Preparing Sequences and Training Multi-Modal LSTM (IMP)...")
    data_dict = features.generate_sequences(df_features, window_size=window_size)
    
    model = train.train_and_prune(
        data_dict=data_dict,
        window_size=window_size,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        epochs=args.epochs,
        verbose=0
    )
    
    # Save model
    model.save("eurojackpot_lstm_model.keras")
    print("Trained model saved to 'eurojackpot_lstm_model.keras'.")
    
    # Load scalers for inference
    scaler_x = joblib.load(train.SCALER_X_PATH)
    scaler_sum = joblib.load(train.SCALER_SUM_PATH)
    scaler_counts = joblib.load(train.SCALER_COUNTS_PATH)
    
    # 5. Backtesting & Metrics
    print("\n[Step 5/6] Performing Hold-out Backtesting (Last 50 Draws)...")
    metrics = backtest.run_backtest(
        model, data_dict, scaler_x, scaler_sum, scaler_counts, val_split=50
    )
    
    print("\n--- Backtest Evaluation Metrics ---")
    print(f"Model Sum MSE: {metrics['sum_mse']:.4f}")
    print(f"Model Sum MAE: {metrics['sum_mae']:.4f} numbers")
    print(f"Model Counts MAE: {metrics['counts_mae']:.4f}")
    print(f"Main Numbers Top-5 Accuracy:  {metrics['main_top5']:.2%}")
    print(f"Main Numbers Top-10 Accuracy: {metrics['main_top10']:.2%}")
    print(f"Main Numbers Top-15 Accuracy: {metrics['main_top15']:.2%}")
    print(f"Euro Numbers Top-2 Accuracy:  {metrics['euro_top2']:.2%}")
    print(f"Euro Numbers Top-4 Accuracy:  {metrics['euro_top4']:.2%}")
    
    # Get layer sparsity
    sparsity_dict = {}
    for name in ["sum_head", "counts_head", "main_logits_head", "euro_logits_head"]:
        layer = model.get_layer(name)
        kernel = layer.get_weights()[0]
        sparsity_dict[name] = float(np.mean(kernel == 0.0))
        
    # 6. Prediction Generation for next draw
    print("\n[Step 6/6] Generating Predictions for Next Draw...")
    # Extract last w draws sequence
    last_inputs = get_last_window_inputs(df_features, window_size=window_size)
    
    # Scale inputs
    X_num_scaled = train.transform_3d(scaler_x, last_inputs['X_num'])
    
    # Predict
    pred_sum_scaled, pred_counts_scaled, pred_main_probs, pred_euro_probs = model.predict(
        [X_num_scaled, last_inputs['X_main'], last_inputs['X_euro']],
        verbose=0
    )
    
    # Inverse scale predictions
    next_pred_sum = float(scaler_sum.inverse_transform(pred_sum_scaled)[0, 0])
    next_pred_counts = scaler_counts.inverse_transform(pred_counts_scaled)[0]
    
    # Logits vectors
    next_main_probs = pred_main_probs[0]
    next_euro_probs = pred_euro_probs[0]
    
    print(f"\n--- Model Next-Draw structural estimates ---")
    print(f"Predicted Total Sum: {next_pred_sum:.2f}")
    print(f"Predicted Count Ranges: Even={next_pred_counts[0]:.2f}, Odd={next_pred_counts[1]:.2f}, Low={next_pred_counts[2]:.2f}, High={next_pred_counts[3]:.2f}")
    
    # Generate bets for different temperatures
    print(f"\nGenerating {args.bets_count} bets per risk profile...")
    bets_conservative = generator.generate_bets(
        next_main_probs, next_euro_probs, next_pred_sum, next_pred_counts,
        temperature=0.2, count=args.bets_count
    )
    bets_balanced = generator.generate_bets(
        next_main_probs, next_euro_probs, next_pred_sum, next_pred_counts,
        temperature=1.0, count=args.bets_count
    )
    bets_unique = generator.generate_bets(
        next_main_probs, next_euro_probs, next_pred_sum, next_pred_counts,
        temperature=2.0, count=args.bets_count
    )
    # Convert numpy types to native Python ints for clean output
    bets_conservative = [([int(x) for x in m], [int(y) for y in e]) for m, e in bets_conservative]
    bets_balanced = [([int(x) for x in m], [int(y) for y in e]) for m, e in bets_balanced]
    bets_unique = [([int(x) for x in m], [int(y) for y in e]) for m, e in bets_unique]
    
    
    # Print results
    print("\n>>> CONSERVATIVE BETS (Low Temp T=0.2, Hot numbers) <<<")
    for idx, (m, e) in enumerate(bets_conservative):
        print(f"  Combination #{idx+1:02d}: Main {m} | Euro {e}")
        
    print("\n>>> BALANCED BETS (T=1.0) <<<")
    for idx, (m, e) in enumerate(bets_balanced):
        print(f"  Combination #{idx+1:02d}: Main {m} | Euro {e}")
        
    print("\n>>> UNIQUE BETS (High Temp T=2.0, minimizes split-jackpot risk) <<<")
    for idx, (m, e) in enumerate(bets_unique):
        print(f"  Combination #{idx+1:02d}: Main {m} | Euro {e}")
        
    # Write walkthrough.md report in the artifact directory
    write_walkthrough_report(metrics, sparsity_dict, next_pred_sum, next_pred_counts, bets_conservative, bets_balanced, bets_unique)

def write_walkthrough_report(metrics, sparsity, pred_sum, pred_counts, conservative, balanced, unique):
    """
    Creates the walkthrough.md artifact with training metrics and prediction combinations.
    """
    artifact_dir = "C:\\Users\\Acer\\.gemini\\antigravity\\brain\\31f05b5c-3bb6-453d-878d-498e7d64a5f3"
    walkthrough_path = os.path.join(artifact_dir, "walkthrough.md")
    
    # Pre-format lists to avoid backslashes inside f-strings (for Python < 3.12 compatibility)
    conservative_str = "".join([f"* **Combination {i+1}**: Main `{m}` | Euro `{e}`\n" for i, (m, e) in enumerate(conservative)])
    balanced_str = "".join([f"* **Combination {i+1}**: Main `{m}` | Euro `{e}`\n" for i, (m, e) in enumerate(balanced)])
    unique_str = "".join([f"* **Combination {i+1}**: Main `{m}` | Euro `{e}`\n" for i, (m, e) in enumerate(unique)])
    
    # Build markdown content
    content = f"""# Eurojackpot AI Forecasting walkthrough & predictions

This walkthrough details the performance results, network sparsity levels, and output predictions generated for the next Eurojackpot draw.

## Backtesting Metrics (Validation on Last 50 Draws)

| Metric | Value | Interpretation |
|---|---|---|
| **Model Sum MSE** | {metrics['sum_mse']:.4f} | Mean Squared Error for prediction of the draw total sum. |
| **Model Sum MAE** | {metrics['sum_mae']:.4f} | Average absolute distance from true sum. |
| **Model Counts MAE** | {metrics['counts_mae']:.4f} | Average error on parity (even/odd) and size (low/high) counts. |
| **Main Top-5 Accuracy** | {metrics['main_top5']:.2%} | Fraction of drawn main numbers present in the model's top 5 predictions. |
| **Main Top-10 Accuracy** | {metrics['main_top10']:.2%} | Fraction of drawn main numbers present in the model's top 10 predictions. |
| **Main Top-15 Accuracy** | {metrics['main_top15']:.2%} | Fraction of drawn main numbers present in the model's top 15 predictions. |
| **Euro Top-2 Accuracy** | {metrics['euro_top2']:.2%} | Fraction of drawn Euro numbers present in the model's top 2 predictions. |
| **Euro Top-4 Accuracy** | {metrics['euro_top4']:.2%} | Fraction of drawn Euro numbers present in the model's top 4 predictions. |

---

## Model Sparsity via Iterative Magnitude Pruning (IMP)

The Multi-Modal network classification and regression heads were pruned to 80% sparsity. The final zeros fraction of the dense layers after retraining is:

* **Sum Head Sparsity**: {sparsity['sum_head']:.1%}
* **Counts Head Sparsity**: {sparsity['counts_head']:.1%}
* **Main Logits Head Sparsity**: {sparsity['main_logits_head']:.1%}
* **Euro Logits Head Sparsity**: {sparsity['euro_logits_head']:.1%}

---

## Next Draw Predictions & Structural Bounds

### Estimates
* **Estimated Next Draw Sum**: {pred_sum:.2f}
* **Estimated Count Ranges**:
  - Even numbers: {pred_counts[0]:.2f} (Odd: {pred_counts[1]:.2f})
  - Low numbers (1-25): {pred_counts[2]:.2f} (High: {pred_counts[3]:.2f})

### Generated Bet Combinations

````carousel
### Conservative Bets (T=0.2)
*Low temperature sampling to prioritize 'hot' numbers predicted by the engine.*

{conservative_str}
<!-- slide -->
### Balanced Bets (T=1.0)
*Standard sampling balance matching historical patterns and model outputs.*

{balanced_str}
<!-- slide -->
### Unique Bets (T=2.0)
*High temperature sampling to generate unexpected combinations to minimize split-jackpot risk.*

{unique_str}
````
"""
    
    print(f"\nWriting walkthrough artifact to {walkthrough_path}...")
    
    # We must save this using the standard write file path
    with open(walkthrough_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("walkthrough.md successfully generated.")

if __name__ == "__main__":
    main()
