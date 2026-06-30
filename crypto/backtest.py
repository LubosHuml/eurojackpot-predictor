import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
sys.path.append(os.path.join(project_path, "crypto"))

import bybit_client
import features
import train

def run_crypto_backtest(val_split=150):
    # 1. Load assets
    crypto_dir = os.path.dirname(__file__)
    model_path = os.path.join(crypto_dir, train.MODEL_PATH)
    scaler_path = os.path.join(crypto_dir, train.SCALER_PATH)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Train the model first.")
        return
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Fetch and prepare data
    df = bybit_client.fetch_historical_klines(symbol="BTCUSDT", interval="60", limit=1000)
    df_indicators = features.calculate_indicators(df)
    
    window_size = 20
    data_dict = features.generate_sequences(df_indicators, window_size=window_size)
    
    X_val = data_dict["X"][-val_split:]
    y_val = data_dict["y"][-val_split:]
    val_dates = data_dict["datetimes"][-val_split:]
    val_closes = data_dict["closes"][-val_split:]
    
    # 3. Scale input
    samples, w, num_feats = X_val.shape
    X_val_flat = X_val.reshape(-1, num_feats)
    X_val_scaled_flat = scaler.transform(X_val_flat)
    X_val_scaled = X_val_scaled_flat.reshape(samples, w, num_feats)
    
    # 4. Predict probabilities
    pred_probs = model.predict(X_val_scaled, verbose=0).flatten()
    
    # 5. Simulate Trading Strategy (Long-Short with Threshold)
    # Signal: 1.0 (Long), -1.0 (Short), 0.0 (Flat)
    signals = []
    for p in pred_probs:
        if p > 0.51:
            signals.append(1.0)
        elif p < 0.49:
            signals.append(-1.0)
        else:
            signals.append(0.0)
            
    signals = np.array(signals)
    
    # Target values (actual direction: 1.0 or 0.0)
    # Win rate computation: signal direction matches target direction
    correct_preds = 0
    total_active_preds = 0
    for i in range(len(pred_probs)):
        actual_up = y_val[i, 0] > 0.5
        predicted_up = pred_probs[i] > 0.5
        if actual_up == predicted_up:
            correct_preds += 1
            
    win_rate = (correct_preds / len(pred_probs)) * 100
    
    # Compute returns
    # BTC log returns
    closes = np.array(val_closes)
    btc_returns = np.diff(np.log(closes))
    
    # Align signals (shift by 1 so we trade based on previous signal)
    strategy_signals = signals[:-1]
    strat_returns = strategy_signals * btc_returns
    
    # Simulate Bybit Taker Fee: 0.06% per transaction
    # We pay a fee whenever we change our position
    fee_rate = 0.0006
    transaction_costs = []
    
    prev_sig = 0.0
    for sig in strategy_signals:
        if sig != prev_sig:
            # Closing old and opening new, or just entering/exiting
            cost = fee_rate if prev_sig == 0.0 or sig == 0.0 else fee_rate * 2
            transaction_costs.append(cost)
        else:
            transaction_costs.append(0.0)
        prev_sig = sig
        
    transaction_costs = np.array(transaction_costs)
    
    # Net returns after fees
    net_strat_returns = strat_returns - transaction_costs
    
    # Cumulative returns
    cum_btc = np.exp(np.cumsum(btc_returns))
    cum_strat = np.exp(np.cumsum(net_strat_returns))
    
    # Equity curves (starting with $10,000)
    initial_capital = 10000.0
    equity_btc = initial_capital * cum_btc
    equity_strat = initial_capital * cum_strat
    
    # Compute max drawdown of strategy
    running_max = np.maximum.accumulate(equity_strat)
    drawdowns = (equity_strat - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100
    
    # Compute profit factor
    pos_returns = net_strat_returns[net_strat_returns > 0]
    neg_returns = net_strat_returns[net_strat_returns < 0]
    gross_profits = np.sum(pos_returns)
    gross_losses = -np.sum(neg_returns)
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Trading counts
    total_trades = np.sum(transaction_costs > 0)
    
    # Prepare results JSON
    results = {
        "win_rate": float(win_rate),
        "total_trades": int(total_trades),
        "btc_return": float((cum_btc[-1] - 1) * 100),
        "strat_return": float((cum_strat[-1] - 1) * 100),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "initial_capital": initial_capital,
        "final_capital": float(equity_strat[-1]),
        "history_equity": [float(x) for x in equity_strat.tolist()]
    }
    
    # Save backtest results as JSON cache
    metrics_save_path = os.path.join(crypto_dir, "crypto_backtest_results.json")
    import json
    with open(metrics_save_path, "w") as f:
        json.dump(results, f)
        
    print("\n===========================================================")
    print("      BTC/USDT AI PREDICTOR: QUANTITATIVE BACKTEST SHEET")
    print("===========================================================")
    print(f"Validation Horizon:       {val_split} hours (approx. 6 days)")
    print(f"Starting Capital:         {initial_capital:,.2f} USDT")
    print(f"Directional Win Rate:     {win_rate:.2f}%")
    print(f"Total Trades Executed:    {total_trades}")
    print(f"Profit Factor:            {profit_factor:.2f}")
    print(f"Max Strategy Drawdown:    {max_drawdown:.2f}%")
    print("-----------------------------------------------------------")
    print(f"Buy-and-Hold BTC Return:  {results['btc_return']:+.2f}% (Final: {equity_btc[-1]:,.2f} USDT)")
    print(f"AI Strategy Net Return:   {results['strat_return']:+.2f}% (Final: {results['final_capital']:,.2f} USDT)")
    print("===========================================================")
    
    return results

if __name__ == "__main__":
    run_crypto_backtest()
