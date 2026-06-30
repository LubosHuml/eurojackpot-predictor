import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
sys.path.append(os.path.join(project_path, "crypto"))

import bybit_client
import features

def run_single_backtest(symbol, val_split=150):
    crypto_dir = os.path.dirname(__file__)
    sym_lower = symbol.lower().replace("/", "")
    
    model_path = os.path.join(crypto_dir, f"crypto_lstm_model_{sym_lower}.keras")
    scaler_path = os.path.join(crypto_dir, f"crypto_scaler_{sym_lower}.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler not found for {symbol}. Skip backtest.")
        return None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Fetch data
    df = bybit_client.fetch_historical_klines(symbol=symbol, interval="60", limit=1000)
    df_indicators = features.calculate_indicators(df)
    
    window_size = 20
    data_dict = features.generate_sequences(df_indicators, window_size=window_size)
    
    X_val = data_dict["X"][-val_split:]
    y_val = data_dict["y"][-val_split:]
    val_closes = data_dict["closes"][-val_split:]
    
    # Scale input
    samples, w, num_feats = X_val.shape
    X_val_flat = X_val.reshape(-1, num_feats)
    X_val_scaled_flat = scaler.transform(X_val_flat)
    X_val_scaled = X_val_scaled_flat.reshape(samples, w, num_feats)
    
    # Predict
    pred_probs = model.predict(X_val_scaled, verbose=0).flatten()
    
    # Strategy signals
    signals = []
    for p in pred_probs:
        if p > 0.51:
            signals.append(1.0)
        elif p < 0.49:
            signals.append(-1.0)
        else:
            signals.append(0.0)
    signals = np.array(signals)
    
    # Win rate
    correct_preds = 0
    for i in range(len(pred_probs)):
        actual_up = y_val[i, 0] > 0.5
        predicted_up = pred_probs[i] > 0.5
        if actual_up == predicted_up:
            correct_preds += 1
    win_rate = (correct_preds / len(pred_probs)) * 100
    
    # Returns
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
    
    # Net strategy returns
    net_strat_returns = strat_returns - transaction_costs
    
    return {
        "win_rate": win_rate,
        "asset_returns": asset_returns,
        "strat_returns": net_strat_returns,
        "trades": np.sum(transaction_costs > 0)
    }

def run_portfolio_backtest():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results = {}
    
    asset_returns_dict = {}
    strat_returns_dict = {}
    
    for sym in symbols:
        res = run_single_backtest(sym)
        if res is not None:
            results[sym] = res
            asset_returns_dict[sym] = res["asset_returns"]
            strat_returns_dict[sym] = res["strat_returns"]
            
    if not strat_returns_dict:
        print("No assets successfully backtested.")
        return
        
    # Compute combined portfolio returns (1/3 weight each)
    # Align length just in case
    min_len = min(len(r) for r in strat_returns_dict.values())
    
    portfolio_strat_returns = np.zeros(min_len)
    portfolio_asset_returns = np.zeros(min_len)
    
    for sym in strat_returns_dict.keys():
        portfolio_strat_returns += strat_returns_dict[sym][-min_len:] / len(strat_returns_dict)
        portfolio_asset_returns += asset_returns_dict[sym][-min_len:] / len(asset_returns_dict)
        
    cum_asset = np.exp(np.cumsum(portfolio_asset_returns))
    cum_strat = np.exp(np.cumsum(portfolio_strat_returns))
    
    initial_capital = 10000.0
    equity_strat = initial_capital * cum_strat
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity_strat)
    drawdowns = (equity_strat - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100
    
    # Profit factor
    pos_returns = portfolio_strat_returns[portfolio_strat_returns > 0]
    neg_returns = portfolio_strat_returns[portfolio_strat_returns < 0]
    gross_profits = np.sum(pos_returns)
    gross_losses = -np.sum(neg_returns)
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    portfolio_results = {
        "portfolio": {
            "initial_capital": initial_capital,
            "final_capital": float(equity_strat[-1]),
            "strat_return": float((cum_strat[-1] - 1) * 100),
            "asset_return": float((cum_asset[-1] - 1) * 100),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor)
        }
    }
    
    # Add individual asset highlights
    for sym in symbols:
        if sym in results:
            sym_lower = sym.lower()
            cum_asset_ind = np.exp(np.cumsum(results[sym]["asset_returns"]))
            cum_strat_ind = np.exp(np.cumsum(results[sym]["strat_returns"]))
            
            portfolio_results[sym_lower] = {
                "win_rate": float(results[sym]["win_rate"]),
                "trades": int(results[sym]["trades"]),
                "asset_return": float((cum_asset_ind[-1] - 1) * 100),
                "strat_return": float((cum_strat_ind[-1] - 1) * 100)
            }
            
    # Save backtest results as JSON
    crypto_dir = os.path.dirname(__file__)
    metrics_save_path = os.path.join(crypto_dir, "crypto_backtest_results.json")
    with open(metrics_save_path, "w") as f:
        json.dump(portfolio_results, f)
        
    print("\n===========================================================")
    print("      DIVERSIFIED AI PORTFOLIO: QUANTITATIVE SHEET")
    print("===========================================================")
    print(f"Validation Horizon:       150 hours (approx. 6 days)")
    print(f"Number of Assets:         {len(strat_returns_dict)}")
    print("-----------------------------------------------------------")
    for sym in symbols:
        sym_l = sym.lower()
        if sym_l in portfolio_results:
            print(f" * {sym}: Win Rate = {portfolio_results[sym_l]['win_rate']:.2f}%, Trades = {portfolio_results[sym_l]['trades']}, Net Return = {portfolio_results[sym_l]['strat_return']:+.2f}%")
    print("-----------------------------------------------------------")
    print(f"Combined Portfolio Return: {portfolio_results['portfolio']['strat_return']:+.2f}% (Final: {portfolio_results['portfolio']['final_capital']:,.2f} USDT)")
    print(f"Combined Buy-and-Hold:     {portfolio_results['portfolio']['asset_return']:+.2f}%")
    print(f"Max Portfolio Drawdown:    {portfolio_results['portfolio']['max_drawdown']:.2f}%")
    print(f"Portfolio Profit Factor:   {portfolio_results['portfolio']['profit_factor']:.2f}")
    print("===========================================================")
    
    return portfolio_results

if __name__ == "__main__":
    run_portfolio_backtest()
