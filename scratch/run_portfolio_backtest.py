import os
import sys
import numpy as np
import pandas as pd
import json
import sqlite3
import tensorflow as tf
import joblib

# Resolve project path dynamically
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import database
import features
import train
import crypto.quantum_lotto as quantum_lotto
import crypto.ticket_optimizer as ticket_optimizer

# Official Eurojackpot prize tiers definition
PRIZE_TIERS = {
    (5, 2): "5+2 (Jackpot!)",
    (5, 1): "5+1 (Tier 2)",
    (5, 0): "5+0 (Tier 3)",
    (4, 2): "4+2 (Tier 4)",
    (4, 1): "4+1 (Tier 5)",
    (4, 0): "4+0 (Tier 6)",
    (3, 2): "3+2 (Tier 7)",
    (3, 1): "3+1 (Tier 8)",
    (2, 2): "2+2 (Tier 9)",
    (3, 0): "3+0 (Tier 10)",
    (1, 2): "1+2 (Tier 11)",
    (2, 1): "2+1 (Tier 12)",
}

def get_prize_tier(main_hits, euro_hits):
    return PRIZE_TIERS.get((main_hits, euro_hits), "No Win")

def is_winning_tier(main_hits, euro_hits):
    # Winning tiers in Eurojackpot are: 2+1, 1+2, 2+2, 3+0, 3+1, 3+2, 4+0, 4+1, 4+2, 5+0, 5+1, 5+2
    # Standard 2+0 is NOT a winning tier in Eurojackpot (since late 2022).
    # Winning tiers require at least 2+1, 1+2, or 3+0.
    return (main_hits, euro_hits) in PRIZE_TIERS

def run_portfolio_backtest():
    features_path = os.path.join(project_path, "physical_features.json")
    db_path = os.path.join(project_path, "eurojackpot.db")
    model_path = os.path.join(project_path, "eurojackpot_lstm_model.keras")
    
    if not os.path.exists(features_path) or not os.path.exists(db_path) or not os.path.exists(model_path):
        print("Missing physical features, database, or LSTM model.")
        return
        
    # 1. Load physical features
    with open(features_path, "r") as f:
        physical_features = json.load(f)
        
    physical_map = {}
    for fn, feat in physical_features.items():
        name = fn.replace("Eurojackpot - ", "").replace("Eurojackpot ", "")
        parts = name.split(" - Allwyn")
        if len(parts) >= 2:
            date_str = parts[0].strip()
            try:
                day, month, year = [int(x) for x in date_str.split(".")]
                db_date = f"{year}-{month:02d}-{day:02d}"
                physical_map[db_date] = feat
            except Exception:
                pass
                
    # Calculate baseline average physical values
    avg_k = [f["avg_kinetic_energy"] for f in physical_map.values()]
    max_k = [f["max_kinetic_energy"] for f in physical_map.values()]
    std_k = [f["std_kinetic_energy"] for f in physical_map.values()]
    col_f = [f["collision_frequency"] for f in physical_map.values()]
    eject_s = [f["avg_ejection_speed"] for f in physical_map.values()]
    
    global_avg_k = np.mean(avg_k) if avg_k else 13.0
    global_max_k = np.mean(max_k) if max_k else 35.0
    global_std_k = np.mean(std_k) if std_k else 9.8
    global_avg_col = np.mean(col_f) if col_f else 260.0
    global_avg_eject = np.mean(eject_s) if eject_s else 14.5
    
    # 2. Load model and scalers
    model = tf.keras.models.load_model(model_path)
    scaler_x = joblib.load(os.path.join(project_path, "scaler_x.joblib"))
    scaler_sum = joblib.load(os.path.join(project_path, "scaler_sum.joblib"))
    scaler_counts = joblib.load(os.path.join(project_path, "scaler_counts.joblib"))
    
    # 3. Fetch draws
    draws_raw = database.get_all_draws()
    df_features = features.compute_draw_features(draws_raw)
    
    # Filter draws to match exactly the rows in df_features
    valid_dates = set(df_features['date'].values)
    draws = [d for d in draws_raw if d['date'] in valid_dates]
    
    window_size = 10
    
    feature_cols = [
        'mean', 'std', 'median', 'sum', 'product_diff',
        'even_count', 'odd_count', 'low_count', 'high_count',
        'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
    ]
    num_features_all = df_features[feature_cols].values
    main_nums_all = df_features[['num1', 'num2', 'num3', 'num4', 'num5']].values
    euro_nums_all = df_features[['euro1', 'euro2']].values
    
    # 4. Construct QRC inputs for the entire database
    inputs_8q = []
    targets_main = []
    targets_euro = []
    
    for i in range(len(draws)):
        past = draws[max(0, i-5):i]
        if not past:
            inputs_8q.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            flat_past = [x for d in past for x in d['main_nums']]
            m = np.mean(flat_past)
            s = sum(past[-1]['main_nums'])
            even_count = sum(1 for x in past[-1]['main_nums'] if x % 2 == 0)
            high_count = sum(1 for x in past[-1]['main_nums'] if x > 25)
            
            d_date = draws[i]['date']
            p_avg = physical_map[d_date]["avg_kinetic_energy"] if d_date in physical_map else global_avg_k
            p_max = physical_map[d_date]["max_kinetic_energy"] if d_date in physical_map else global_max_k
            p_std = physical_map[d_date]["std_kinetic_energy"] if d_date in physical_map else global_std_k
            p_col = physical_map[d_date]["collision_frequency"] if d_date in physical_map else global_avg_col
            p_eject = physical_map[d_date]["avg_ejection_speed"] if d_date in physical_map else global_avg_eject
            
            f1 = (m - 25.0) / 25.0
            f2 = (s - 125.0) / 125.0
            f3 = (even_count - 2.5) / 2.5
            f4 = (high_count - 2.5) / 2.5
            
            # Sigmoid / Min-Max scaled features to ensure perfect [-1, 1] bounds
            f5 = 2.0 * (p_avg - 9.0) / (18.0 - 9.0) - 1.0
            f6 = 2.0 * (p_max - 30.0) / (37.0 - 30.0) - 1.0
            f7 = 2.0 * (p_col - 0.0) / (400.0 - 0.0) - 1.0
            f8 = 2.0 * (p_eject - 2.0) / (25.0 - 2.0) - 1.0
            inputs_8q.append([f1, f2, f3, f4, f5, f6, f7, f8])
            
        t_m = np.zeros(50)
        for n in draws[i]['main_nums']:
            t_m[n-1] = 1.0
        targets_main.append(t_m)
        
        t_e = np.zeros(12)
        for n in draws[i]['euro_nums']:
            t_e[n-1] = 1.0
        targets_euro.append(t_e)
        
    targets_main = np.array(targets_main)
    targets_euro = np.array(targets_euro)
    
    # Run reservoir once for all draws
    print("Simulating 8-qubit Quantum Reservoir features...")
    qrc_8q = quantum_lotto.QuantumReservoir(n_qubits=8, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    feats_8q = []
    for inp in inputs_8q:
        feats_8q.append(qrc_8q.step(inp))
    shrimp_8q, W_8q = quantum_lotto.shrimp_random_features(np.array(feats_8q), num_features=50, sparsity=0.7)
    
    print(f"DEBUG: len(draws)={len(draws)}, len(inputs_8q)={len(inputs_8q)}, feats_8q shape={np.array(feats_8q).shape}, shrimp_8q shape={shrimp_8q.shape}")
    
    # 5. Backtest loop on draws having matched physical features
    # Filter draws matching physical maps
    phys_indices = [idx for idx, d in enumerate(draws) if d['date'] in physical_map and idx >= window_size]
    
    print(f"Starting Walk-forward Portfolio Simulation on {len(phys_indices)} matched draws...")
    
    total_draws_sim = 0
    winning_draws_count = 0
    hits_distribution = {}
    draw_results = []
    
    # Store exact details for top wins
    top_wins = []
    
    for idx in phys_indices:
        target_draw = draws[idx]
        actual_main = set(target_draw['main_nums'])
        actual_euro = set(target_draw['euro_nums'])
        
        # --- Run LSTM on window up to idx-1 ---
        X_num_last = np.expand_dims(num_features_all[idx-window_size:idx], axis=0).astype(np.float32)
        X_main_last = np.expand_dims(main_nums_all[idx-window_size:idx], axis=0).astype(np.int32)
        X_euro_last = np.expand_dims(euro_nums_all[idx-window_size:idx], axis=0).astype(np.int32)
        X_num_scaled = train.transform_3d(scaler_x, X_num_last)
        
        pred_sum_scaled, pred_counts_scaled, pred_main_probs, pred_euro_probs = model.predict(
            [X_num_scaled, X_main_last, X_euro_last], verbose=0
        )
        
        lstm_sum = float(scaler_sum.inverse_transform(pred_sum_scaled)[0, 0])
        lstm_counts = scaler_counts.inverse_transform(pred_counts_scaled)[0]
        lstm_main_probs = pred_main_probs[0]
        lstm_euro_probs = pred_euro_probs[0]
        
        # --- Run QRC Readout up to idx-1 ---
        Z_8q = shrimp_8q[:idx]
        t_main_train = targets_main[5:idx] # Aligning QRC offset
        t_euro_train = targets_euro[5:idx]
        Z_train = Z_8q[5:idx]
        
        ridge_inv = np.linalg.inv(Z_train.T @ Z_train + 1.0 * np.eye(Z_train.shape[1]))
        
        W_out_main = ridge_inv @ Z_train.T @ t_main_train
        qrc_main_amps = np.clip((shrimp_8q[idx] @ W_out_main), 0.0, 1.0)
        qrc_main_probs = qrc_main_amps / np.sum(qrc_main_amps)
        
        W_out_euro = ridge_inv @ Z_train.T @ t_euro_train
        qrc_euro_amps = np.clip((shrimp_8q[idx] @ W_out_euro), 0.0, 1.0)
        qrc_euro_probs = qrc_euro_amps / np.sum(qrc_euro_amps)
        
        # --- Fusion ---
        hybrid_main_probs = 0.55 * lstm_main_probs + 0.45 * qrc_main_probs
        hybrid_main_probs /= np.sum(hybrid_main_probs)
        
        hybrid_euro_probs = 0.55 * lstm_euro_probs + 0.45 * qrc_euro_probs
        hybrid_euro_probs /= np.sum(hybrid_euro_probs)
        
        # --- Simulated Annealing Ticket Optimization (6 Tickets Portfolio) ---
        optimizer = ticket_optimizer.TicketOptimizer(
            main_probs=hybrid_main_probs,
            euro_probs=hybrid_euro_probs,
            pred_sum=lstm_sum,
            pred_counts=lstm_counts
        )
        
        # We run 1000 steps for fast simulation in backtest
        bets = optimizer.optimize(count=6, steps=1000)
        
        # --- Evaluate matches for all 6 tickets ---
        best_tier_name = "No Win"
        best_tier_tuple = (0, 0)
        best_ticket = None
        
        for main_nums, euro_nums in bets:
            main_hits = len(actual_main.intersection(main_nums))
            euro_hits = len(actual_euro.intersection(euro_nums))
            tier_name = get_prize_tier(main_hits, euro_hits)
            
            # Compare tiers to find best (lower index or higher hits is better)
            if is_winning_tier(main_hits, euro_hits):
                if best_tier_name == "No Win" or (main_hits + euro_hits > best_tier_tuple[0] + best_tier_tuple[1]):
                    best_tier_name = tier_name
                    best_tier_tuple = (main_hits, euro_hits)
                    best_ticket = (main_nums, euro_nums)
                    
        total_draws_sim += 1
        if best_tier_name != "No Win":
            winning_draws_count += 1
            hits_distribution[best_tier_name] = hits_distribution.get(best_tier_name, 0) + 1
            
            # Record top wins (3+0, 2+2, 3+1, 4+0, 4+1, 5+0)
            if best_tier_tuple[0] >= 3 or (best_tier_tuple[0] == 2 and best_tier_tuple[1] == 2):
                top_wins.append({
                    "date": target_draw['date'],
                    "tier": best_tier_name,
                    "ticket_main": best_ticket[0],
                    "ticket_euro": best_ticket[1],
                    "actual_main": list(actual_main),
                    "actual_euro": list(actual_euro)
                })
        else:
            hits_distribution["No Win"] = hits_distribution.get("No Win", 0) + 1
            
        print(f"Draw {target_draw['date']}: Actual={list(actual_main)}+{list(actual_euro)} | Best Portfolio Win={best_tier_name}")

    winrate = (winning_draws_count / total_draws_sim) * 100 if total_draws_sim > 0 else 0
    
    print("\n========================================================")
    # Print results to the user in a beautiful format
    print("      LIVE 6-TICKET PORTFOLIO BACKTEST SUMMARY        ")
    print(f"      (Tested on {total_draws_sim} matched physical drawings) ")
    print("========================================================")
    print(f"Total Draws Simulated : {total_draws_sim}")
    print(f"Winning Draws (>=2+1) : {winning_draws_count} / {total_draws_sim}")
    print(f"Portfolio Winrate     : {winrate:.1f}%")
    print("--------------------------------------------------------")
    print("Hit Category Distribution in Portfolio:")
    for tier, count in sorted(hits_distribution.items(), key=lambda x: x[0]):
        print(f"  - {tier:<18} : {count} times")
    print("--------------------------------------------------------")
    
    if top_wins:
        print("EXACT HIGH-TIER HITS RECORDED IN BACKTEST:")
        for w in top_wins:
            print(f"  * Date: {w['date']} | Win: {w['tier']}")
            print(f"    - Predicted: {w['ticket_main']} + {w['ticket_euro']}")
            print(f"    - Drawn    : {w['actual_main']} + {w['actual_euro']}")
    else:
        print("No high-tier hits (3+0 or higher) recorded in this run.")
    print("========================================================")

if __name__ == "__main__":
    run_portfolio_backtest()
