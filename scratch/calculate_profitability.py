import os
import sys
import numpy as np
import pandas as pd
import json
import sqlite3
import tensorflow as tf
import joblib

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import database
import features
import train
import crypto.quantum_lotto as quantum_lotto
import crypto.ticket_optimizer as ticket_optimizer

# Official Eurojackpot prize payouts in CZK
PRIZE_PAYOUTS = {
    (5, 2): 1573179300,  # 1. pořadí (Jackpot)
    (5, 1): 15032602,    # 2. pořadí
    (5, 0): 3767862,     # 3. pořadí
    (4, 2): 124301,      # 4. pořadí
    (4, 1): 7769,        # 5. pořadí
    (3, 2): 3884,        # 6. pořadí
    (4, 0): 2762,        # 7. pořadí
    (2, 2): 628,         # 8. pořadí
    (3, 1): 503,         # 9. pořadí
    (3, 0): 424,         # 10. pořadí
    (1, 2): 317,         # 11. pořadí
    (2, 1): 250,         # 12. pořadí
}

TICKET_COST_CZK = 60 # Cost per row/column in Czech Republic Sazka

def get_payout(main_hits, euro_hits):
    return PRIZE_PAYOUTS.get((main_hits, euro_hits), 0)

def run_profitability_analysis():
    features_path = os.path.join(project_path, "physical_features.json")
    db_path = os.path.join(project_path, "eurojackpot.db")
    model_path = os.path.join(project_path, "eurojackpot_lstm_model.keras")
    
    if not os.path.exists(features_path) or not os.path.exists(db_path) or not os.path.exists(model_path):
        print("Missing physical features, database, or LSTM model.")
        return
        
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
    
    # Pre-load LSTM models to save time
    print("Loading LSTM neural network model...")
    model = tf.keras.models.load_model(model_path)
    scaler_x = joblib.load(os.path.join(project_path, "scaler_x.joblib"))
    scaler_sum = joblib.load(os.path.join(project_path, "scaler_sum.joblib"))
    scaler_counts = joblib.load(os.path.join(project_path, "scaler_counts.joblib"))
    
    draws_raw = database.get_all_draws()
    df_features = features.compute_draw_features(draws_raw)
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
    
    print("Simulating 8-qubit Quantum Reservoir features once...")
    qrc_8q = quantum_lotto.QuantumReservoir(n_qubits=8, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    feats_8q = []
    for inp in inputs_8q:
        feats_8q.append(qrc_8q.step(inp))
    shrimp_8q, W_8q = quantum_lotto.shrimp_random_features(np.array(feats_8q), num_features=50, sparsity=0.7)
    
    # Filter draws matching physical maps (we take the last 50 draws in the database)
    all_matched_draw_indices = [idx for idx, d in enumerate(draws) if d['date'] in physical_map and idx >= window_size]
    # Limit to the last 50 draws
    phys_indices = all_matched_draw_indices[-50:]
    
    print(f"Running financial profitability simulation on the last {len(phys_indices)} draws...")
    
    total_cost = len(phys_indices) * 6 * TICKET_COST_CZK
    total_winnings = 0
    payout_details = []
    
    # Store wins count per tier
    wins_per_tier = {}
    
    for count_idx, idx in enumerate(phys_indices):
        target_draw = draws[idx]
        actual_main = set(target_draw['main_nums'])
        actual_euro = set(target_draw['euro_nums'])
        
        # LSTM prediction
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
        
        # QRC prediction
        Z_train = shrimp_8q[5:idx]
        t_main_train = targets_main[5:idx]
        t_euro_train = targets_euro[5:idx]
        ridge_inv = np.linalg.inv(Z_train.T @ Z_train + 1.0 * np.eye(Z_train.shape[1]))
        
        W_out_main = ridge_inv @ Z_train.T @ t_main_train
        qrc_main_amps = np.clip((shrimp_8q[idx] @ W_out_main), 0.0, 1.0)
        qrc_main_probs = qrc_main_amps / np.sum(qrc_main_amps)
        
        W_out_euro = ridge_inv @ Z_train.T @ t_euro_train
        qrc_euro_amps = np.clip((shrimp_8q[idx] @ W_out_euro), 0.0, 1.0)
        qrc_euro_probs = qrc_euro_amps / np.sum(qrc_euro_amps)
        
        # Fusion
        hybrid_main_probs = 0.55 * lstm_main_probs + 0.45 * qrc_main_probs
        hybrid_main_probs /= np.sum(hybrid_main_probs)
        hybrid_euro_probs = 0.55 * lstm_euro_probs + 0.45 * qrc_euro_probs
        hybrid_euro_probs /= np.sum(hybrid_euro_probs)
        
        # SA Optimize
        optimizer = ticket_optimizer.TicketOptimizer(
            main_probs=hybrid_main_probs,
            euro_probs=hybrid_euro_probs,
            pred_sum=lstm_sum,
            pred_counts=lstm_counts
        )
        bets = optimizer.optimize(count=6, steps=1000)
        
        # Evaluate draw payouts
        draw_payout = 0
        draw_wins = []
        for main_nums, euro_nums in bets:
            main_hits = len(actual_main.intersection(main_nums))
            euro_hits = len(actual_euro.intersection(euro_nums))
            payout = get_payout(main_hits, euro_hits)
            if payout > 0:
                draw_payout += payout
                tier_name = f"{main_hits}+{euro_hits}"
                draw_wins.append(f"{tier_name} ({payout:,} Kč)")
                wins_per_tier[tier_name] = wins_per_tier.get(tier_name, 0) + 1
                
        total_winnings += draw_payout
        if draw_payout > 0:
            payout_details.append({
                "date": target_draw['date'],
                "wins": ", ".join(draw_wins),
                "total_payout": draw_payout
            })
        print(f"[{count_idx+1}/{len(phys_indices)}] Draw {target_draw['date']} evaluated. Winnings: {draw_payout:,} Kč")
            
    net_profit = total_winnings - total_cost
    roi = (total_winnings / total_cost) * 100 if total_cost > 0 else 0
    
    print("\n========================================================")
    print("      FINANCIAL PROFITABILITY ANALYSIS REPORT         ")
    print(f"      (Simulation on last {len(phys_indices)} matched draws - 6 Tickets) ")
    print("========================================================")
    print(f"Total Invested Capital (Sázky) : {total_cost:,} Kč")
    print(f"Total Won Capital (Výhry)       : {total_winnings:,} Kč")
    print(f"Net Profit / Loss (Čistý zisk)  : {net_profit:+,} Kč")
    print(f"Return on Investment (ROI)      : {roi:.2f}%")
    print("--------------------------------------------------------")
    print("Hit Distribution by Prize Tier:")
    for tier, count in sorted(wins_per_tier.items()):
        print(f"  - {tier}: {count} times")
    print("--------------------------------------------------------")
    print("Detail of Draws with Winnings:")
    for d in payout_details:
        print(f"  * Draw: {d['date']} | Total Win: {d['total_payout']:,} Kč")
        print(f"    - Breakdown: {d['wins']}")
    print("========================================================")

if __name__ == "__main__":
    run_profitability_analysis()
