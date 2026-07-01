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

PRIZE_TIERS = {
    (5, 2): "5+2", (5, 1): "5+1", (5, 0): "5+0",
    (4, 2): "4+2", (4, 1): "4+1", (4, 0): "4+0",
    (3, 2): "3+2", (3, 1): "3+1", (2, 2): "2+2",
    (3, 0): "3+0", (1, 2): "1+2", (2, 1): "2+1",
}

def is_winning(main_hits, euro_hits):
    return (main_hits, euro_hits) in PRIZE_TIERS

def simulate_portfolio(draws, df_features, physical_map, model, scalers, shrimp_8q, targets_main, targets_euro, ticket_count):
    window_size = 10
    phys_indices = [idx for idx, d in enumerate(draws) if d['date'] in physical_map and idx >= window_size]
    
    feature_cols = [
        'mean', 'std', 'median', 'sum', 'product_diff',
        'even_count', 'odd_count', 'low_count', 'high_count',
        'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
    ]
    num_features_all = df_features[feature_cols].values
    main_nums_all = df_features[['num1', 'num2', 'num3', 'num4', 'num5']].values
    euro_nums_all = df_features[['euro1', 'euro2']].values
    
    scaler_x, scaler_sum, scaler_counts = scalers
    
    winning_weeks = 0
    total_weeks = 0
    
    for idx in phys_indices:
        target_draw = draws[idx]
        actual_main = set(target_draw['main_nums'])
        actual_euro = set(target_draw['euro_nums'])
        
        # LSTM
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
        
        # QRC
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
        
        # Optimize
        optimizer = ticket_optimizer.TicketOptimizer(
            main_probs=hybrid_main_probs,
            euro_probs=hybrid_euro_probs,
            pred_sum=lstm_sum,
            pred_counts=lstm_counts
        )
        
        bets = optimizer.optimize(count=ticket_count, steps=800)
        
        # Check if any ticket is winning
        won_this_week = False
        for main_nums, euro_nums in bets:
            main_hits = len(actual_main.intersection(main_nums))
            euro_hits = len(actual_euro.intersection(euro_nums))
            if is_winning(main_hits, euro_hits):
                won_this_week = True
                break
                
        total_weeks += 1
        if won_this_week:
            winning_weeks += 1
            
    return winning_weeks, total_weeks

def main():
    features_path = os.path.join(project_path, "physical_features.json")
    db_path = os.path.join(project_path, "eurojackpot.db")
    model_path = os.path.join(project_path, "eurojackpot_lstm_model.keras")
    
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
    
    model = tf.keras.models.load_model(model_path)
    scaler_x = joblib.load(os.path.join(project_path, "scaler_x.joblib"))
    scaler_sum = joblib.load(os.path.join(project_path, "scaler_sum.joblib"))
    scaler_counts = joblib.load(os.path.join(project_path, "scaler_counts.joblib"))
    
    draws_raw = database.get_all_draws()
    df_features = features.compute_draw_features(draws_raw)
    valid_dates = set(df_features['date'].values)
    draws = [d for d in draws_raw if d['date'] in valid_dates]
    
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
            f5 = (p_avg - 13.0) / 2.0
            f6 = (p_max - 35.0) / 3.0
            f7 = (p_col - 260.0) / 50.0
            f8 = (p_eject - 14.5) / 3.0
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
    
    qrc_8q = quantum_lotto.QuantumReservoir(n_qubits=8, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    feats_8q = []
    for inp in inputs_8q:
        feats_8q.append(qrc_8q.step(inp))
    shrimp_8q, W_8q = quantum_lotto.shrimp_random_features(np.array(feats_8q), num_features=50, sparsity=0.7)
    
    scalers = (scaler_x, scaler_sum, scaler_counts)
    
    print("\n--- Running simulations for different portfolio sizes ---")
    for size in [6, 12, 18, 24]:
        wins, total = simulate_portfolio(draws, df_features, physical_map, model, scalers, shrimp_8q, targets_main, targets_euro, size)
        rate = (wins / total) * 100
        empty_weeks = total - wins
        print(f"Portfolio size: {size:<2} tickets | Winning weeks: {wins}/{total} | Winrate: {rate:.1f}% | Empty weeks: {empty_weeks}")

if __name__ == "__main__":
    main()
