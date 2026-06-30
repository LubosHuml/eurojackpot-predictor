import os
import sys
import numpy as np
import pandas as pd
import json
import sqlite3

# Resolve project path dynamically
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import crypto.quantum_lotto as quantum_lotto

def run_backtest():
    features_path = os.path.join(project_path, "physical_features.json")
    db_path = os.path.join(project_path, "eurojackpot.db")
    
    if not os.path.exists(features_path) or not os.path.exists(db_path):
        print("Missing physical features or database.")
        return
        
    with open(features_path, "r") as f:
        physical_features = json.load(f)
        
    # Map video filename dates to YYYY-MM-DD
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
                
    # Load all draws
    conn = sqlite3.connect(db_path)
    df_draws = pd.read_sql_query("SELECT date, num1, num2, num3, num4, num5 FROM draws ORDER BY date ASC", conn)
    conn.close()
    
    # Filter to draws for which we have physical features
    df_draws['has_phys'] = df_draws['date'].apply(lambda d: d in physical_map)
    df_phys_draws = df_draws[df_draws['has_phys'] == True].copy()
    
    if len(df_phys_draws) < 15:
        print(f"Not enough matched physical draws ({len(df_phys_draws)}) to run a stable backtest.")
        return
        
    print(f"Starting optimized backtest on {len(df_phys_draws)} physical draws...")
    
    hits_4q = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    hits_8q_phys = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    draws_list = df_phys_draws.to_dict('records')
    
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
    
    # 1. Construct QRC inputs for the entire dataset of matched draws
    inputs_4q = []
    inputs_8q = []
    targets = []
    
    for i in range(len(draws_list)):
        # Compute statistical features
        past = draws_list[max(0, i-5):i]
        if not past:
            inputs_4q.append([0.0, 0.0, 0.0, 0.0])
            inputs_8q.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            flat_past = [x[f'num{j}'] for x in past for j in range(1, 6)]
            m = np.mean(flat_past)
            s = sum([x[f'num{j}'] for j in range(1, 6) for x in [past[-1]]])
            even_count = sum(1 for x in [past[-1]] for j in range(1, 6) if x[f'num{j}'] % 2 == 0)
            high_count = sum(1 for x in [past[-1]] for j in range(1, 6) if x[f'num{j}'] > 25)
            
            # 4-qubit QRC inputs
            inputs_4q.append([(m - 25.0)/25.0, (s - 125.0)/125.0, 0.0, 0.0])
            
            # 8-qubit physical QRC inputs
            d_date = draws_list[i]['date']
            p_avg = physical_map[d_date]["avg_kinetic_energy"] if d_date in physical_map else global_avg_k
            p_max = physical_map[d_date]["max_kinetic_energy"] if d_date in physical_map else global_max_k
            p_std = physical_map[d_date]["std_kinetic_energy"] if d_date in physical_map else global_std_k
            p_col = physical_map[d_date]["collision_frequency"] if d_date in physical_map else global_avg_col
            p_eject = physical_map[d_date]["avg_ejection_speed"] if d_date in physical_map else global_avg_eject
            
            f1 = (m - 25.0) / 25.0
            f2 = (s - 125.0) / 125.0
            f3 = (even_count - 2.5) / 2.5
            f4 = (high_count - 2.5) / 2.5
            
            # Normalize physical features to [-1, 1] range
            f5 = (p_avg - 13.0) / 2.0
            f6 = (p_max - 35.0) / 3.0
            f7 = (p_col - 260.0) / 50.0
            f8 = (p_eject - 14.5) / 3.0
            inputs_8q.append([f1, f2, f3, f4, f5, f6, f7, f8])
            
        t_nums = [draws_list[i]['num1'], draws_list[i]['num2'], draws_list[i]['num3'], draws_list[i]['num4'], draws_list[i]['num5']]
        t_vec = np.zeros(50)
        for n in t_nums:
            t_vec[n-1] = 1.0
        targets.append(t_vec)
        
    targets = np.array(targets)
    
    # 2. Run reservoirs exactly once (extremely fast!)
    print("Simulating 4-qubit Quantum Reservoir...")
    qrc_4q = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    feats_4q = []
    for inp in inputs_4q:
        feats_4q.append(qrc_4q.step(inp))
    shrimp_4q, W_4q = quantum_lotto.shrimp_random_features(np.array(feats_4q), num_features=50, sparsity=0.7)
    
    print("Simulating 8-qubit Physical Quantum Reservoir...")
    qrc_8q = quantum_lotto.QuantumReservoir(n_qubits=8, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    feats_8q = []
    for inp in inputs_8q:
        feats_8q.append(qrc_8q.step(inp))
    shrimp_8q, W_8q = quantum_lotto.shrimp_random_features(np.array(feats_8q), num_features=50, sparsity=0.7)
    
    # 3. Walk-forward loop (super fast, only Ridge fitting!)
    print("Running walk-forward evaluation...")
    for idx in range(10, len(draws_list)):
        target_draw = draws_list[idx]
        actual_nums = {target_draw['num1'], target_draw['num2'], target_draw['num3'], target_draw['num4'], target_draw['num5']}
        
        # 4-qubit prediction
        Z_4q = shrimp_4q[:idx]
        targets_4q = targets[:idx]
        ridge_inv_4q = np.linalg.inv(Z_4q.T @ Z_4q + 1.0 * np.eye(Z_4q.shape[1]))
        W_out_4q = ridge_inv_4q @ Z_4q.T @ targets_4q
        pred_4q = shrimp_4q[idx] @ W_out_4q
        pred_nums_4q = np.argsort(pred_4q)[-5:] + 1
        hits_4q_count = len(actual_nums.intersection(pred_nums_4q))
        hits_4q[hits_4q_count] += 1
        
        # 8-qubit physical prediction
        Z_8q = shrimp_8q[:idx]
        ridge_inv_8q = np.linalg.inv(Z_8q.T @ Z_8q + 1.0 * np.eye(Z_8q.shape[1]))
        W_out_8q = ridge_inv_8q @ Z_8q.T @ targets_4q
        pred_8q = shrimp_8q[idx] @ W_out_8q
        pred_nums_8q = np.argsort(pred_8q)[-5:] + 1
        hits_8q_count = len(actual_nums.intersection(pred_nums_8q))
        hits_8q_phys[hits_8q_count] += 1
        
    print("\n=============================================")
    print("        WALK-FORWARD BACKTEST RESULTS        ")
    print(f"        (Testing on {len(draws_list)-10} consecutive draws) ")
    print("=============================================")
    print("Hit Count | 4-qubit Standard QRC | 8-qubit Physical QRC")
    print("----------+----------------------+---------------------")
    for h in range(6):
        print(f"   {h}+0   |        {hits_4q[h]:<13} |        {hits_8q_phys[h]:<12}")
    print("---------------------------------------------")
    
    # Calculate performance index (Weighted score based on prize distribution)
    # Higher hits (3+0, 4+0) are exponentially more valuable
    score_4q = hits_4q[1]*1 + hits_4q[2]*10 + hits_4q[3]*100 + hits_4q[4]*1000 + hits_4q[5]*50000
    score_8q = hits_8q_phys[1]*1 + hits_8q_phys[2]*10 + hits_8q_phys[3]*100 + hits_8q_phys[4]*1000 + hits_8q_phys[5]*50000
    print(f"Total Score (Weighted Hits):")
    print(f"- 4-qubit Standard Model   : {score_4q}")
    print(f"- 8-qubit Physical Model   : {score_8q}")
    improvement = ((score_8q - score_4q) / score_4q * 100) if score_4q > 0 else 0
    print(f"Relative Improvement       : {improvement:+.1f}%")
    print("=============================================")

if __name__ == "__main__":
    run_backtest()
