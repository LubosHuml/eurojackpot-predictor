import os
import sys
import numpy as np

# Add project path dynamically to sys.path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import crypto.quantum_lotto as quantum_lotto

def run_historical_validation(game="eurojackpot", val_count=50):
    data = quantum_lotto.fetch_lotto_data(game)
    if not data:
        print(f"No database found for {game}.")
        return
        
    main_history, euro_history = data
    if len(main_history) < val_count + 10:
        print(f"Not enough draws to run {val_count} draw validation.")
        return
        
    pool_max = 50 if game == "eurojackpot" else 49
    n_main = 5 if game == "eurojackpot" else 6
    
    # Pre-generate inputs for QRC
    inputs = []
    for i in range(5, len(main_history)):
        past_draws = main_history[i-5:i]
        flat_past = np.array(past_draws).flatten()
        mean_val = np.mean(flat_past)
        sum_val = np.sum(past_draws[-1])
        even_count = sum(1 for x in past_draws[-1] if x % 2 == 0)
        high_count = sum(1 for x in past_draws[-1] if x > (pool_max / 2))
        
        f1 = (mean_val - (pool_max/2)) / (pool_max/2)
        f2 = (sum_val - (pool_max*2.5)) / (pool_max*2.5)
        f3 = (even_count - 2.5) / 2.5
        f4 = (high_count - 2.5) / 2.5
        inputs.append([f1, f2, f3, f4])
    inputs = np.array(inputs)
    
    # Run QRC reservoir forward for all inputs to get states
    qrc = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    qrc_features = []
    for inp in inputs:
        states = qrc.step(inp)
        qrc_features.append(states)
    qrc_features = np.array(qrc_features)
    
    # Sparser Random projections (ShRIMP)
    shrimp_feats, W_sparse = quantum_lotto.shrimp_random_features(qrc_features, num_features=50, sparsity=0.7)
    
    # We validate over the last `val_count` draws
    # July 2026 starts at the end of the history
    start_idx = len(main_history) - val_count
    
    hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    top10_hits = 0
    top15_hits = 0
    total_nums_tested = 0
    
    # Detailed log of last 10 draws
    last_10_details = []
    
    for idx in range(start_idx, len(main_history)):
        # Train Ridge readout layer using only data up to draw idx - 1 (Walk-Forward validation!)
        train_end = idx - 5 # offset because history inputs start at index 5
        
        Z_train = shrimp_feats[:train_end]
        targets_train = np.zeros((len(Z_train), pool_max))
        for t_idx in range(len(Z_train)):
            next_draw = main_history[5 + t_idx]
            for num in next_draw:
                if 1 <= num <= pool_max:
                    targets_train[t_idx, num - 1] = 1.0
                    
        # Solve Ridge regression
        Z_T_Z = Z_train.T @ Z_train
        alpha = 1.0
        ridge_inv = np.linalg.inv(Z_T_Z + alpha * np.eye(Z_train.shape[1]))
        W_readout = ridge_inv @ Z_train.T @ targets_train
        
        # Predict probability amplitudes for draw `idx`
        # corresponding to Z[train_end]
        last_reservoir_state = shrimp_feats[train_end:train_end+1]
        amplitudes = last_reservoir_state @ W_readout
        amplitudes = np.clip(amplitudes[0], 0.0, 1.0)
        probabilities = amplitudes / np.sum(amplitudes)
        
        # Collapse wave function to select numbers
        selected_main = []
        temp_probs = probabilities.copy()
        for _ in range(n_main):
            temp_probs /= np.sum(temp_probs)
            num = np.random.choice(range(1, pool_max + 1), p=temp_probs)
            selected_main.append(int(num))
            temp_probs[num - 1] = 0.0
        selected_main.sort()
        
        # Check actual results
        actual_draw = main_history[idx]
        matches = list(set(selected_main) & set(actual_draw))
        hits_distribution[len(matches)] += 1
        
        # Top 10 / 15 accuracy (is the drawn number in our top quantum amplitudes?)
        top_10_indices = np.argsort(probabilities)[-10:] + 1
        top_15_indices = np.argsort(probabilities)[-15:] + 1
        
        matches_top10 = list(set(top_10_indices) & set(actual_draw))
        matches_top15 = list(set(top_15_indices) & set(actual_draw))
        
        top10_hits += len(matches_top10)
        top15_hits += len(matches_top15)
        total_nums_tested += len(actual_draw)
        
        # Log details for last 10
        if idx >= len(main_history) - 10:
            last_10_details.append({
                "draw_index": idx,
                "predicted": selected_main,
                "actual": actual_draw,
                "matches": matches
            })
            
    print(f"\n===========================================================")
    print(f"      QUANTUM QRC + ShRIMP BACKTEST: {game.upper()}")
    print(f"===========================================================")
    print(f"Validation Horizon:       {val_count} draws (Walk-Forward)")
    print(f"Hilbert State Space:      16 Dimensions (4 Qubits)")
    print(f"Readout Sparsity:         70.0% (ShRIMP-pruned)")
    print("-----------------------------------------------------------")
    print("MATCHES DISTRIBUTION:")
    for k, v in hits_distribution.items():
        if k <= n_main:
            pct = (v / val_count) * 100
            print(f" * {k} Matches: {v} times ({pct:.1f}%)")
    print("-----------------------------------------------------------")
    print(f"Top 10 Quantum Coverage:  {(top10_hits / total_nums_tested)*100:.2f}% (Average of {top10_hits/val_count:.2f} numbers per draw)")
    print(f"Top 15 Quantum Coverage:  {(top15_hits / total_nums_tested)*100:.2f}% (Average of {top15_hits/val_count:.2f} numbers per draw)")
    print("===========================================================")
    
    print(f"\nDETAILED ANALYSIS OF THE LAST 10 DRAWS ({game.upper()}):")
    print("-----------------------------------------------------------")
    for d in last_10_details:
        print(f"Draw #{d['draw_index']}:")
        print(f"  * Quantum collapsed: {d['predicted']}")
        print(f"  * Actual drawn:      {d['actual']}")
        print(f"  * Matched:           {d['matches']} ({len(d['matches'])} hits)")
    print("-----------------------------------------------------------")

if __name__ == "__main__":
    run_historical_validation("eurojackpot", 50)
    run_historical_validation("sportka", 50)
