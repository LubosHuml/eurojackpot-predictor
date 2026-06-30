import os
import sys
import numpy as np
import tensorflow as tf
import joblib
import sqlite3

# Resolve project path dynamically
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import database
import features
import train
import generator
import crypto.quantum_lotto as quantum_lotto

def run_hybrid_predictions(count=6):
    """
    Combines LSTM Deep Learning predictions with Quantum Reservoir Computing
    to generate optimal, noise-filtered Eurojackpot combinations.
    """
    db_path = os.path.join(project_path, "eurojackpot.db")
    model_path = os.path.join(project_path, "eurojackpot_lstm_model.keras")
    if not os.path.exists(db_path) or not os.path.exists(model_path):
        return {"error": f"Eurojackpot database or LSTM model not found. db_path={db_path} (exists: {os.path.exists(db_path)}), model_path={model_path} (exists: {os.path.exists(model_path)})"}
        
    # 1. Fetch latest draws
    draws = database.get_all_draws()
    if len(draws) < 20:
        return {"error": "Not enough draws in database."}
        
    # 2. LSTM prediction pipeline
    # Load model and scalers
    model = tf.keras.models.load_model(model_path)
    scaler_x = joblib.load(os.path.join(project_path, "scaler_x.joblib"))
    scaler_sum = joblib.load(os.path.join(project_path, "scaler_sum.joblib"))
    scaler_counts = joblib.load(os.path.join(project_path, "scaler_counts.joblib"))
    
    # Compute features and format input
    df_features = features.compute_draw_features(draws)
    window_size = 10
    feature_cols = [
        'mean', 'std', 'median', 'sum', 'product_diff',
        'even_count', 'odd_count', 'low_count', 'high_count',
        'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
    ]
    num_features = df_features[feature_cols].values
    main_nums = df_features[['num1', 'num2', 'num3', 'num4', 'num5']].values
    euro_nums = df_features[['euro1', 'euro2']].values
    
    X_num_last = np.expand_dims(num_features[-window_size:], axis=0).astype(np.float32)
    X_main_last = np.expand_dims(main_nums[-window_size:], axis=0).astype(np.int32)
    X_euro_last = np.expand_dims(euro_nums[-window_size:], axis=0).astype(np.int32)
    
    X_num_scaled = train.transform_3d(scaler_x, X_num_last)
    
    # Predict with LSTM
    pred_sum_scaled, pred_counts_scaled, pred_main_probs, pred_euro_probs = model.predict(
        [X_num_scaled, X_main_last, X_euro_last],
        verbose=0
    )
    
    lstm_sum = float(scaler_sum.inverse_transform(pred_sum_scaled)[0, 0])
    lstm_counts = scaler_counts.inverse_transform(pred_counts_scaled)[0]
    lstm_main_probs = pred_main_probs[0]
    lstm_euro_probs = pred_euro_probs[0]
    
    # 3. Quantum Reservoir Computing pipeline
    # Construct 8 QRC features for 8-qubit simulation
    inputs = []
    for i in range(5, len(draws)):
        past_draws = [d['main_nums'] for d in draws[i-5:i]]
        flat_past = np.array(past_draws).flatten()
        mean_val = np.mean(flat_past)
        std_val = np.std(flat_past)
        median_val = np.median(flat_past)
        sum_val = np.sum(past_draws[-1])
        even_count = sum(1 for x in past_draws[-1] if x % 2 == 0)
        high_count = sum(1 for x in past_draws[-1] if x > 25)
        low_count = sum(1 for x in past_draws[-1] if x <= 25)
        odd_count = sum(1 for x in past_draws[-1] if x % 2 != 0)
        
        f1 = (mean_val - 25.0) / 25.0
        f2 = (sum_val - 125.0) / 125.0
        f3 = (even_count - 2.5) / 2.5
        f4 = (high_count - 2.5) / 2.5
        f5 = (std_val - 14.4) / 14.4
        f6 = (median_val - 25.0) / 25.0
        f7 = (low_count - 2.5) / 2.5
        f8 = (odd_count - 2.5) / 2.5
        inputs.append([f1, f2, f3, f4, f5, f6, f7, f8])
    inputs = np.array(inputs)
    
    qrc = quantum_lotto.QuantumReservoir(n_qubits=8, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    qrc_features = []
    for inp in inputs:
        states = qrc.step(inp)
        qrc_features.append(states)
    qrc_features = np.array(qrc_features)
    
    shrimp_feats, W_sparse = quantum_lotto.shrimp_random_features(qrc_features, num_features=50, sparsity=0.7)
    
    # Train Ridge readouts
    Z = shrimp_feats
    Z_T_Z = Z.T @ Z
    alpha = 1.0
    ridge_inv = np.linalg.inv(Z_T_Z + alpha * np.eye(Z.shape[1]))
    
    # Main numbers QRC readout
    targets_main = np.zeros((len(Z), 50))
    for idx, next_draw in enumerate(draws[5:]):
        for num in next_draw['main_nums']:
            if 1 <= num <= 50:
                targets_main[idx, num - 1] = 1.0
    W_readout = ridge_inv @ Z.T @ targets_main
    last_state = Z[-1:]
    qrc_main_amps = np.clip((last_state @ W_readout)[0], 0.0, 1.0)
    qrc_main_probs = qrc_main_amps / np.sum(qrc_main_amps)
    
    # Euro numbers QRC readout
    targets_euro = np.zeros((len(Z), 12))
    for idx, next_draw in enumerate(draws[5:]):
        for num in next_draw['euro_nums']:
            if 1 <= num <= 12:
                targets_euro[idx, num - 1] = 1.0
    W_readout_euro = ridge_inv @ Z.T @ targets_euro
    qrc_euro_amps = np.clip((last_state @ W_readout_euro)[0], 0.0, 1.0)
    qrc_euro_probs = qrc_euro_amps / np.sum(qrc_euro_amps)
    
    # 4. FUSION: Combine LSTM and QRC probability waves
    # We assign 55% weight to LSTM and 45% weight to QRC
    hybrid_main_probs = 0.55 * lstm_main_probs + 0.45 * qrc_main_probs
    hybrid_main_probs /= np.sum(hybrid_main_probs)
    
    hybrid_euro_probs = 0.55 * lstm_euro_probs + 0.45 * qrc_euro_probs
    hybrid_euro_probs /= np.sum(hybrid_euro_probs)
    
    # 5. Monte Carlo Generation with structural filters (using softmax-transformed logs for generator)
    pseudo_main_logits = np.log(hybrid_main_probs + 1e-12)
    pseudo_euro_logits = np.log(hybrid_euro_probs + 1e-12)
    
    bets = generator.generate_bets(
        main_logits=pseudo_main_logits,
        euro_logits=pseudo_euro_logits,
        pred_sum=lstm_sum,
        pred_counts=lstm_counts,
        temperature=0.8,
        count=count
    )
    
    # Format results
    formatted_bets = []
    for idx, bet in enumerate(bets):
        main_nums, euro_nums = bet
        
        # Calculate pseudo-confidence score
        conf_sum = sum(hybrid_main_probs[num - 1] for num in main_nums)
        conf_euro = sum(hybrid_euro_probs[num - 1] for num in euro_nums)
        total_conf = (conf_sum / 5.0) * 0.7 + (conf_euro / 2.0) * 0.3
        
        formatted_bets.append({
            "row_id": idx + 1,
            "main_nums": [int(x) for x in main_nums],
            "euro_nums": [int(x) for x in euro_nums],
            "confidence": round(float(total_conf) * 100, 1)
        })
        
    return {
        "success": True,
        "draw_date": draws[-1]['date'],
        "bets": formatted_bets,
        "consensus_purity": float(np.real(np.trace(qrc.rho @ qrc.rho)))
    }

if __name__ == "__main__":
    print("Testing Hybrid Quantum-LSTM Generator:")
    res = run_hybrid_predictions(6)
    print(res)
