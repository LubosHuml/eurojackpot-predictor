import numpy as np
import train

def compute_top_k_accuracy(y_true, y_pred_probs, k):
    """
    Computes the fraction of actual drawn numbers that are captured
    within the top K highest predicted probabilities.
    """
    total_hits = 0
    total_drawn = 0
    
    for true, pred in zip(y_true, y_pred_probs):
        true_indices = np.where(true == 1.0)[0]
        # Get indices of top K probabilities
        pred_indices = np.argsort(pred)[-k:]
        
        hits = len(set(true_indices).intersection(set(pred_indices)))
        total_hits += hits
        total_drawn += len(true_indices)
        
    return total_hits / total_drawn if total_drawn > 0 else 0.0

def evaluate_model(model, X_num_raw, X_main, X_euro, y_sum_raw, y_counts_raw, y_main_logits, y_euro_logits, scaler_x, scaler_sum, scaler_counts, val_dates=None):
    """
    Evaluates the model on validation data, performing inverse scaling on predictions
    and computing regression/classification metrics.
    
    Returns:
        dict: dictionary of evaluation metrics
    """
    # Scale numerical inputs using pre-fitted scaler
    X_num_scaled = train.transform_3d(scaler_x, X_num_raw)
    
    # Predict
    pred_sum_scaled, pred_counts_scaled, pred_main_probs, pred_euro_probs = model.predict(
        [X_num_scaled, X_main, X_euro],
        verbose=0
    )
    
    # Inverse scaling for sums and counts
    pred_sum = scaler_sum.inverse_transform(pred_sum_scaled)
    pred_counts = scaler_counts.inverse_transform(pred_counts_scaled)
    
    # 1. Sum MSE
    sum_mse = np.mean((y_sum_raw - pred_sum) ** 2)
    sum_mae = np.mean(np.abs(y_sum_raw - pred_sum))
    
    # 2. Counts MAE
    counts_mae = np.mean(np.abs(y_counts_raw - pred_counts))
    
    # 3. Top-K accuracy for main numbers (K=5, 10, 15)
    main_top5 = compute_top_k_accuracy(y_main_logits, pred_main_probs, 5)
    main_top10 = compute_top_k_accuracy(y_main_logits, pred_main_probs, 10)
    main_top15 = compute_top_k_accuracy(y_main_logits, pred_main_probs, 15)
    
    # 4. Top-K accuracy for Euro numbers (K=2, 4)
    euro_top2 = compute_top_k_accuracy(y_euro_logits, pred_euro_probs, 2)
    euro_top4 = compute_top_k_accuracy(y_euro_logits, pred_euro_probs, 4)
    
    # Calculate historical accuracy details for each draw
    history_hits = []
    for i in range(len(y_main_logits)):
        actual_m_idx = np.where(y_main_logits[i] == 1.0)[0]
        actual_e_idx = np.where(y_euro_logits[i] == 1.0)[0]
        
        pred_m = pred_main_probs[i]
        pred_e = pred_euro_probs[i]
        
        top5_m = np.argsort(pred_m)[-5:]
        top10_m = np.argsort(pred_m)[-10:]
        top15_m = np.argsort(pred_m)[-15:]
        
        top2_e = np.argsort(pred_e)[-2:]
        top4_e = np.argsort(pred_e)[-4:]
        
        hits_top5 = len(set(actual_m_idx).intersection(set(top5_m)))
        hits_top10 = len(set(actual_m_idx).intersection(set(top10_m)))
        hits_top15 = len(set(actual_m_idx).intersection(set(top15_m)))
        
        hits_euro2 = len(set(actual_e_idx).intersection(set(top2_e)))
        hits_euro4 = len(set(actual_e_idx).intersection(set(top4_e)))
        
        history_hits.append({
            "date": val_dates[i] if val_dates else f"Draw {i+1}",
            "hits_top5": int(hits_top5),
            "hits_top10": int(hits_top10),
            "hits_top15": int(hits_top15),
            "hits_euro2": int(hits_euro2),
            "hits_euro4": int(hits_euro4)
        })
        
    return {
        "sum_mse": float(sum_mse),
        "sum_mae": float(sum_mae),
        "counts_mae": float(counts_mae),
        "main_top5": float(main_top5),
        "main_top10": float(main_top10),
        "main_top15": float(main_top15),
        "euro_top2": float(euro_top2),
        "euro_top4": float(euro_top4),
        "history_hits": history_hits
    }

def run_backtest(model, data_dict, scaler_x, scaler_sum, scaler_counts, val_split=50):
    """
    Runs evaluation on the reserved validation set (the last val_split draws).
    """
    # Slice the validation data from the end
    X_num_val = data_dict['X_num'][-val_split:]
    X_main_val = data_dict['X_main'][-val_split:]
    X_euro_val = data_dict['X_euro'][-val_split:]
    
    y_sum_val = data_dict['y_sum'][-val_split:]
    y_counts_val = data_dict['y_counts'][-val_split:]
    y_main_logits_val = data_dict['y_main_logits'][-val_split:]
    y_euro_logits_val = data_dict['y_euro_logits'][-val_split:]
    
    val_dates = data_dict.get('dates', [])[-val_split:]
    
    print(f"Running backtest evaluation on the last {val_split} draws...")
    metrics = evaluate_model(
        model, X_num_val, X_main_val, X_euro_val,
        y_sum_val, y_counts_val, y_main_logits_val, y_euro_logits_val,
        scaler_x, scaler_sum, scaler_counts, val_dates=val_dates
    )
    return metrics
