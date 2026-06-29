import os
import sys
import numpy as np
import tensorflow as tf
import joblib
import hashlib
import random

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
sys.path.append(project_path)

import sportka_database
import sportka_features
import sportka_train
import app

# SPORTKA PAYOUT ESTIMATES in CZK
# 1. pořadí (6): ~15,000,000 CZK (variable, average used)
# 2. pořadí (5+1): ~800,000 CZK (variable)
# 3. pořadí (5): ~25,000 CZK
# 4. pořadí (4): ~600 CZK
# 5. pořadí (3): 115 CZK (fixed)
def get_sportka_payout(matched_count, has_supplementary):
    if matched_count == 6:
        return 15000000.0, "1. pořadí (6)"
    elif matched_count == 5 and has_supplementary:
        return 800000.0, "2. pořadí (5+1)"
    elif matched_count == 5:
        return 25000.0, "3. pořadí (5)"
    elif matched_count == 4:
        return 600.0, "4. pořadí (4)"
    elif matched_count == 3:
        return 115.0, "5. pořadí (3)"
    return 0.0, ""

def main():
    sportka_database.init_db()
    draws = sportka_database.get_all_draws()
    if len(draws) < 60:
        print("Not enough draws.")
        return

    # Load model and scalers
    model = tf.keras.models.load_model(os.path.join(project_path, "sportka_lstm_model.keras"))
    scaler_x = joblib.load(os.path.join(project_path, "sportka_scaler_x.joblib"))
    scaler_sum = joblib.load(os.path.join(project_path, "sportka_scaler_sum.joblib"))
    scaler_counts = joblib.load(os.path.join(project_path, "sportka_scaler_counts.joblib"))

    # Compute features
    df_features = sportka_features.compute_draw_features(draws)
    data_dict = sportka_features.generate_sequences(df_features, window_size=10)
    
    val_split = 50
    X_num_val = data_dict['X_num'][-val_split:]
    X_main_val = data_dict['X_main'][-val_split:]
    
    y_sum_val = data_dict['y_sum'][-val_split:]
    y_counts_val = data_dict['y_counts'][-val_split:]
    y_main_logits_val = data_dict['y_main_logits'][-val_split:]
    
    val_dates = data_dict['dates'][-val_split:]

    # Scale inputs
    X_num_scaled = sportka_train.transform_3d(scaler_x, X_num_val)
    
    # Predict
    pred_sum_scaled, pred_counts_scaled, pred_main_probs = model.predict(
        [X_num_scaled, X_main_val],
        verbose=0
    )

    ticket_cost = 180.0 # 6 columns * 30 CZK
    total_cost = val_split * ticket_cost
    
    total_std_winnings = 0.0
    total_sys_winnings = 0.0
    
    sys_draw_results = []
    
    # 8-number 3-if-3 coverage wheel template
    wheel_indices = [
        [1, 2, 3, 4, 6, 7],
        [0, 1, 3, 4, 5, 6],
        [0, 1, 2, 3, 5, 7],
        [0, 2, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 3, 4, 6, 7]
    ]
    
    for i in range(val_split):
        draw_date = val_dates[i]
        
        # Get actual drawn numbers and supplementary
        actual_main_idx = np.where(y_main_logits_val[i] == 1.0)[0]
        actual_main = [int(x + 1) for x in actual_main_idx]
        
        # Supplementary number is not directly in logits (it was supplementary key in draw).
        # Let's query it from draws list
        # Since draws are sorted chronologically, the val_split aligns with the last 50 draws:
        draw_idx = len(draws) - val_split + i
        actual_supp = draws[draw_idx]["supplementary"]
        
        # Probs
        next_main_probs = pred_main_probs[i]
        
        # --- 1. SIMULATE STANDARD TICKET ---
        # Seed generator deterministically
        seed_src = draw_date
        seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        
        pred_sum_val = scaler_sum.inverse_transform(pred_sum_scaled)[i, 0]
        pred_counts_val = scaler_counts.inverse_transform(pred_counts_scaled)[i]
        
        bets_c = app.generate_sportka_bets(next_main_probs, pred_sum_val, pred_counts_val, temperature=0.2, count=2)
        bets_b = app.generate_sportka_bets(next_main_probs, pred_sum_val, pred_counts_val, temperature=1.0, count=2)
        bets_u = app.generate_sportka_bets(next_main_probs, pred_sum_val, pred_counts_val, temperature=2.0, count=2)
        
        raw_std_bets = [bets_c[0], bets_c[1], bets_b[0], bets_b[1], bets_u[0], bets_u[1]]
        for comb in raw_std_bets:
            matched = list(set(comb) & set(actual_main))
            has_supp = 1 if (actual_supp in comb) else 0
            payout, _ = get_sportka_payout(len(matched), has_supp)
            total_std_winnings += payout
            
        # --- 2. SIMULATE SYSTEM WHEEL TICKET ---
        top8_idx = np.argsort(next_main_probs)[-8:]
        top8_nums = sorted([int(x + 1) for x in top8_idx])
        
        sys_draw_winnings = 0.0
        sys_hits_list = []
        
        for idx, row in enumerate(wheel_indices):
            comb = sorted([top8_nums[k] for k in row])
            matched = list(set(comb) & set(actual_main))
            has_supp = 1 if (actual_supp in comb) else 0
            
            payout, label = get_sportka_payout(len(matched), has_supp)
            sys_draw_winnings += payout
            
            if payout > 0:
                sys_hits_list.append(f"Row #{idx+1} ({label}): +{payout:,.0f} CZK (Matched: {matched})")
                
        total_sys_winnings += sys_draw_winnings
        
        if sys_draw_winnings > 0:
            sys_draw_results.append({
                "date": draw_date,
                "actual_main": actual_main,
                "actual_supp": actual_supp,
                "winnings": sys_draw_winnings,
                "hits": sys_hits_list
            })
            
    std_roi = (total_std_winnings / total_cost) * 100
    sys_roi = (total_sys_winnings / total_cost) * 100
    
    print("\n===========================================================")
    print("      SPORTKA BACKTEST COMPARISON: STANDARD VS SYSTEM WHEEL")
    print("===========================================================")
    print(f"Total Draws Simulated:     {val_split} (approx. 16 weeks)")
    print(f"Total Invested per Strategy: {total_cost:,.0f} CZK")
    print("-----------------------------------------------------------")
    print(" 1. STANDARD PORTFOLIO STRATEGY:")
    print(f"    Total Winnings:        {total_std_winnings:,.0f} CZK")
    print(f"    Net Balance:           {total_std_winnings - total_cost:+,.0f} CZK")
    print(f"    RETURN ON INVESTMENT:  {std_roi:.2f}%")
    print("-----------------------------------------------------------")
    print(" 2. SYSTEM WHEEL STRATEGY (8 numbers, 3-if-3 cover):")
    print(f"    Total Winnings:        {total_sys_winnings:,.0f} CZK")
    print(f"    Net Balance:           {total_sys_winnings - total_cost:+,.0f} CZK")
    print(f"    RETURN ON INVESTMENT:  {sys_roi:.2f}%")
    print("===========================================================")
    
    print("\nDetailed list of Sportka System Wheel wins (Draws with payouts):")
    for res in sys_draw_results:
        print(f"Draw Date {res['date']}: Actual {res['actual_main']} + Supp {res['actual_supp']}")
        for hit in res["hits"]:
            print(f"  - {hit}")
        print(f"  Total draw payout: {res['winnings']:,.0f} CZK")

if __name__ == "__main__":
    main()
