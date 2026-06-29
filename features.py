import numpy as np
import pandas as pd
from collections import Counter

def compute_draw_features(draws, vwap_lookback=50):
    """
    Computes statistical and financial features for a list of draws.
    
    Parameters:
        draws (list of dicts): list of draws returned from database.get_all_draws()
        vwap_lookback (int): sliding window size to compute number frequency for VWAP
        
    Returns:
        pd.DataFrame: DataFrame containing all computed features and the raw draw values.
    """
    df_list = []
    
    # Store history for running calculations
    main_history = []
    
    for i, draw in enumerate(draws):
        date = draw['date']
        m = sorted(draw['main_nums'])
        e = sorted(draw['euro_nums'])
        
        # 1. Statistical Aggregates
        mean_val = np.mean(m)
        std_val = np.std(m)
        median_val = np.median(m)
        sum_val = sum(m)
        
        # Product of differences between adjacent sorted main numbers
        prod_diff = (m[1] - m[0]) * (m[2] - m[1]) * (m[3] - m[2]) * (m[4] - m[3])
        
        # 2. Combinatorial Distributions
        even_count = sum(1 for x in m if x % 2 == 0)
        odd_count = 5 - even_count
        low_count = sum(1 for x in m if 1 <= x <= 25)
        high_count = 5 - low_count
        
        # 3. Central Pivot Range (CPR)
        high_val = m[4]
        low_val = m[0]
        close_val = mean_val
        
        pp = (high_val + low_val + close_val) / 3.0
        bc = (high_val + low_val) / 2.0
        tc = (2.0 * pp) - bc
        
        # 4. Volume Weighted Average Price (VWAP)
        # Compute number frequency over sliding lookback window in the past
        past_draws = main_history[-vwap_lookback:] if i > 0 else []
        if past_draws:
            # Flatten the past draws list
            flat_past = [num for sublist in past_draws for num in sublist]
            counts = Counter(flat_past)
            # Default frequency to 1 if not appeared
            freqs = [counts.get(x, 1) for x in m]
        else:
            freqs = [1] * 5
            
        vwap = sum(x * f for x, f in zip(m, freqs)) / sum(freqs)
        
        # Add to history
        main_history.append(m)
        
        df_list.append({
            'date': date,
            'num1': m[0], 'num2': m[1], 'num3': m[2], 'num4': m[3], 'num5': m[4],
            'euro1': e[0], 'euro2': e[1],
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'sum': sum_val,
            'product_diff': prod_diff,
            'even_count': even_count,
            'odd_count': odd_count,
            'low_count': low_count,
            'high_count': high_count,
            'cpr_pp': pp,
            'cpr_bc': bc,
            'cpr_tc': tc,
            'vwap': vwap
        })
        
    return pd.DataFrame(df_list)

def generate_sequences(df, window_size=10):
    """
    Generates training sequences and targets from the features DataFrame.
    
    Parameters:
        df (pd.DataFrame): features DataFrame
        window_size (int): sequence lookback window w
        
    Returns:
        dict: containing inputs and target arrays
    """
    feature_cols = [
        'mean', 'std', 'median', 'sum', 'product_diff',
        'even_count', 'odd_count', 'low_count', 'high_count',
        'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
    ]
    
    num_features = df[feature_cols].values
    main_nums = df[['num1', 'num2', 'num3', 'num4', 'num5']].values
    euro_nums = df[['euro1', 'euro2']].values
    
    X_num = []
    X_main = []
    X_euro = []
    
    y_sum = []
    y_counts = []
    y_main_logits = []
    y_euro_logits = []
    
    for i in range(len(df) - window_size):
        # Input sequence (t-w to t-1)
        X_num.append(num_features[i : i + window_size])
        X_main.append(main_nums[i : i + window_size])
        X_euro.append(euro_nums[i : i + window_size])
        
        # Target (t + window_size)
        target_idx = i + window_size
        target_main = main_nums[target_idx]
        target_euro = euro_nums[target_idx]
        
        # Sum target S_t+1
        y_sum.append(df.iloc[target_idx]['sum'])
        
        # Counts target E_t+1: [Even, Odd, Low, High]
        y_counts.append([
            df.iloc[target_idx]['even_count'],
            df.iloc[target_idx]['odd_count'],
            df.iloc[target_idx]['low_count'],
            df.iloc[target_idx]['high_count']
        ])
        
        # Logits target for main numbers (0-indexed representation of 1-50: size 50)
        main_binary = np.zeros(50)
        for n in target_main:
            main_binary[int(n) - 1] = 1.0
        y_main_logits.append(main_binary)
        
        # Logits target for euro numbers (0-indexed representation of 1-12: size 12)
        euro_binary = np.zeros(12)
        for n in target_euro:
            euro_binary[int(n) - 1] = 1.0
        y_euro_logits.append(euro_binary)
        
    return {
        'X_num': np.array(X_num, dtype=np.float32),
        'X_main': np.array(X_main, dtype=np.int32),
        'X_euro': np.array(X_euro, dtype=np.int32),
        'y_sum': np.array(y_sum, dtype=np.float32).reshape(-1, 1),
        'y_counts': np.array(y_counts, dtype=np.float32),
        'y_main_logits': np.array(y_main_logits, dtype=np.float32),
        'y_euro_logits': np.array(y_euro_logits, dtype=np.float32),
        'dates': df.iloc[window_size:]['date'].values.tolist()
    }
