import numpy as np
import pandas as pd
import sportka_database

def compute_draw_features(draws):
    """
    Computes statistical features for each Sportka draw.
    """
    df = pd.DataFrame(draws)
    
    # Extract numbers as columns
    df['num1'] = df['numbers'].apply(lambda x: x[0])
    df['num2'] = df['numbers'].apply(lambda x: x[1])
    df['num3'] = df['numbers'].apply(lambda x: x[2])
    df['num4'] = df['numbers'].apply(lambda x: x[3])
    df['num5'] = df['numbers'].apply(lambda x: x[4])
    df['num6'] = df['numbers'].apply(lambda x: x[5])
    
    nums_matrix = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
    
    # 1. Simple Stats
    df['mean'] = np.mean(nums_matrix, axis=1)
    df['std'] = np.std(nums_matrix, axis=1)
    df['median'] = np.median(nums_matrix, axis=1)
    df['sum'] = np.sum(nums_matrix, axis=1)
    
    # 2. Difference Products
    diffs = np.diff(nums_matrix, axis=1)
    df['product_diff'] = np.prod(diffs, axis=1) / 1e6
    
    # 3. Even/Odd
    df['even_count'] = np.sum(nums_matrix % 2 == 0, axis=1)
    df['odd_count'] = 6 - df['even_count']
    
    # 4. Low/High (1-24 vs 25-49)
    df['low_count'] = np.sum((1 <= nums_matrix) & (nums_matrix <= 24), axis=1)
    df['high_count'] = 6 - df['low_count']
    
    # 5. Consecutive Pairs (cpr)
    cpr_pp = []
    cpr_bc = []
    cpr_tc = []
    
    for row in nums_matrix:
        diff = np.diff(row)
        c_pairs = np.sum(diff == 1)
        cpr_pp.append(c_pairs)
        cpr_bc.append(1 if (row[0] == 1 or row[-1] == 49) else 0)
        cpr_tc.append(1 if (c_pairs >= 2) else 0)
        
    df['cpr_pp'] = cpr_pp
    df['cpr_bc'] = cpr_bc
    df['cpr_tc'] = cpr_tc
    
    # 6. VWAP-like index
    vwap = []
    for i in range(len(df)):
        if i == 0:
            vwap.append(df.iloc[0]['mean'])
        else:
            window = nums_matrix[max(0, i-9) : i+1]
            flat = window.flatten()
            unique, counts = np.unique(flat, return_counts=True)
            freq_map = dict(zip(unique, counts))
            
            curr_draw = nums_matrix[i]
            val_sum = sum(x * freq_map.get(x, 1) for x in curr_draw)
            weight_sum = sum(freq_map.get(x, 1) for x in curr_draw)
            vwap.append(val_sum / weight_sum if weight_sum > 0 else df.iloc[i]['mean'])
            
    df['vwap'] = vwap
    return df

def generate_sequences(df, window_size=10):
    """
    Generates training sequences and targets from the features DataFrame.
    """
    feature_cols = [
        'mean', 'std', 'median', 'sum', 'product_diff',
        'even_count', 'odd_count', 'low_count', 'high_count',
        'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
    ]
    
    num_features = df[feature_cols].values
    main_nums = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
    
    X_num = []
    X_main = []
    
    y_sum = []
    y_counts = []
    y_main_logits = []
    
    for i in range(len(df) - window_size):
        X_num.append(num_features[i : i + window_size])
        X_main.append(main_nums[i : i + window_size])
        
        target_idx = i + window_size
        target_main = main_nums[target_idx]
        
        y_sum.append(df.iloc[target_idx]['sum'])
        y_counts.append([
            df.iloc[target_idx]['even_count'],
            df.iloc[target_idx]['odd_count'],
            df.iloc[target_idx]['low_count'],
            df.iloc[target_idx]['high_count']
        ])
        
        # Logits target for main numbers (0-indexed representation of 1-49: size 49)
        main_binary = np.zeros(49)
        for n in target_main:
            main_binary[int(n) - 1] = 1.0
        y_main_logits.append(main_binary)
        
    return {
        'X_num': np.array(X_num, dtype=np.float32),
        'X_main': np.array(X_main, dtype=np.int32),
        'y_sum': np.array(y_sum, dtype=np.float32).reshape(-1, 1),
        'y_counts': np.array(y_counts, dtype=np.float32),
        'y_main_logits': np.array(y_main_logits, dtype=np.float32),
        'dates': df.iloc[window_size:]['date'].values.tolist(),
        'draw_nums': df.iloc[window_size:]['draw_num'].values.tolist()
    }
