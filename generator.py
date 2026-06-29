import numpy as np

def softmax(x, temperature=1.0):
    """
    Computes temperature-controlled softmax.
    """
    # Prevent division by zero
    temp = max(temperature, 1e-6)
    x = np.array(x) / temp
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def sample_combination(main_probs, euro_probs):
    """
    Samples a single Eurojackpot combination based on probabilities.
    
    Returns:
        tuple: (sorted_main_nums, sorted_euro_nums)
    """
    # Sample main numbers (1 to 50) without replacement
    main_choices = np.arange(1, 51)
    main_nums = np.random.choice(main_choices, size=5, replace=False, p=main_probs)
    
    # Sample Euro numbers (1 to 12) without replacement
    euro_choices = np.arange(1, 13)
    euro_nums = np.random.choice(euro_choices, size=2, replace=False, p=euro_probs)
    
    return sorted(list(main_nums)), sorted(list(euro_nums))

def generate_bets(main_logits, euro_logits, pred_sum, pred_counts, temperature=1.0, count=5, max_attempts=10000):
    """
    Generates a set of filtered bet combinations using Monte Carlo sampling.
    
    Parameters:
        main_logits (np.array): shape (50,) raw logits from model
        euro_logits (np.array): shape (12,) raw logits from model
        pred_sum (float): predicted sum of main numbers
        pred_counts (list/np.array): predicted [Even, Odd, Low, High] counts
        temperature (float): softmax temperature (e.g. 0.2, 1.0, 2.0)
        count (int): number of combinations to generate
        max_attempts (int): max sampling attempts before relaxing filters
        
    Returns:
        list of tuples: [(main_nums, euro_nums), ...]
    """
    # Convert logits to probabilities
    main_probs = softmax(main_logits, temperature)
    euro_probs = softmax(euro_logits, temperature)
    
    # Target counts from model (clamped to valid lottery bounds)
    target_even = int(np.clip(np.round(pred_counts[0]), 0, 5))
    target_low = int(np.clip(np.round(pred_counts[2]), 0, 5))
    
    bets = []
    attempts = 0
    relaxed = False
    
    while len(bets) < count and attempts < max_attempts:
        attempts += 1
        
        main_nums, euro_nums = sample_combination(main_probs, euro_probs)
        
        # Verify if combination already generated
        if (main_nums, euro_nums) in bets:
            continue
            
        # Compute stats for candidate
        cand_sum = sum(main_nums)
        cand_even = sum(1 for x in main_nums if x % 2 == 0)
        cand_low = sum(1 for x in main_nums if 1 <= x <= 25)
        
        # Filter 1: Sum prediction interval (predicted sum +/- 15)
        sum_tolerance = 15 if not relaxed else 25
        sum_ok = abs(cand_sum - pred_sum) <= sum_tolerance
        
        # Filter 2: Parity and Size distributions
        if not relaxed:
            even_ok = cand_even == target_even
            low_ok = cand_low == target_low
        else:
            # Relaxed filter: allow +/- 1 difference from targets
            even_ok = abs(cand_even - target_even) <= 1
            low_ok = abs(cand_low - target_low) <= 1
            
        if sum_ok and even_ok and low_ok:
            bets.append((main_nums, euro_nums))
            
        # Relax constraints if we are struggling to find matches
        if attempts == max_attempts // 2:
            print("Warning: Sampling limit reached. Relaxing generator constraints...")
            relaxed = True
            
    # If we still did not generate enough, fill the rest without strict filtering
    while len(bets) < count:
        main_nums, euro_nums = sample_combination(main_probs, euro_probs)
        if (main_nums, euro_nums) not in bets:
            bets.append((main_nums, euro_nums))
            
    return bets
