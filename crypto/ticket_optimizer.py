import numpy as np
import random
import os

class TicketOptimizer:
    def __init__(self, main_probs, euro_probs, pred_sum, pred_counts):
        self.main_probs = np.array(main_probs)
        self.euro_probs = np.array(euro_probs)
        self.pred_sum = pred_sum
        
        # Predicted counts: [Even, Odd, Low, High]
        self.target_even = int(np.clip(np.round(pred_counts[0]), 0, 5))
        self.target_low = int(np.clip(np.round(pred_counts[2]), 0, 5))
        
        # Load Euro-Main lift correlation matrix if available
        self.lift = np.ones((12, 50))
        try:
            # Try to resolve path relative to this script
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            lift_path = os.path.join(curr_dir, "euro_main_lift.npy")
            if os.path.exists(lift_path):
                self.lift = np.load(lift_path)
        except Exception as e:
            print(f"Error loading lift matrix: {e}")
        
    def calculate_energy(self, tickets, count):
        """
        Calculates the energy (loss) of a set of tickets.
        We want to minimize this energy.
        """
        energy = 0.0
        epsilon = 1e-12
        
        # 1. Probability component (maximize prob of selected numbers)
        for main_nums, euro_nums in tickets:
            for n in main_nums:
                energy -= np.log(self.main_probs[n - 1] + epsilon) * 2.0
            for e in euro_nums:
                energy -= np.log(self.euro_probs[e - 1] + epsilon) * 3.0
                
            # 2. Entanglement/Correlation component: reward main numbers that are highly correlated with this ticket's Euro numbers
            for m in main_nums:
                for e in euro_nums:
                    lift_val = self.lift[e - 1, m - 1]
                    # We subtract log of lift. If lift > 1, this decreases energy (rewards correlation)
                    energy -= np.log(lift_val + epsilon) * 4.0
                
        # 3. Overlap penalty (minimize duplicate main numbers to maximize coverage across the portfolio)
        main_counts = np.zeros(50)
        euro_counts = np.zeros(12)
        for main_nums, euro_nums in tickets:
            for n in main_nums:
                main_counts[n - 1] += 1
            for e in euro_nums:
                euro_counts[e - 1] += 1
                
        # Quadratic penalty for multiple occurrences of the same main number
        energy += np.sum(main_counts ** 2) * 5.0
        
        # If count is exactly 6, we highly penalize duplicate Euro numbers to enforce 100% unique coverage
        if count == 6:
            energy += np.sum(euro_counts ** 2) * 250.0
        else:
            energy += np.sum(euro_counts ** 2) * 10.0
        
        # 4. Structural constraint penalties for each ticket
        for main_nums, euro_nums in tickets:
            cand_sum = sum(main_nums)
            cand_even = sum(1 for x in main_nums if x % 2 == 0)
            cand_low = sum(1 for x in main_nums if 1 <= x <= 25)
            
            # Sum penalty: quadratic penalty for deviation from pred_sum
            sum_diff = abs(cand_sum - self.pred_sum)
            if sum_diff > 15:
                energy += (sum_diff - 15) ** 2 * 2.0
                
            # Parity and low/high penalties
            if cand_even != self.target_even:
                energy += (cand_even - self.target_even) ** 2 * 25.0
            if cand_low != self.target_low:
                energy += (cand_low - self.target_low) ** 2 * 25.0
                
        return energy

    def optimize(self, count=6, steps=5000, init_temp=100.0, alpha=0.995):
        """
        Runs Simulated Annealing to find the optimal set of tickets.
        """
        # Generate initial random state
        tickets = []
        
        if count == 6:
            # Partition [1..12] into 6 unique disjoint pairs to guarantee 100% coverage of Euro numbers
            euro_pool = list(range(1, 13))
            random.shuffle(euro_pool)
            for i in range(count):
                main = sorted(random.sample(range(1, 51), 5))
                euro = sorted([euro_pool[2*i], euro_pool[2*i+1]])
                tickets.append((main, euro))
        else:
            for _ in range(count):
                main = sorted(random.sample(range(1, 51), 5))
                euro = sorted(random.sample(range(1, 13), 2))
                tickets.append((main, euro))
            
        current_energy = self.calculate_energy(tickets, count)
        best_tickets = [(list(t[0]).copy(), list(t[1]).copy()) for t in tickets]
        best_energy = current_energy
        
        temp = init_temp
        
        for step in range(steps):
            # Propose a neighbor state by mutating one ticket
            new_tickets = [list(t[0]).copy() for t in tickets]
            new_euros = [list(t[1]).copy() for t in tickets]
            
            # Decide to mutate main numbers or Euro numbers
            if random.random() < 0.7:
                # Mutate one main number on a random ticket
                t_idx = random.randint(0, count - 1)
                idx_to_replace = random.randint(0, 4)
                choices = list(range(1, 51))
                valid_choices = [c for c in choices if c not in new_tickets[t_idx]]
                probs = self.main_probs[[c-1 for c in valid_choices]]
                probs = probs / np.sum(probs)
                
                new_num = np.random.choice(valid_choices, p=probs)
                new_tickets[t_idx][idx_to_replace] = new_num
                new_tickets[t_idx].sort()
            else:
                # Mutate Euro numbers
                if count == 6:
                    # Enforce uniqueness by swapping Euro numbers between two random tickets
                    t1_idx = random.randint(0, count - 1)
                    t2_idx = random.randint(0, count - 1)
                    while t1_idx == t2_idx:
                        t2_idx = random.randint(0, count - 1)
                        
                    e1_idx = random.randint(0, 1)
                    e2_idx = random.randint(0, 1)
                    
                    # Swap
                    new_euros[t1_idx][e1_idx], new_euros[t2_idx][e2_idx] = new_euros[t2_idx][e2_idx], new_euros[t1_idx][e1_idx]
                    new_euros[t1_idx].sort()
                    new_euros[t2_idx].sort()
                else:
                    # Standard mutation for other counts
                    t_idx = random.randint(0, count - 1)
                    idx_to_replace = random.randint(0, 1)
                    choices = list(range(1, 13))
                    valid_choices = [c for c in choices if c not in new_euros[t_idx]]
                    probs = self.euro_probs[[c-1 for c in valid_choices]]
                    probs = probs / np.sum(probs)
                    
                    new_num = np.random.choice(valid_choices, p=probs)
                    new_euros[t_idx][idx_to_replace] = new_num
                    new_euros[t_idx].sort()
                
            candidate_state = list(zip(new_tickets, new_euros))
            candidate_energy = self.calculate_energy(candidate_state, count)
            
            # Acceptance probability
            if candidate_energy < current_energy:
                accept = True
            else:
                prob = np.exp(-(candidate_energy - current_energy) / temp)
                accept = random.random() < prob
                
            if accept:
                tickets = candidate_state
                current_energy = candidate_energy
                
                if current_energy < best_energy:
                    best_tickets = [(list(t[0]).copy(), list(t[1]).copy()) for t in tickets]
                    best_energy = current_energy
                    
            # Cool down
            temp *= alpha
            
        return best_tickets
