import numpy as np
import random

class TicketOptimizer:
    def __init__(self, main_probs, euro_probs, pred_sum, pred_counts):
        self.main_probs = np.array(main_probs)
        self.euro_probs = np.array(euro_probs)
        self.pred_sum = pred_sum
        
        # Predicted counts: [Even, Odd, Low, High]
        self.target_even = int(np.clip(np.round(pred_counts[0]), 0, 5))
        self.target_low = int(np.clip(np.round(pred_counts[2]), 0, 5))
        
    def calculate_energy(self, tickets):
        """
        Calculates the energy (loss) of a set of tickets.
        We want to minimize this energy.
        """
        energy = 0.0
        
        # 1. Probability component (maximize prob of selected numbers)
        # We use negative log-likelihood
        epsilon = 1e-12
        for main_nums, euro_nums in tickets:
            for n in main_nums:
                energy -= np.log(self.main_probs[n - 1] + epsilon) * 2.0
            for e in euro_nums:
                energy -= np.log(self.euro_probs[e - 1] + epsilon) * 3.0
                
        # 2. Overlap penalty (minimize duplicate numbers across tickets to maximize coverage)
        main_counts = np.zeros(50)
        euro_counts = np.zeros(12)
        for main_nums, euro_nums in tickets:
            for n in main_nums:
                main_counts[n - 1] += 1
            for e in euro_nums:
                euro_counts[e - 1] += 1
                
        # Quadratic penalty for multiple occurrences of the same number
        energy += np.sum(main_counts ** 2) * 5.0
        energy += np.sum(euro_counts ** 2) * 10.0
        
        # 3. Structural constraint penalties for each ticket
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
        for _ in range(count):
            main = sorted(random.sample(range(1, 51), 5))
            euro = sorted(random.sample(range(1, 13), 2))
            tickets.append((main, euro))
            
        current_energy = self.calculate_energy(tickets)
        best_tickets = [(list(t[0]).copy(), list(t[1]).copy()) for t in tickets]
        best_energy = current_energy
        
        temp = init_temp
        
        for step in range(steps):
            # Propose a neighbor state by mutating one ticket
            new_tickets = [list(t[0]).copy() for t in tickets]
            new_euros = [list(t[1]).copy() for t in tickets]
            
            t_idx = random.randint(0, count - 1)
            
            # Mutate main or euro numbers
            if random.random() < 0.7:
                # Mutate one main number
                idx_to_replace = random.randint(0, 4)
                # Sample a new number according to probabilities
                choices = list(range(1, 51))
                # Exclude numbers already on this ticket
                valid_choices = [c for c in choices if c not in new_tickets[t_idx]]
                probs = self.main_probs[[c-1 for c in valid_choices]]
                probs = probs / np.sum(probs)
                
                new_num = np.random.choice(valid_choices, p=probs)
                new_tickets[t_idx][idx_to_replace] = new_num
                new_tickets[t_idx].sort()
            else:
                # Mutate one euro number
                idx_to_replace = random.randint(0, 1)
                choices = list(range(1, 13))
                valid_choices = [c for c in choices if c not in new_euros[t_idx]]
                probs = self.euro_probs[[c-1 for c in valid_choices]]
                probs = probs / np.sum(probs)
                
                new_num = np.random.choice(valid_choices, p=probs)
                new_euros[t_idx][idx_to_replace] = new_num
                new_euros[t_idx].sort()
                
            candidate_state = list(zip(new_tickets, new_euros))
            candidate_energy = self.calculate_energy(candidate_state)
            
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
