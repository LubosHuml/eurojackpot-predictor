import os
import sys
import numpy as np
import sqlite3
import json
from scipy.linalg import expm

# Add project path to sys.path
project_path = "c:\\Users\\Acer\\Desktop\\Euro"
sys.path.append(project_path)

# Pauli matrices
I_spin = np.array([[1, 0], [0, 1]], dtype=complex)
X_spin = np.array([[0, 1], [1, 0]], dtype=complex)
Y_spin = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_spin = np.array([[1, 0], [0, -1]], dtype=complex)

def get_qubit_op(op, index, n_qubits=4):
    res = 1
    for i in range(n_qubits):
        curr = op if i == index else I_spin
        if i == 0:
            res = curr
        else:
            res = np.kron(res, curr)
    return res

class QuantumReservoir:
    def __init__(self, n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.epsilon = epsilon
        
        # Initialize density matrix to ground state |0...0><0...0|
        self.rho = np.zeros((self.dim, self.dim), dtype=complex)
        self.rho[0, 0] = 1.0
        
        # Construct coupling Hamiltonian (1D Ring Coupling)
        self.H_coupling = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            Zi = get_qubit_op(Z_spin, i, n_qubits)
            Zj = get_qubit_op(Z_spin, j, n_qubits)
            self.H_coupling += -J_coeff * (Zi @ Zj)
            
        # Base transverse fields
        self.X_ops = [get_qubit_op(X_spin, i, n_qubits) for i in range(n_qubits)]
        self.Z_ops = [get_qubit_op(Z_spin, i, n_qubits) for i in range(n_qubits)]
        self.h_field = h_field

    def step(self, input_features):
        """
        Evolve density matrix under input-modulated Hamiltonian.
        input_features: array of size n_qubits normalized to [-1, 1]
        """
        # Modulated transverse field Hamiltonian
        H_transverse = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.n_qubits):
            u_t = input_features[i] if i < len(input_features) else 0.0
            H_transverse += -self.h_field * (1.0 + u_t) * self.X_ops[i]
            
        H_total = self.H_coupling + H_transverse
        
        # Evolution operator U = exp(-i * H * dt) with dt = 1.0
        U = expm(-1j * H_total)
        
        # Evolve density matrix with feed-in injection
        self.rho = (1.0 - self.epsilon) * (U @ self.rho @ U.conj().T)
        # Inject ground state as noise/relaxation
        rho_inject = np.zeros((self.dim, self.dim), dtype=complex)
        rho_inject[0, 0] = 1.0
        self.rho += self.epsilon * rho_inject
        
        # Extract expectation values as reservoir features
        observables = []
        for i in range(self.n_qubits):
            # <Z_i>
            observables.append(np.real(np.trace(self.rho @ self.Z_ops[i])))
            # <X_i>
            observables.append(np.real(np.trace(self.rho @ self.X_ops[i])))
            
        # Add correlation features <Z_i Z_j>
        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            Zi = self.Z_ops[i]
            Zj = self.Z_ops[j]
            observables.append(np.real(np.trace(self.rho @ (Zi @ Zj))))
            
        return np.array(observables)

def shrimp_random_features(X, num_features=100, sparsity=0.8):
    """
    ShRIMP-inspired Sparser Random Feature projection.
    Generates sparse random projections in polynomial time.
    """
    n_samples, n_inputs = X.shape
    np.random.seed(42)
    
    # Generate random weight matrix
    W = np.random.normal(0, 1, (n_inputs, num_features))
    
    # Apply threshold pruning to enforce strict sparsity (dequantized Winning Tickets)
    threshold = np.percentile(np.abs(W), sparsity * 100)
    W[np.abs(W) < threshold] = 0.0
    
    # Bias
    b = np.random.uniform(-np.pi, np.pi, num_features)
    
    # Random projection with sine activation
    Z = np.sin(X @ W + b)
    return Z, W

def fetch_lotto_data(game="eurojackpot"):
    """
    Fetches historical drawings from SQLite database.
    """
    db_file = "eurojackpot.db" if game == "eurojackpot" else "sportka.db"
    db_path = os.path.join(project_path, db_file)
    if not os.path.exists(db_path):
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if game == "eurojackpot":
        cursor.execute("SELECT num1, num2, num3, num4, num5, euro1, euro2 FROM draws ORDER BY date ASC")
        rows = cursor.fetchall()
        conn.close()
        
        main_nums = [list(r[0:5]) for r in rows]
        euro_nums = [list(r[5:7]) for r in rows]
        return main_nums, euro_nums
    else:
        # Sportka
        cursor.execute("SELECT num1, num2, num3, num4, num5, num6, supplementary FROM draws ORDER BY draw_date ASC, draw_num ASC")
        rows = cursor.fetchall()
        conn.close()
        
        rows = [r for r in rows if None not in r]
        main_nums = [list(r[0:6]) for r in rows]
        supp_nums = [[r[6]] for r in rows]
        return main_nums, supp_nums

def run_quantum_prediction(game="eurojackpot"):
    """
    Runs 4-qubit Quantum Reservoir Computing and ShRIMP feature selector
    to output collapsed quantum state draw numbers.
    """
    data = fetch_lotto_data(game)
    if not data:
        return {"error": "No database found. Run synchronization first."}
        
    main_history, euro_history = data
    if len(main_history) < 20:
        return {"error": "Not enough draws in database."}
        
    # Prepare input features: rolling sum, parity, high/low, and mean of past 5 draws
    inputs = []
    pool_max = 50 if game == "eurojackpot" else 49
    
    for i in range(5, len(main_history)):
        past_draws = main_history[i-5:i]
        flat_past = np.array(past_draws).flatten()
        
        mean_val = np.mean(flat_past)
        sum_val = np.sum(past_draws[-1])
        even_count = sum(1 for x in past_draws[-1] if x % 2 == 0)
        high_count = sum(1 for x in past_draws[-1] if x > (pool_max / 2))
        
        # Scale to [-1, 1] range for QRC magnetic field modulation
        f1 = (mean_val - (pool_max/2)) / (pool_max/2)
        f2 = (sum_val - (pool_max*2.5)) / (pool_max*2.5)
        f3 = (even_count - 2.5) / 2.5
        f4 = (high_count - 2.5) / 2.5
        
        inputs.append([f1, f2, f3, f4])
        
    inputs = np.array(inputs)
    
    # 1. Run Quantum Reservoir Computing
    qrc = QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    qrc_features = []
    for inp in inputs:
        states = qrc.step(inp)
        qrc_features.append(states)
    qrc_features = np.array(qrc_features) # Shape: (N, 12)
    
    # 2. Apply ShRIMP Sparser Feature mapping
    shrimp_feats, W_sparse = shrimp_random_features(qrc_features, num_features=50, sparsity=0.7)
    
    # Readout Layer: Ridge Regression target to predict probability amplitudes of the number pool
    # Target matrix: 1 if number was drawn in next draw, else 0
    targets_main = np.zeros((len(shrimp_feats), pool_max))
    for idx, next_draw in enumerate(main_history[5:]):
        for num in next_draw:
            if 1 <= num <= pool_max:
                targets_main[idx, num - 1] = 1.0
                
    # Ridge Regression analytical solver (extremely fast, zero overhead!)
    # W_readout = (Z^T Z + alpha*I)^-1 Z^T T
    Z = shrimp_feats
    Z_T_Z = Z.T @ Z
    alpha = 1.0
    ridge_inv = np.linalg.inv(Z_T_Z + alpha * np.eye(Z.shape[1]))
    W_readout = ridge_inv @ Z.T @ targets_main
    
    # Predict probability amplitudes for the next draw (last sequence)
    last_reservoir_state = Z[-1:]
    amplitudes = last_reservoir_state @ W_readout
    amplitudes = np.clip(amplitudes[0], 0.0, 1.0)
    
    # Quantum superposition normalization (probabilities sum to 1.0)
    probabilities = amplitudes / np.sum(amplitudes)
    
    # Collapse Wave Function (perform simulated quantum measurement to select numbers)
    # Pick 5 numbers (or 6 for Sportka) without replacement based on probabilities
    n_main = 5 if game == "eurojackpot" else 6
    selected_main = []
    temp_probs = probabilities.copy()
    
    # Ensure no division by zero if all probabilities are zero
    if np.sum(temp_probs) == 0:
        temp_probs = np.ones(pool_max) / pool_max
        
    for _ in range(n_main):
        temp_probs /= np.sum(temp_probs)
        num = np.random.choice(range(1, pool_max + 1), p=temp_probs)
        selected_main.append(int(num))
        temp_probs[num - 1] = 0.0 # collapse probability to 0 to prevent duplicates
        
    selected_main.sort()
    
    # Quantum Entanglement simulation for Euro / Supplementary numbers:
    # We simulate a Bell-state entanglement where the sum of main numbers shifts the Euro probabilities
    selected_euro = []
    if game == "eurojackpot":
        euro_max = 12
        targets_euro = np.zeros((len(shrimp_feats), euro_max))
        for idx, next_euro in enumerate(euro_history[5:]):
            for num in next_euro:
                if 1 <= num <= euro_max:
                    targets_euro[idx, num - 1] = 1.0
                    
        W_readout_euro = ridge_inv @ Z.T @ targets_euro
        amplitudes_euro = last_reservoir_state @ W_readout_euro
        amplitudes_euro = np.clip(amplitudes_euro[0], 0.0, 1.0)
        
        # Apply Entanglement shift: if sum of main numbers is even, boost even Euro numbers by 20%
        main_sum = sum(selected_main)
        for num_idx in range(euro_max):
            num = num_idx + 1
            if main_sum % 2 == 0 and num % 2 == 0:
                amplitudes_euro[num_idx] *= 1.2
            elif main_sum % 2 != 0 and num % 2 != 0:
                amplitudes_euro[num_idx] *= 1.2
                
        probabilities_euro = amplitudes_euro / np.sum(amplitudes_euro)
        temp_probs_euro = probabilities_euro.copy()
        
        for _ in range(2):
            temp_probs_euro /= np.sum(temp_probs_euro)
            num = np.random.choice(range(1, euro_max + 1), p=temp_probs_euro)
            selected_euro.append(int(num))
            temp_probs_euro[num - 1] = 0.0
            
        selected_euro.sort()
    else:
        # Sportka supplementary number
        supp_max = 49
        targets_supp = np.zeros((len(shrimp_feats), supp_max))
        for idx, next_supp in enumerate(euro_history[5:]):
            for num in next_supp:
                if 1 <= num <= supp_max:
                    targets_supp[idx, num - 1] = 1.0
                    
        W_readout_supp = ridge_inv @ Z.T @ targets_supp
        amplitudes_supp = last_reservoir_state @ W_readout_supp
        amplitudes_supp = np.clip(amplitudes_supp[0], 0.0, 1.0)
        
        probabilities_supp = amplitudes_supp / np.sum(amplitudes_supp)
        temp_probs_supp = probabilities_supp.copy()
        
        num = np.random.choice(range(1, supp_max + 1), p=temp_probs_supp)
        selected_euro.append(int(num))
        
    return {
        "game": game,
        "quantum_main": selected_main,
        "quantum_euro": selected_euro,
        "superposition_state": [round(float(p) * 100, 2) for p in probabilities],
        "qrc_energy": float(np.real(np.trace(qrc.rho @ qrc.rho))) # purity of state
    }

if __name__ == "__main__":
    print("Testing Eurojackpot Quantum Reservoir Engine:")
    res_euro = run_quantum_prediction("eurojackpot")
    print(res_euro)
    
    print("\nTesting Sportka Quantum Reservoir Engine:")
    res_sport = run_quantum_prediction("sportka")
    print(res_sport)
