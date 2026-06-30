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
        cursor.execute("SELECT date, num1, num2, num3, num4, num5, euro1, euro2 FROM draws ORDER BY date ASC")
        rows = cursor.fetchall()
        conn.close()
        
        dates = [r[0] for r in rows]
        main_nums = [list(r[1:6]) for r in rows]
        euro_nums = [list(r[6:8]) for r in rows]
        return dates, main_nums, euro_nums
    else:
        # Sportka
        cursor.execute("SELECT draw_date, num1, num2, num3, num4, num5, num6, supplementary FROM draws ORDER BY draw_date ASC, draw_num ASC")
        rows = cursor.fetchall()
        conn.close()
        
        rows = [r for r in rows if None not in r]
        dates = [r[0] for r in rows]
        main_nums = [list(r[1:7]) for r in rows]
        supp_nums = [[r[7]] for r in rows]
        return dates, main_nums, supp_nums

def run_quantum_prediction(game="eurojackpot"):
    """
    Runs 8-qubit (Eurojackpot) or 4-qubit (Sportka) Quantum Reservoir Computing
    and ShRIMP feature selector to output collapsed quantum state draw numbers.
    """
    data = fetch_lotto_data(game)
    if not data:
        return {"error": "No database found. Run synchronization first."}
        
    dates, main_history, euro_history = data
    if len(main_history) < 20:
        return {"error": "Not enough draws in database."}
        
    inputs = []
    pool_max = 50 if game == "eurojackpot" else 49
    
    if game == "eurojackpot":
        # Load physical features
        physical_map = {}
        physical_path = os.path.join(project_path, "physical_features.json")
        if os.path.exists(physical_path):
            try:
                with open(physical_path, "r", encoding="utf-8") as f:
                    features_data = json.load(f)
                    for fn, feat in features_data.items():
                        name = fn.replace("Eurojackpot - ", "").replace("Eurojackpot ", "")
                        parts = name.split(" - Allwyn")
                        if len(parts) >= 2:
                            date_str = parts[0].strip()
                            try:
                                day, month, year = [int(x) for x in date_str.split(".")]
                                db_date = f"{year}-{month:02d}-{day:02d}"
                                physical_map[db_date] = feat
                            except Exception:
                                pass
            except Exception as e:
                print(f"Error loading physical features: {e}")
                
        avg_kinetic_vals = [f["avg_kinetic_energy"] for f in physical_map.values()]
        max_kinetic_vals = [f["max_kinetic_energy"] for f in physical_map.values()]
        std_kinetic_vals = [f["std_kinetic_energy"] for f in physical_map.values()]
        col_freq_vals = [f["collision_frequency"] for f in physical_map.values()]
        eject_speed_vals = [f["avg_ejection_speed"] for f in physical_map.values()]
        
        global_avg_kinetic = np.mean(avg_kinetic_vals) if avg_kinetic_vals else 13.0
        global_max_kinetic = np.mean(max_kinetic_vals) if max_kinetic_vals else 35.0
        global_std_kinetic = np.mean(std_kinetic_vals) if std_kinetic_vals else 9.8
        global_avg_col = np.mean(col_freq_vals) if col_freq_vals else 260.0
        global_avg_eject = np.mean(eject_speed_vals) if eject_speed_vals else 14.5
        
        for i in range(5, len(main_history)):
            past_draws = main_history[i-5:i]
            flat_past = np.array(past_draws).flatten()
            mean_val = np.mean(flat_past)
            sum_val = np.sum(past_draws[-1])
            even_count = sum(1 for x in past_draws[-1] if x % 2 == 0)
            high_count = sum(1 for x in past_draws[-1] if x > 25)
            
            d_date = dates[i-1]
            p_avg = physical_map[d_date]["avg_kinetic_energy"] if d_date in physical_map else global_avg_kinetic
            p_max = physical_map[d_date]["max_kinetic_energy"] if d_date in physical_map else global_max_kinetic
            p_std = physical_map[d_date]["std_kinetic_energy"] if d_date in physical_map else global_std_kinetic
            p_col = physical_map[d_date]["collision_frequency"] if d_date in physical_map else global_avg_col
            p_eject = physical_map[d_date]["avg_ejection_speed"] if d_date in physical_map else global_avg_eject
            
            f1 = (mean_val - 25.0) / 25.0
            f2 = (sum_val - 125.0) / 125.0
            f3 = (even_count - 2.5) / 2.5
            f4 = (high_count - 2.5) / 2.5
            
            # Normalize physical features to [-1, 1] range
            f5 = (p_avg - 13.0) / 2.0
            f6 = (p_max - 35.0) / 3.0
            f7 = (p_col - 260.0) / 50.0
            f8 = (p_eject - 14.5) / 3.0
            inputs.append([f1, f2, f3, f4, f5, f6, f7, f8])
            
        inputs = np.array(inputs)
        qrc = QuantumReservoir(n_qubits=8, J_coeff=0.5, h_field=1.0, epsilon=0.1)
    else:
        # Sportka
        for i in range(5, len(main_history)):
            past_draws = main_history[i-5:i]
            flat_past = np.array(past_draws).flatten()
            mean_val = np.mean(flat_past)
            sum_val = np.sum(past_draws[-1])
            even_count = sum(1 for x in past_draws[-1] if x % 2 == 0)
            high_count = sum(1 for x in past_draws[-1] if x > 24)
            
            f1 = (mean_val - 24.5) / 24.5
            f2 = (sum_val - 147.0) / 147.0
            f3 = (even_count - 3.0) / 3.0
            f4 = (high_count - 3.0) / 3.0
            inputs.append([f1, f2, f3, f4])
            
        inputs = np.array(inputs)
        qrc = QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
        
    # 1. Run Quantum Reservoir Computing
    qrc_features = []
    for inp in inputs:
        states = qrc.step(inp)
        qrc_features.append(states)
    qrc_features = np.array(qrc_features)
    
    # 2. Apply ShRIMP Sparser Feature mapping
    shrimp_feats, W_sparse = shrimp_random_features(qrc_features, num_features=50, sparsity=0.7)
    
    # Readout Layer
    targets_main = np.zeros((len(shrimp_feats), pool_max))
    for idx, next_draw in enumerate(main_history[5:]):
        for num in next_draw:
            if 1 <= num <= pool_max:
                targets_main[idx, num - 1] = 1.0
                
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
        
    p_col = 0
    p_eject = 0.0
    if game == "eurojackpot" and len(dates) > 0:
        last_date = dates[-1]
        if last_date in physical_map:
            p_col = physical_map[last_date]["collision_frequency"]
            p_eject = physical_map[last_date]["avg_ejection_speed"]
            
    return {
        "game": game,
        "quantum_main": selected_main,
        "quantum_euro": selected_euro,
        "superposition_state": [round(float(p) * 100, 2) for p in probabilities],
        "qrc_energy": float(np.real(np.trace(qrc.rho @ qrc.rho))), # purity of state
        "collision_frequency": int(p_col),
        "avg_ejection_speed": float(p_eject)
    }

if __name__ == "__main__":
    print("Testing Eurojackpot Quantum Reservoir Engine:")
    res_euro = run_quantum_prediction("eurojackpot")
    print(res_euro)
    
    print("\nTesting Sportka Quantum Reservoir Engine:")
    res_sport = run_quantum_prediction("sportka")
    print(res_sport)
