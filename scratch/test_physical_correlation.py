import sqlite3
import pandas as pd
import numpy as np

def analyze_correlations():
    conn = sqlite3.connect("eurojackpot.db")
    df = pd.read_sql_query("SELECT num1, num2, num3, num4, num5, euro1, euro2 FROM draws", conn)
    conn.close()
    
    print(f"Total draws analyzed: {len(df)}")
    
    # Calculate conditional probabilities: P(Main = m | Euro = e)
    # We want to see if certain Euro numbers have strong affinity to certain main numbers
    affinity = np.zeros((12, 50)) # 12 Euros, 50 Mains
    euro_counts = np.zeros(12)
    
    for _, row in df.iterrows():
        m_nums = [int(row[f'num{i}']) for i in range(1, 6)]
        e_nums = [int(row[f'euro{i}']) for i in range(1, 3)]
        
        for e in e_nums:
            if 1 <= e <= 12:
                euro_counts[e-1] += 1
                for m in m_nums:
                    if 1 <= m <= 50:
                        affinity[e-1, m-1] += 1
                        
    # Normalize to get P(Main | Euro)
    for e in range(12):
        if euro_counts[e] > 0:
            affinity[e, :] /= euro_counts[e]
            
    # Calculate global main number probabilities P(Main)
    global_main = np.zeros(50)
    for _, row in df.iterrows():
        m_nums = [int(row[f'num{i}']) for i in range(1, 6)]
        for m in m_nums:
            if 1 <= m <= 50:
                global_main[m-1] += 1
    global_main /= len(df)
    
    # Lift = P(Main | Euro) / P(Main)
    # If lift > 1.2, it means the main number is 20%+ more likely to appear when that Euro number is drawn!
    lift = np.zeros((12, 50))
    for e in range(12):
        for m in range(50):
            if global_main[m] > 0:
                lift[e, m] = affinity[e, m] / global_main[m]
                
    print("\nTop 3 main numbers with highest lift for each Euro number:")
    for e in range(12):
        top_m = np.argsort(lift[e, :])[-3:][::-1]
        top_str = ", ".join([f"Main {m+1} (lift: {lift[e, m]:.2f})" for m in top_m])
        print(f"Euro {e+1}: {top_str}")
        
    # Save lift matrix to a file so TicketOptimizer can load it dynamically
    np.save("crypto/euro_main_lift.npy", lift)
    print("\nSaved Euro-Main lift matrix to crypto/euro_main_lift.npy")

if __name__ == "__main__":
    analyze_correlations()
