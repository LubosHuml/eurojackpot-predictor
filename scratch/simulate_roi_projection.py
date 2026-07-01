import os
import sys

def simulate_projection(initial_capital=84.0, monthly_deposit=100.0, months=54, monthly_rate=0.07):
    capital = initial_capital
    total_deposited = initial_capital
    history = []
    
    for m in range(1, months + 1):
        # Deposit at the start of the month
        capital += monthly_deposit
        total_deposited += monthly_deposit
        
        # Apply risk management rules to determine exposure
        if capital < 1000.0:
            alloc_pct = 0.50
            leverage = 10
        elif capital < 5000.0:
            alloc_pct = 0.50
            leverage = 8
        elif capital < 20000.0:
            alloc_pct = 0.50
            leverage = 5
        else:
            alloc_pct = 0.50
            leverage = 3
            
        exposure_value = capital * alloc_pct * leverage
        
        # Calculate monthly strategy return
        # Since the monthly_rate is strategy-level return (net after leverage), we compound it on total capital:
        # capital_new = capital * (1 + monthly_rate)
        # Note: If the monthly_rate is asset-level, we would multiply by exposure, but standard backtest return
        # is already strategy net return. Let's assume monthly_rate is strategy-level net return.
        profit = capital * monthly_rate
        capital += profit
        
        history.append({
            "month": m,
            "deposited": total_deposited,
            "capital": capital,
            "leverage": leverage,
            "profit": profit
        })
        
    return history

def print_projection_results():
    months = 54 # July 2026 to Dec 2030 = 54 months
    initial_cap = 84.32
    monthly_dep = 100.0
    
    rates = {
        "Conservative (3% monthly / ~42.6% APR)": 0.03,
        "Balanced (7% monthly / ~125.2% APR)": 0.07,
        "Optimistic (12% monthly / ~289.6% APR)": 0.12
    }
    
    print("=========================================================================")
    print("      BTCUSDT TRADING STRATEGY PROJECTION UP TO 2030 (54 MONTHS)        ")
    print("      Initial: 84.32 USDT | Monthly Deposit: +100.00 USDT              ")
    print("=========================================================================")
    
    for label, rate in rates.items():
        history = simulate_projection(initial_cap, monthly_dep, months, rate)
        final_cap = history[-1]["capital"]
        final_dep = history[-1]["deposited"]
        total_profit = final_cap - final_dep
        
        print(f"\nScenario: {label}")
        print(f"  - Total Deposited: {final_dep:,.2f} USDT")
        print(f"  - Final Account Value: {final_cap:,.2f} USDT")
        print(f"  - Net Profit Generated: {total_profit:,.2f} USDT")
        print("  - Growth Milestones:")
        # Print milestones
        for idx in [11, 23, 35, 47, 53]: # Years 1, 2, 3, 4, 4.5
            h = history[idx]
            yr = (idx + 1) // 12
            month_label = f"Month {idx+1} (Year {yr + 2026} / {(idx+1)%12 if (idx+1)%12 != 0 else 12})"
            print(f"    * {month_label}: {h['capital']:,.2f} USDT (Lev: {h['leverage']}x)")

if __name__ == "__main__":
    print_projection_results()
