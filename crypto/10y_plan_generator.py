import json
from datetime import datetime

def generate_plan():
    # Start in July 2026
    start_year = 2026
    start_month = 7
    
    balance = 100.11
    plan = []
    
    months_czech = {
        1: "Leden", 2: "Únor", 3: "Březen", 4: "Duben", 5: "Květen", 6: "Červen",
        7: "Červenec", 8: "Srpen", 9: "Září", 10: "Říjen", 11: "Listopad", 12: "Prosinec"
    }
    
    # 126 months from July 2026 to December 2036
    curr_year = start_year
    curr_month = start_month
    
    for i in range(126):
        month_label = f"{months_czech[curr_month]} {curr_year}"
        
        deposit = 100.0
        start_bal = balance + deposit
        
        # Deleveraging logic rules
        if start_bal < 1000.0:
            rate = 0.165
            leverage = 10
            alloc = 15
        elif start_bal < 5000.0:
            rate = 0.105
            leverage = 8
            alloc = 10
        elif start_bal < 20000.0:
            rate = 0.055
            leverage = 5
            alloc = 6
        else:
            rate = 0.025
            leverage = 3
            alloc = 3
            
        profit = start_bal * rate
        end_bal = start_bal + profit
        balance = end_bal
        
        plan.append({
            "index": i + 1,
            "month": month_label,
            "year": curr_year,
            "deposit": deposit,
            "rate_pct": round(rate * 100, 1),
            "leverage": leverage,
            "alloc_pct": alloc,
            "expected_balance": round(end_bal, 2),
            "actual_balance": None
        })
        
        # Increment month
        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1
            
    with open("crypto_10y_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4, ensure_ascii=False)
    print("crypto_10y_plan.json generated successfully!")

if __name__ == "__main__":
    generate_plan()
