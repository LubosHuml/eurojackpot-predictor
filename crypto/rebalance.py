import os
import sys
import time

# Add project path dynamically to sys.path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_path, "crypto"))

import executor

def rebalance_portfolio():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    print("Rebalancing portfolio to scale up sizes for new 100 USDT balance...")
    
    # 1. Close all existing positions first
    for sym in symbols:
        side, size = executor.get_active_position(sym)
        if side is not None:
            print(f"Closing existing position for {sym} ({side} size={size}) to scale up...")
            executor.close_futures_position(sym, side, size)
            
    print("Waiting 3 seconds for order settlement on Bybit...")
    time.sleep(3.0)
    
    # 2. Run executor to open new correctly-sized positions
    print("Running executor to open new compounding positions...")
    executor.run_execution_loop()
    print("Rebalancing completed successfully!")

if __name__ == "__main__":
    rebalance_portfolio()
