import MetaTrader5 as mt5

def check_symbols():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
        
    print("Checking symbol names in MT5...")
    
    # 1. USDJPY check
    usdjpy_symbols = mt5.symbols_get("*USDJPY*")
    print("\n--- USDJPY Matches ---")
    for s in usdjpy_symbols:
        print(s.name)
        
    # 2. XAUUSD check
    gold_symbols = mt5.symbols_get("*XAUUSD*")
    if not gold_symbols:
        gold_symbols = mt5.symbols_get("*GOLD*")
    print("\n--- Gold Matches ---")
    for s in gold_symbols:
        print(s.name)
        
    # 3. S&P 500 Index / Tech Index check
    # Let's search for US500, SPX, USA500, USTEC, US100
    index_symbols = mt5.symbols_get("*500*")
    if not index_symbols:
        index_symbols = mt5.symbols_get("*SPX*")
    print("\n--- S&P 500 Matches ---")
    for s in index_symbols:
        print(s.name)
        
    # 4. Nasdaq 100 check
    nasdaq_symbols = mt5.symbols_get("*100*")
    if not nasdaq_symbols:
        nasdaq_symbols = mt5.symbols_get("*TEC*")
    print("\n--- Nasdaq Matches ---")
    for s in nasdaq_symbols:
        print(s.name)
        
    mt5.shutdown()

if __name__ == "__main__":
    check_symbols()
