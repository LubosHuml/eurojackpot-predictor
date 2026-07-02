import os
import sys
import time

def test_mt5():
    print("=========================================================")
    print("          METATRADER 5 / PEPPERSTONE CONNECTION TEST      ")
    print("=========================================================")
    
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] The 'MetaTrader5' Python library is not installed in this environment.")
        print("To install it, run: pip install MetaTrader5")
        return False
        
    print("[OK] MetaTrader5 Python library is successfully imported.")
    
    # Attempt to initialize MT5
    # This requires the MetaTrader 5 terminal to be installed on the PC!
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialization failed. Error code: {mt5.last_error()}")
        print("Please ensure that:")
        print("1. MetaTrader 5 terminal is installed on this computer.")
        print("2. You are logged into your Pepperstone account in the MT5 terminal.")
        print("3. Algorithmic trading is allowed in MT5 settings (Tools -> Options -> Expert Advisors -> Allow Algorithmic Trading).")
        return False
        
    print("[OK] Successfully connected to the MetaTrader 5 terminal!")
    
    # Fetch terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info is not None:
        terminal_info_dict = terminal_info._asdict()
        print(f"\n--- Terminal Info ---")
        print(f"Company: {terminal_info_dict.get('company')}")
        print(f"Name: {terminal_info_dict.get('name')}")
        print(f"Path: {terminal_info_dict.get('path')}")
        print(f"Connected: {terminal_info_dict.get('connected')}")
    
    # Fetch account info
    account_info = mt5.account_info()
    if account_info is not None:
        acc_dict = account_info._asdict()
        print(f"\n--- Account Info ---")
        print(f"Account Login ID: {acc_dict.get('login')}")
        print(f"Server Name: {acc_dict.get('server')}")
        print(f"Leverage: 1:{acc_dict.get('leverage')}")
        print(f"Balance: {acc_dict.get('balance'):.2f} {acc_dict.get('currency')}")
        print(f"Equity: {acc_dict.get('equity'):.2f} {acc_dict.get('currency')}")
        print(f"Margin Free: {acc_dict.get('margin_free'):.2f} {acc_dict.get('currency')}")
    else:
        print("[WARNING] Could not retrieve account info. Ensure you are logged into a broker server.")
        
    # Test fetching some ticks for XAUUSD (Gold) or EURUSD
    symbol = "XAUUSD"
    selected = mt5.symbol_select(symbol, True)
    if not selected:
        # Try GOLD or other variants
        symbol = "GOLD"
        selected = mt5.symbol_select(symbol, True)
        
    if selected:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            print(f"\n--- {symbol} Real-Time Price ---")
            print(f"Bid: {tick.bid}")
            print(f"Ask: {tick.ask}")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tick.time))}")
        else:
            print(f"[WARNING] Could not fetch tick for {symbol}.")
    else:
        print(f"[WARNING] Symbol {symbol} is not available or selected in MT5 Market Watch.")
        
    # Shut down connection
    mt5.shutdown()
    print("\n=========================================================")
    print("                CONNECTION TEST COMPLETED                ")
    print("=========================================================")
    return True

if __name__ == "__main__":
    test_mt5()
