import os
import sys
import time
import hmac
import hashlib
import requests
import json
from datetime import datetime

# API credentials
API_KEY = "NhyIWck9muRVkEIf1I"
API_SECRET = "hKAHzNihIOmSveESysf6xpVYpbhLVUljjIZ0"

def get_bybit_headers(payload_str, timestamp):
    val = timestamp + API_KEY + "5000" + payload_str
    signature = hmac.new(API_SECRET.encode('utf-8'), val.encode('utf-8'), hashlib.sha256).hexdigest()
    return {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json"
    }

def get_wallet_state():
    """
    Fetches available USDT margin and total USDT wallet balance (cash).
    Returns (available_margin, total_cash).
    """
    timestamp = str(int(time.time() * 1000))
    params = "accountType=UNIFIED"
    
    val = timestamp + API_KEY + "5000" + params
    signature = hmac.new(API_SECRET.encode('utf-8'), val.encode('utf-8'), hashlib.sha256).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json"
    }
    
    url = f"https://api.bybit.com/v5/account/wallet-balance?{params}"
    try:
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            print(f"Bybit API Error (status {res.status_code}) fetching wallet-balance: {res.text[:200]}")
            return 0.0, 12.0
        data = res.json()
        if data.get("retCode") == 0:
            coins = data["result"]["list"][0]["coin"]
            usdt_avail = 0.0
            usdt_total = 12.0 # fallback
            for c in coins:
                if c["coin"] == "USDT":
                    wb = float(c.get("walletBalance", 0.0) or 0.0)
                    im = float(c.get("totalPositionIM", 0.0) or 0.0)
                    order_im = float(c.get("totalOrderIM", 0.0) or 0.0)
                    usdt_avail = max(0.0, wb - im - order_im)
                    usdt_total = wb
            return usdt_avail, usdt_total
    except Exception as e:
        print(f"Error fetching balances: {e}")
    return 0.0, 12.0

def get_active_position(symbol):
    """
    Checks if we have an active linear position on a given symbol.
    Returns: (side, size) e.g. ("Buy", 0.001) or (None, 0.0)
    """
    timestamp = str(int(time.time() * 1000))
    params = f"category=linear&symbol={symbol}"
    
    val = timestamp + API_KEY + "5000" + params
    signature = hmac.new(API_SECRET.encode('utf-8'), val.encode('utf-8'), hashlib.sha256).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json"
    }
    
    url = f"https://api.bybit.com/v5/position/list?{params}"
    try:
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            print(f"Bybit API Error (status {res.status_code}) fetching active positions for {symbol}: {res.text[:200]}")
            return None, 0.0
        data = res.json()
        if data.get("retCode") == 0:
            pos_list = data["result"]["list"]
            for pos in pos_list:
                size = float(pos.get("size", 0.0))
                if size > 0.0:
                    return pos["side"], size
    except Exception as e:
        print(f"Error fetching active position for {symbol}: {e}")
    return None, 0.0

def set_leverage(symbol, leverage=10):
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/position/set-leverage"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    }
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        print(f"Bybit Set Leverage Response for {symbol}:", res.json())
    except Exception as e:
        print(f"Error setting leverage: {e}")

def open_futures_position(symbol, side, size, leverage=10, sl_price=0.0, tp_price=0.0):
    """
    Opens a futures position with built-in Stop Loss and Take Profit.
    """
    set_leverage(symbol, leverage)
    time.sleep(0.5)
    
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/order/create"
    
    pos_idx = 1 if side == "Buy" else 2
    
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "positionIdx": pos_idx,
        "orderType": "Market",
        "qty": f"{size}",
        "timeInForce": "GTC",
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice"
    }
    
    if sl_price > 0:
        payload["stopLoss"] = f"{sl_price:.2f}"
    if tp_price > 0:
        payload["takeProfit"] = f"{tp_price:.2f}"
        
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        data = res.json()
        print(f"Bybit Open Futures Response ({symbol} {side} qty={size}):", data)
        return data
    except Exception as e:
        print(f"Error opening position for {symbol}: {e}")
        return None

def close_futures_position(symbol, side, size):
    close_side = "Sell" if side == "Buy" else "Buy"
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/order/create"
    
    pos_idx = 1 if side == "Buy" else 2
    
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": close_side,
        "positionIdx": pos_idx,
        "orderType": "Market",
        "qty": f"{size}",
        "reduceOnly": True
    }
    
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        data = res.json()
        print(f"Bybit Close Futures Response for {symbol}:", data)
        return data
    except Exception as e:
        print(f"Error closing position for {symbol}: {e}")
        return None

def send_alert(subject, message):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    email_to = os.environ.get("EMAIL_TO", "lubos8huml@gmail.com")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print(f"SMTP credentials missing. Alert logged locally: {subject} - {message}")
        return

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = email_to
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, email_to, msg.as_string())
        server.close()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

def load_trade_state():
    state_path = os.path.join(os.path.dirname(__file__), "crypto_trade_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_trade_state(state):
    state_path = os.path.join(os.path.dirname(__file__), "crypto_trade_state.json")
    try:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving trade state: {e}")

def run_execution_loop():
    crypto_dir = os.path.dirname(__file__)
    prediction_path = os.path.join(crypto_dir, "crypto_live_prediction.json")
    
    if not os.path.exists(prediction_path):
        print("No prediction cache found.")
        return
        
    with open(prediction_path, "r") as f:
        pred_data = json.load(f)
        
    # Fetch current account wallet status
    usdt_avail, usdt_total = get_wallet_state()
    print(f"Balances - Available USDT Margin: {usdt_avail:.2f}, Total Cash: {usdt_total:.2f}")
    
    trade_state = load_trade_state()
    
    # Close legacy positions for disabled assets (ETH and SOL) to prevent unmanaged risk
    for sym in ["ETHUSDT", "SOLUSDT"]:
        try:
            pos_side, pos_size = get_active_position(sym)
            if pos_side is not None and pos_size > 0.0:
                print(f"[Cleanup] Closing legacy position for disabled symbol {sym} ({pos_side} size={pos_size})...")
                close_res = close_futures_position(sym, pos_side, pos_size)
                if close_res and close_res.get("retCode") == 0:
                    send_alert(f"[AI Bot] Closed Legacy {sym}", f"Successfully closed orphaned position for {sym} ({pos_side} size={pos_size}) on Bybit.")
        except Exception as e:
            print(f"Error cleaning up disabled symbol {sym}: {e}")
            
    symbols = ["BTCUSDT"]
    
    # Precision decimal mapping for each contract
    qty_decimals = {
        "BTCUSDT": 3,  # Step size: 0.001
        "ETHUSDT": 2,  # Step size: 0.01
        "SOLUSDT": 1   # Step size: 0.1
    }
    
    min_qty = {
        "BTCUSDT": 0.001,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.1
    }
    
    for sym in symbols:
        sym_key = sym.lower()
        if sym_key not in pred_data:
            continue
            
        sym_pred = pred_data[sym_key]
        action = sym_pred.get("action", "WAIT / NEUTRAL")
        current_price = sym_pred.get("price")
        sl = sym_pred.get("stop_loss")
        tp = sym_pred.get("take_profit")
        pred_hour = sym_pred.get("datetime")
        
        # Check current position
        pos_side, pos_size = get_active_position(sym)
        print(f"[{sym}] Active Position - Side: {pos_side}, Size: {pos_size}, AI Action: {action}")
        
        target_side = None
        if action == "BUY / LONG":
            target_side = "Buy"
        elif action == "SELL / SHORT":
            target_side = "Sell"
            
        # Case A: Correct position already running
        if pos_side == target_side and pos_side is not None:
            print(f"[{sym}] Position aligns with target ({pos_side}). No action.")
            continue
            
        # Case B: Incorrect position running (signal change or wait signal)
        if pos_side is not None:
            print(f"[{sym}] Closing incorrect position ({pos_side} size={pos_size}) due to signal shift...")
            close_res = close_futures_position(sym, pos_side, pos_size)
            if close_res and close_res.get("retCode") == 0:
                message = f"{sym} position CLOSED ({pos_side} size={pos_size}) at price {current_price:,.2f} USDT due to AI signal shift."
                send_alert(f"[AI Bot] {sym} Futures Position Closed", message)
                # Refresh local state variables
                pos_side = None
                pos_size = 0.0
                time.sleep(1.0)
                
        # Case C: No position, and active signal triggers
        if pos_side is None and target_side is not None:
            # Re-entry block safety guardrail:
            # If we already opened a position for this specific hourly prediction window,
            # we do not open another one (prevents overtrading and SL-whipsaw loops).
            if trade_state.get(sym_key) == pred_hour:
                print(f"[{sym}] Already traded during this hourly prediction window ({pred_hour}). Skipping re-entry.")
                continue
            # Set allocation weight to exactly 50% of the deposit for BTCUSDT as requested
            alloc_pct = 0.50
            if usdt_total < 1000.0:
                leverage = 10
            elif usdt_total < 5000.0:
                leverage = 8
            elif usdt_total < 20000.0:
                leverage = 5
            else:
                leverage = 3
                
            target_margin = alloc_pct * usdt_total
            target_value = float(leverage) * target_margin
            target_qty = target_value / current_price
            
            # Round according to symbol decimal precision
            dec = qty_decimals[sym]
            size = max(min_qty[sym], round(target_qty, dec))
            
            # Required margin for this position (with 5% slippage buffer)
            margin_required = (size * current_price / float(leverage)) * 1.05
            
            if usdt_avail >= margin_required:
                print(f"[{sym}] Opening {leverage}x Futures position ({target_side}) for {size} (Margin req: {margin_required:.2f} USDT)...")
                order_res = open_futures_position(sym, target_side, size, leverage=leverage, sl_price=sl, tp_price=tp)
                if order_res and order_res.get("retCode") == 0:
                    message = f"{sym} {leverage}x Futures Position OPENED ({target_side} size={size}) at price {current_price:,.2f} USDT.\nStop Loss (SL): {sl:,.2f} USDT\nTake Profit (TP): {tp:,.2f} USDT"
                    send_alert(f"[AI Bot] {sym} Futures Position Opened - {action}", message)
                    # Record the hour of the trade to block further re-entries in this window
                    trade_state[sym_key] = pred_hour
                    save_trade_state(trade_state)
                else:
                    print(f"[{sym}] Failed to open futures position.")
            else:
                print(f"[{sym}] Insufficient available margin ({usdt_avail:.2f} USDT). Required: {margin_required:.2f} USDT.")

if __name__ == "__main__":
    run_execution_loop()
