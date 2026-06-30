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

def get_wallet_balances():
    """
    Fetches Unified wallet balances for BTC and USDT, accounting for locked position margin.
    Returns available USDT margin.
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
        data = res.json()
        if data.get("retCode") == 0:
            coins = data["result"]["list"][0]["coin"]
            usdt = 0.0
            for c in coins:
                if c["coin"] == "USDT":
                    wb = float(c.get("walletBalance", 0.0) or 0.0)
                    im = float(c.get("totalPositionIM", 0.0) or 0.0)
                    order_im = float(c.get("totalOrderIM", 0.0) or 0.0)
                    avail = max(0.0, wb - im - order_im)
                    usdt = avail
            return usdt
    except Exception as e:
        print(f"Error fetching balances: {e}")
    return 0.0

def get_active_position():
    """
    Checks if we have an active linear (Futures) position on BTCUSDT.
    Returns: (side, size) e.g. ("Buy", 0.001) or ("Sell", 0.001) or (None, 0.0)
    """
    timestamp = str(int(time.time() * 1000))
    params = "category=linear&symbol=BTCUSDT"
    
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
        data = res.json()
        if data.get("retCode") == 0:
            pos_list = data["result"]["list"]
            for pos in pos_list:
                size = float(pos.get("size", 0.0))
                if size > 0.0:
                    return pos["side"], size
    except Exception as e:
        print(f"Error fetching active position: {e}")
    return None, 0.0

def set_leverage(leverage=10):
    """
    Sets leverage for BTCUSDT linear contract.
    """
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/position/set-leverage"
    payload = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    }
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        # 110043 is "leverage not modified" which is fine to ignore
        print("Bybit Set Leverage Response:", res.json())
    except Exception as e:
        print(f"Error setting leverage: {e}")

def open_futures_position(side, size=0.001, sl_price=0.0, tp_price=0.0):
    """
    Opens a futures position with built-in Stop Loss and Take Profit.
    """
    set_leverage(10)
    time.sleep(0.5)
    
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/order/create"
    
    payload = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": side, # "Buy" for Long, "Sell" for Short
        "orderType": "Market",
        "qty": f"{size:.3f}",
        "timeInForce": "GTC",
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice"
    }
    
    if sl_price > 0:
        payload["stopLoss"] = f"{sl_price:.1f}"
    if tp_price > 0:
        payload["takeProfit"] = f"{tp_price:.1f}"
        
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        data = res.json()
        print(f"Bybit Open Futures Response ({side} qty={size}):", data)
        return data
    except Exception as e:
        print(f"Error opening position: {e}")
        return None

def close_futures_position(side, size):
    """
    Closes an active futures position.
    """
    close_side = "Sell" if side == "Buy" else "Buy"
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/order/create"
    
    payload = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": close_side,
        "orderType": "Market",
        "qty": f"{size:.3f}",
        "reduceOnly": True
    }
    
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        data = res.json()
        print(f"Bybit Close Futures Response:", data)
        return data
    except Exception as e:
        print(f"Error closing position: {e}")
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

def run_execution_loop():
    crypto_dir = os.path.dirname(__file__)
    prediction_path = os.path.join(crypto_dir, "crypto_live_prediction.json")
    
    if not os.path.exists(prediction_path):
        print("No prediction cache found.")
        return
        
    with open(prediction_path, "r") as f:
        pred_data = json.load(f)
        
    action = pred_data.get("action", "WAIT / NEUTRAL")
    current_price = pred_data.get("price")
    sl = pred_data.get("stop_loss")
    tp = pred_data.get("take_profit")
    
    # 1. Fetch current status
    usdt_bal = get_wallet_balances()
    pos_side, pos_size = get_active_position()
    
    print(f"Balances - Available USDT Margin: {usdt_bal:.2f}")
    print(f"Active Position - Side: {pos_side}, Size: {pos_size}")
    
    # 2. Check alignment with AI prediction
    target_side = None
    if action == "BUY / LONG":
        target_side = "Buy"
    elif action == "SELL / SHORT":
        target_side = "Sell"
        
    # Case A: We are in the correct position already
    if pos_side == target_side and pos_side is not None:
        print(f"Position aligns with target ({pos_side}). No action needed.")
        return
        
    # Case B: We are in a position, but it does NOT align with target (either signal shift or wait signal)
    if pos_side is not None:
        print(f"Closing incorrect position ({pos_side} size={pos_size}) due to signal shift...")
        close_res = close_futures_position(pos_side, pos_size)
        if close_res and close_res.get("retCode") == 0:
            message = f"BTC/USDT position CLOSED ({pos_side} size={pos_size}) at price {current_price:,.1f} USDT due to AI signal shift."
            send_alert("[AI Bot] BTC/USDT Futures Position Closed", message)
            # Update local state
            pos_side = None
            pos_size = 0.0
            time.sleep(1.0) # wait for settlement
            
    # Case C: We have no position, and have an active target signal (Buy or Sell)
    if pos_side is None and target_side is not None:
        # Check if we have enough margin
        # At 10x leverage, minimum position of 0.001 BTC (~60 USD) requires ~6.0 USDT margin.
        margin_required = 6.0
        if usdt_bal >= margin_required:
            print(f"Opening leveraged 10x Futures position ({target_side}) for 0.001 BTC...")
            order_res = open_futures_position(target_side, size=0.001, sl_price=sl, tp_price=tp)
            if order_res and order_res.get("retCode") == 0:
                message = f"BTC/USDT 10x Futures Position OPENED ({target_side} size=0.001 BTC) at price {current_price:,.1f} USDT.\nStop Loss (SL): {sl:,.1f} USDT\nTake Profit (TP): {tp:,.1f} USDT"
                send_alert(f"[AI Bot] BTC/USDT Futures Position Opened - {action}", message)
            else:
                print("Failed to open futures position.")
        else:
            print(f"Insufficient available margin ({usdt_bal:.2f} USDT). Required: {margin_required} USDT.")

if __name__ == "__main__":
    run_execution_loop()
