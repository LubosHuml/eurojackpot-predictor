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
    Fetches Unified wallet balances for BTC and USDT.
    """
    timestamp = str(int(time.time() * 1000))
    params = "accountType=UNIFIED"
    
    # Signature for GET is computed using query params
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
            btc = 0.0
            for c in coins:
                wb = float(c.get("walletBalance", 0.0) or 0.0)
                im = float(c.get("totalPositionIM", 0.0) or 0.0)
                order_im = float(c.get("totalOrderIM", 0.0) or 0.0)
                avail = max(0.0, wb - im - order_im)
                
                if c["coin"] == "USDT":
                    usdt = avail
                elif c["coin"] == "BTC":
                    btc = wb
            return usdt, btc
    except Exception as e:
        print(f"Error fetching balances: {e}")
    return 0.0, 0.0

def place_market_order(side, qty):
    """
    Places a Spot Market order on Bybit.
    For Spot Market Buy: qty is the amount of USDT to spend.
    For Spot Market Sell: qty is the amount of BTC to sell.
    """
    timestamp = str(int(time.time() * 1000))
    url = "https://api.bybit.com/v5/order/create"
    
    payload = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": side,
        "orderType": "Market",
        "qty": f"{qty:.8f}" if side == "Sell" else f"{qty:.2f}"
    }
    
    payload_str = json.dumps(payload)
    headers = get_bybit_headers(payload_str, timestamp)
    
    try:
        res = requests.post(url, data=payload_str, headers=headers)
        data = res.json()
        print(f"Bybit Order Create Response ({side} qty={qty}):", data)
        return data
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def send_alert(subject, message):
    """
    Sends email alert to the user.
    """
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
    trade_path = os.path.join(crypto_dir, "crypto_active_trade.json")
    
    if not os.path.exists(prediction_path):
        print("No prediction cache found.")
        return
        
    with open(prediction_path, "r") as f:
        pred_data = json.load(f)
        
    action = pred_data.get("action", "WAIT / NEUTRAL")
    current_price = pred_data.get("price")
    sl = pred_data.get("stop_loss")
    tp = pred_data.get("take_profit")
    
    usdt_bal, btc_bal = get_wallet_balances()
    print(f"Balances - USDT: {usdt_bal:.2f}, BTC: {btc_bal:.8f}")
    
    # 1. Check if we have an active trade saved
    active_trade = None
    if os.path.exists(trade_path):
        try:
            with open(trade_path, "r") as f:
                active_trade = json.load(f)
        except Exception:
            pass
            
    # 2. Check if we need to Exit (SL or TP hit)
    if active_trade is not None:
        trade_sl = active_trade["stop_loss"]
        trade_tp = active_trade["take_profit"]
        qty_btc = active_trade["qty_btc"]
        
        print(f"Active trade monitored - SL: {trade_sl}, TP: {trade_tp}, Hold: {qty_btc:.6f} BTC")
        
        # Check current price boundaries
        # We can fetch the current price or use current_price from Bybit K-line
        exit_trade = False
        exit_reason = ""
        
        if current_price >= trade_tp:
            exit_trade = True
            exit_reason = "TAKE PROFIT (TP)"
        elif current_price <= trade_sl:
            exit_trade = True
            exit_reason = "STOP LOSS (SL)"
        elif action != "BUY / LONG":
            exit_trade = True
            exit_reason = "SIGNAL SHIFT"
            
        if exit_trade:
            print(f"Exit triggered: {exit_reason} at price {current_price:.1f}")
            # Sell all held BTC
            # Get actual BTC balance to sell to ensure we don't try to sell more than we have
            sell_qty = min(btc_bal, qty_btc)
            if sell_qty > 0.00005: # Bybit spot minimum sell size
                order_res = place_market_order("Sell", sell_qty)
                if order_res and order_res.get("retCode") == 0:
                    message = f"BTC/USDT position CLOSED via {exit_reason} at price {current_price:,.1f} USDT. Sold {sell_qty:.6f} BTC."
                    send_alert(f"[AI Bot] BTC/USDT Trade Closed - {exit_reason}", message)
            
            # Delete active trade record
            if os.path.exists(trade_path):
                os.remove(trade_path)
            return

    # 3. Check if we need to Enter (signal is BUY / LONG and no active trade)
    if action == "BUY / LONG" and active_trade is None:
        # Check if we have USDT to buy BTC
        # Minimum purchase is 2.0 USDT, we use all USDT minus a tiny buffer (0.1 USDT)
        buy_qty = usdt_bal - 0.1
        if buy_qty >= 2.0:
            print(f"Triggering BUY / LONG order for {buy_qty:.2f} USDT...")
            order_res = place_market_order("Buy", buy_qty)
            if order_res and order_res.get("retCode") == 0:
                # Find how much BTC we bought (can query from account or approximate based on price)
                # Bybit spot order response may show orderId.
                # To be simple and robust: we wait 1 second and check the new BTC balance!
                time.sleep(1.5)
                _, new_btc_bal = get_wallet_balances()
                btc_bought = new_btc_bal - btc_bal
                
                if btc_bought > 0.00005:
                    new_trade = {
                        "entry_price": current_price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "qty_btc": btc_bought,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    with open(trade_path, "w") as f:
                        json.dump(new_trade, f)
                        
                    message = f"BTC/USDT position OPENED at price {current_price:,.1f} USDT.\nBought: {btc_bought:.6f} BTC\nStop Loss (SL): {sl:,.1f} USDT\nTake Profit (TP): {tp:,.1f} USDT"
                    send_alert("[AI Bot] BTC/USDT Trade Opened - BUY / LONG", message)
                else:
                    print("Could not verify BTC purchase balance.")
            else:
                print("Failed to place Buy order.")

if __name__ == "__main__":
    run_execution_loop()
