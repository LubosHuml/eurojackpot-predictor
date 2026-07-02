import os
import sys
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import MetaTrader5 as mt5

project_path = "C:\\Users\\lubos\\Desktop\\bybit_meta"
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import crypto.features as features
import crypto.quantum_lotto as quantum_lotto

# Magic number to identify bot orders
BOT_MAGIC_NUMBER = 123456

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
        print(f"SMTP credentials missing. Alert logged locally: {subject}\n{message}")
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

def get_state_path():
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".bybit_ai_bot")
    if not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir)
        except Exception:
            return os.path.join(home_dir, ".mt5_trade_state.json")
    return os.path.join(config_dir, "mt5_trade_state.json")

def load_trade_state():
    state_path = get_state_path()
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_trade_state(state):
    state_path = get_state_path()
    try:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving trade state: {e}")

def get_active_position(symbol):
    """
    Checks if we have an active position for the symbol.
    Returns the first matching position object or None.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is not None and len(positions) > 0:
        # Check if the position has our magic number
        for pos in positions:
            if pos.magic == BOT_MAGIC_NUMBER:
                return pos
    return None

def close_mt5_position(position_ticket):
    pos = mt5.positions_get(ticket=position_ticket)
    if not pos:
        print(f"[MT5] Position ticket {position_ticket} not found.")
        return False
        
    pos = pos[0]
    symbol = pos.symbol
    volume = pos.volume
    pos_type = pos.type # 0 for Buy, 1 for Sell
    
    close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": close_type,
        "position": int(position_ticket),
        "price": price,
        "deviation": 20,
        "magic": BOT_MAGIC_NUMBER,
        "comment": "Close Quantum Reservoir position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        # Fallback to filling RETURN
        if result.retcode in [mt5.TRADE_RETCODE_INVALID_FILL, mt5.TRADE_RETCODE_FILL_CANNOT]:
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] Successfully closed position ticket {position_ticket}")
        return True
    else:
        print(f"[MT5] Failed to close position ticket {position_ticket}. Error: {result.comment}")
        return False

def open_mt5_position(symbol, side, volume, sl_price=0.0, tp_price=0.0):
    mt5.symbol_select(symbol, True)
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"[MT5] Symbol {symbol} not found.")
        return None
        
    order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if side == "Buy" else mt5.symbol_info_tick(symbol).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "sl": float(sl_price) if sl_price > 0 else 0.0,
        "tp": float(tp_price) if tp_price > 0 else 0.0,
        "deviation": 20,
        "magic": BOT_MAGIC_NUMBER,
        "comment": "Quantum Reservoir bot entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        if result.retcode in [mt5.TRADE_RETCODE_INVALID_FILL, mt5.TRADE_RETCODE_FILL_CANNOT]:
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] Opened {side} position for {symbol} (lots={volume}, ticket={result.order})")
        return result.order
    else:
        print(f"[MT5] Failed to open {side} position for {symbol}. Error: {result.comment}")
        return None

def fetch_mt5_bars(symbol, limit=50):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, limit)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    })
    return df

def calculate_atr(df, period=14):
    high = df["high"].values
    low = df["low"].values
    close_prev = df["close"].shift(1).values
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(period).mean().fillna(np.mean(tr)).values
    return atr[-1]

def run_mt5_execution_loop():
    if not mt5.initialize():
        print("[MT5] Could not initialize connection to MT5 terminal.")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        print("[MT5] Could not fetch account info.")
        mt5.shutdown()
        return
        
    balance = account_info.balance
    print(f"\n[MT5] Account Balance: {balance:.2f} USD | Leverage: 1:{account_info.leverage}")
    
    # Plan C Risk Allocation
    is_gold_unlocked = balance >= 350.0
    
    # Compounding lot sizes: 0.01 lot for every 200 USD balance
    lot_multiplier = max(1, int(balance // 200))
    base_lot_size = lot_multiplier * 0.01
    
    print(f"[MT5] Risk Settings - Lot Size Multiplier: {lot_multiplier}x | Base Lot: {base_lot_size:.2f} | Gold Unlocked: {is_gold_unlocked}")
    
    trade_state = load_trade_state()
    symbols = ["USDJPY", "NAS100", "XAUUSD"]
    
    for sym in symbols:
        if sym == "XAUUSD" and not is_gold_unlocked:
            print(f"[{sym}] Gold is currently LOCKED due to small account size. Skipping.")
            continue
            
        sym_lower = sym.lower()
        
        # Load scaler and model
        crypto_dir = os.path.join(project_path, "crypto")
        scaler_path = os.path.join(crypto_dir, f"crypto_scaler_{sym_lower}.joblib")
        model_path = os.path.join(crypto_dir, f"crypto_lstm_model_{sym_lower}.keras")
        
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            print(f"[{sym}] Scaler or Model files missing. Run train_mt5_assets.py first.")
            continue
            
        scaler = joblib.load(scaler_path)
        model = tf.keras.models.load_model(model_path)
        
        # Fetch last 50 bars to compute rolling indicators
        df = fetch_mt5_bars(sym, limit=50)
        if df is None or len(df) < 35:
            print(f"[{sym}] Failed to fetch sufficient bar data from MT5.")
            continue
            
        # Calculate indicators and drop NaNs
        df_indicators = features.calculate_indicators(df)
        
        # We need the last 20 rows of indicators to construct the sequence window for LSTM
        if len(df_indicators) < 20:
            print(f"[{sym}] Insufficient clean indicator rows.")
            continue
            
        df_window = df_indicators.tail(20).reset_index(drop=True)
        
        close_sma10 = df_window["close_to_sma10"].values
        close_ema10 = df_window["close_to_ema10"].values
        sma10_sma30 = df_window["sma10_to_sma30"].values
        bb_pos = df_window["bb_position"].values
        rsi_vals = df_window["rsi"].values
        log_ret = df_window["log_return"].values
        vol = df_window["volatility"].values
        vol_chg = df_window["volume_change"].values
        
        u_res_a = np.column_stack([
            np.clip(close_sma10 * 20.0, -1.0, 1.0),
            np.clip(close_ema10 * 20.0, -1.0, 1.0),
            np.clip(sma10_sma30 * 20.0, -1.0, 1.0),
            np.clip(2.0 * bb_pos - 1.0, -1.0, 1.0)
        ])
        
        u_res_b = np.column_stack([
            np.clip(2.0 * rsi_vals - 1.0, -1.0, 1.0),
            np.clip(log_ret * 50.0, -1.0, 1.0),
            np.clip(vol * 200.0 - 1.0, -1.0, 1.0),
            np.clip(np.tanh(vol_chg), -1.0, 1.0)
        ])
        
        # Simulate Dual Reservoirs
        res_a = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
        res_b = quantum_lotto.QuantumReservoir(n_qubits=4, J_coeff=0.5, h_field=1.0, epsilon=0.1)
        
        q_feats_a = []
        q_feats_b = []
        for idx in range(len(df_window)):
            q_feats_a.append(res_a.step(u_res_a[idx]))
            q_feats_b.append(res_b.step(u_res_b[idx]))
            
        q_feats_a = np.array(q_feats_a)
        q_feats_b = np.array(q_feats_b)
        
        classical_feats = df_window[[
            "close_to_sma10", "close_to_ema10", "sma10_to_sma30",
            "bb_position", "rsi", "log_return", "volatility", "volume_change"
        ]].values
        
        total_feats = np.column_stack([classical_feats, q_feats_a, q_feats_b])
        
        # Scale and reshape to (1, 20, 32)
        X_flat = total_feats.reshape(-1, total_feats.shape[1])
        X_scaled_flat = scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(1, 20, total_feats.shape[1])
        
        # Run prediction
        prediction = float(model.predict(X_scaled)[0][0])
        current_price = df_window["close"].iloc[-1]
        last_candle_time = str(df_window["timestamp"].iloc[-1])
        
        # Determine signals
        action = "WAIT / NEUTRAL"
        target_side = None
        if prediction > 0.53:
            action = "BUY / LONG"
            target_side = "Buy"
        elif prediction < 0.47:
            action = "SELL / SHORT"
            target_side = "Sell"
            
        # Check MT5 positions
        pos = get_active_position(sym)
        pos_side = None
        pos_size = 0.0
        pos_ticket = None
        
        if pos is not None:
            pos_side = "Buy" if pos.type == mt5.POSITION_TYPE_BUY else "Sell"
            pos_size = pos.volume
            pos_ticket = pos.ticket
            
        print(f"[{sym}] Active Position - Side: {pos_side}, Size: {pos_size}, AI Action: {action} (Pred: {prediction:.3f})")
        
        # Case A: Correct position running
        if pos_side == target_side and pos_side is not None:
            print(f"[{sym}] Position aligns with target. No action.")
            continue
            
        # Case B: Incorrect position running (signal change or neutral wait)
        if pos_side is not None:
            print(f"[{sym}] Closing incorrect position (ticket={pos_ticket}) due to signal shift...")
            closed = close_mt5_position(pos_ticket)
            if closed:
                message = f"[MT5] {sym} position CLOSED (ticket={pos_ticket}, {pos_side} size={pos_size}) at price {current_price:.4f} due to AI signal shift."
                send_alert(f"[AI Bot] MT5 {sym} Position Closed", message)
                pos_side = None
                pos_size = 0.0
                time.sleep(1.0)
                
        # Case C: No position, open new position
        if pos_side is None and target_side is not None:
            # Re-entry block safety guardrail
            if trade_state.get(sym_lower) == last_candle_time:
                print(f"[{sym}] Already traded during this hourly candle window ({last_candle_time}). Skipping.")
                continue
                
            # Compute ATR and Stop Loss/Take Profit
            atr = calculate_atr(df, period=14)
            
            # Asset specific lot limits
            lot_size = base_lot_size
            if sym == "NAS100":
                # Nasdaq minimum is usually 0.1 lots or 1 lot depending on MT5 broker type.
                # In Pepperstone NAS100 minimum size is 0.1 lot or 1 lot index.
                # Let's adjust:
                symbol_info = mt5.symbol_info("NAS100")
                if symbol_info:
                    lot_size = max(symbol_info.volume_min, base_lot_size)
                    
            if target_side == "Buy":
                sl_price = current_price - 1.5 * atr
                tp_price = current_price + 2.0 * atr
            else:
                sl_price = current_price + 1.5 * atr
                tp_price = current_price - 2.0 * atr
                
            ticket = open_mt5_position(sym, target_side, lot_size, sl_price, tp_price)
            if ticket:
                # Save hour time to avoid re-entry
                trade_state[sym_lower] = last_candle_time
                save_trade_state(trade_state)
                
                message = f"[MT5] {sym} {target_side} Position OPENED (lots={lot_size}, ticket={ticket}) at price {current_price:.4f}.\nStop Loss: {sl_price:.4f}\nTake Profit: {tp_price:.4f}"
                send_alert(f"[AI Bot] MT5 {sym} Position Opened", message)
                
    mt5.shutdown()

if __name__ == "__main__":
    run_mt5_execution_loop()
