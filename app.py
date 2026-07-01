import os
import sys
import json
import threading
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

import database
import scraper
import features
import generator
import train
import backtest
import sportka_database
import sportka_features
import sportka_models
import sportka_train
import sportka_backtest

def start_crypto_scheduler():
    """Starts a 24/7 daemon thread that runs crypto predictions and trade executions every 5 minutes."""
    def loop():
        import time
        import socket
        import subprocess
        
        # Try to bind to port 5081 to ensure only one worker process runs the loop
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', 5081))
            s.listen(1)
        except socket.error:
            print("[Scheduler] Port 5081 is already bound. Another Gunicorn worker is running the scheduler. Exiting thread.")
            return
            
        print("[Scheduler] Crypto 24/7 background loop started (Active Worker).")
        # Initial sleep buffer to let Flask server bind and initialize
        time.sleep(10)
        
        last_lotto_sync = 0
        
        while True:
            current_time = time.time()
            
            # Run automatic lottery database sync and ticket evaluation every 6 hours (21600 seconds)
            if current_time - last_lotto_sync > 21600:
                try:
                    print("[Scheduler] Running automatic lottery database sync and ticket evaluation...")
                    # Update Eurojackpot draws
                    inserted = scraper.update_database(force_all=False)
                    evaluate_pending_tickets()
                    if inserted > 0:
                        print(f"[Scheduler] Found {inserted} new Eurojackpot draws. Triggering background retraining...")
                        t_train = threading.Thread(target=background_training_thread)
                        t_train.daemon = True
                        t_train.start()
                    
                    # Update Sportka draws from CSV
                    sportka_inserted = sportka_database.load_csv_data() or 0
                    evaluate_sportka_pending_tickets()
                    if sportka_inserted > 0:
                        print(f"[Scheduler] Found {sportka_inserted} new Sportka draws. Triggering background retraining...")
                        t_train = threading.Thread(target=background_sportka_training_thread)
                        t_train.daemon = True
                        t_train.start()
                        
                    last_lotto_sync = current_time
                except Exception as ex:
                    print(f"[Scheduler] Lotto automatic sync error: {ex}")
                    
            try:
                print("[Scheduler] Running price fetch and predictions (output_bridge.py)...")
                subprocess.run([sys.executable, "crypto/output_bridge.py"])
                print("[Scheduler] Checking and executing orders (executor.py)...")
                subprocess.run([sys.executable, "crypto/executor.py"])
            except Exception as e:
                print(f"[Scheduler] Loop execution error: {e}")
            # Run every 5 minutes (300 seconds)
            time.sleep(300)
            
    t = threading.Thread(target=loop)
    t.daemon = True
    t.start()

app = Flask(__name__, static_folder="static", template_folder="templates")

# Start 24/7 Bybit automation thread at module load time (supports Gunicorn workers)
start_crypto_scheduler()

# Global state for background training task
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "error": None,
    "last_success": None
}

MODEL_PATH = "eurojackpot_lstm_model.keras"
METRICS_PATH = "metrics.json"

from datetime import datetime, timedelta

def get_next_draw_date(latest_draw_date_str):
    try:
        dt = datetime.strptime(latest_draw_date_str, "%Y-%m-%d")
        if dt.weekday() == 1: # Tuesday
            next_dt = dt + timedelta(days=3)
        else: # Friday (or fallback)
            next_dt = dt + timedelta(days=4)
        return next_dt.strftime("%Y-%m-%d")
    except Exception:
        return "N/A"

def get_draw_for_date(date_str):
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT num1, num2, num3, num4, num5, euro1, euro2 FROM draws WHERE date = ?", (date_str,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return [row[0], row[1], row[2], row[3], row[4]], [row[5], row[6]]
    except Exception:
        pass
    return None

def evaluate_pending_tickets():
    """Scans all registered tickets that are pending evaluation and checks if their draw results are now available."""
    try:
        tickets = database.get_all_tickets()
        for t in tickets:
            if t['prize_tier'] == "Pending":
                res = get_draw_for_date(t['draw_date'])
                if res:
                    drawn_main, drawn_euro = res
                    matched_m = list(set(t['main_nums']) & set(drawn_main))
                    matched_e = list(set(t['euro_nums']) & set(drawn_euro))
                    prize_tier = f"{len(matched_m)}+{len(matched_e)}"
                    database.update_ticket_results(t['draw_date'], t['row_id'], matched_m, matched_e, prize_tier)
    except Exception as e:
        print(f"Error evaluating pending tickets: {e}")

import sqlite3

def get_sportka_next_draw_date(latest_draw_date_str):
    if not latest_draw_date_str:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        dt = datetime.strptime(latest_draw_date_str, "%Y-%m-%d")
        for i in range(1, 8):
            next_dt = dt + timedelta(days=i)
            # Sportka draws are on Wed (2), Fri (4), Sun (6)
            if next_dt.weekday() in [2, 4, 6]:
                return next_dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")

def get_sportka_db_status():
    sportka_database.init_db()
    draws = sportka_database.get_all_draws()
    latest_date = None
    if draws:
        latest_date = draws[-1]["date"]
        
    model_exists = os.path.exists("sportka_lstm_model.keras")
    scalers_exist = (
        os.path.exists("sportka_scaler_x.joblib") and
        os.path.exists("sportka_scaler_sum.joblib") and
        os.path.exists("sportka_scaler_counts.joblib")
    )
    
    next_draw_date = get_sportka_next_draw_date(latest_date)
    tickets = sportka_database.get_all_tickets()
    next_tickets = [t for t in tickets if t['draw_date'] == next_draw_date]
    is_registered = len(next_tickets) > 0
    registered_type = "standard"
    if is_registered:
        if "System" in next_tickets[0]["profile"]:
            registered_type = "system"
            
    return {
        "total_draws": len(draws),
        "latest_draw_date": latest_date if latest_date else "N/A",
        "next_draw_date": next_draw_date,
        "ticket_registered": is_registered,
        "registered_type": registered_type,
        "model_trained": model_exists and scalers_exist,
        "training_state": "Ready" if (model_exists and scalers_exist) else "Untrained",
        "error": None
    }

def evaluate_sportka_pending_tickets():
    try:
        tickets = sportka_database.get_all_tickets()
        conn = sqlite3.connect("sportka.db")
        cursor = conn.cursor()
        for t in tickets:
            if t['prize_tier'] == "Pending":
                cursor.execute("SELECT num1, num2, num3, num4, num5, num6, supplementary FROM draws WHERE draw_date = ?", (t['draw_date'],))
                rows = cursor.fetchall()
                if rows:
                    best_matches = []
                    best_tier = "0"
                    
                    for r in rows:
                        drawn_nums = [r[0], r[1], r[2], r[3], r[4], r[5]]
                        supp = r[6]
                        
                        matched = list(set(t['nums']) & set(drawn_nums))
                        has_supp = 1 if (supp in t['nums']) else 0
                        
                        curr_tier = "0"
                        if len(matched) == 6:
                            curr_tier = "6"
                        elif len(matched) == 5 and has_supp:
                            curr_tier = "5+1"
                        elif len(matched) == 5:
                            curr_tier = "5"
                        elif len(matched) == 4:
                            curr_tier = "4"
                        elif len(matched) == 3:
                            curr_tier = "3"
                        elif len(matched) == 2:
                            curr_tier = "2"
                        elif len(matched) == 1:
                            curr_tier = "1"
                            
                        tier_scores = {"6": 8, "5+1": 7, "5": 6, "4": 5, "3": 4, "2": 3, "1": 2, "0": 0}
                        best_score = tier_scores.get(best_tier, 0)
                        curr_score = tier_scores.get(curr_tier, 0)
                        
                        if curr_score > best_score:
                            best_tier = curr_tier
                            best_matches = matched
                            
                    sportka_database.update_ticket_results(t['draw_date'], t['row_id'], best_matches, best_tier)
        conn.close()
    except Exception as e:
        print(f"Error evaluating pending Sportka tickets: {e}")

def get_db_status():
    """Helper to check the state of the database and model files."""
    database.init_db()
    latest_date = database.get_latest_draw_date()
    draws = database.get_all_draws()
    
    model_exists = os.path.exists(MODEL_PATH)
    scalers_exist = (
        os.path.exists(train.SCALER_X_PATH) and
        os.path.exists(train.SCALER_SUM_PATH) and
        os.path.exists(train.SCALER_COUNTS_PATH)
    )
    
    next_draw_date = get_next_draw_date(latest_date)
    tickets = database.get_all_tickets()
    next_tickets = [t for t in tickets if t['draw_date'] == next_draw_date]
    is_registered = len(next_tickets) > 0
    registered_type = "standard"
    if is_registered:
        if "System" in next_tickets[0]["profile"]:
            registered_type = "system"
            
    return {
        "total_draws": len(draws),
        "latest_draw_date": latest_date if latest_date else "N/A",
        "next_draw_date": next_draw_date,
        "ticket_registered": is_registered,
        "registered_type": registered_type,
        "model_trained": model_exists and scalers_exist,
        "training_state": "Training" if training_status["is_training"] else ("Ready" if (model_exists and scalers_exist) else "Untrained"),
        "error": training_status["error"]
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    game = request.args.get("game", "eurojackpot")
    if game == "sportka":
        status = get_sportka_db_status()
    else:
        status = get_db_status()
    return jsonify(status)

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    game = request.args.get("game", "eurojackpot")
    metrics_path = "sportka_metrics.json" if game == "sportka" else METRICS_PATH
    
    # If pre-computed metrics exist and have history_hits, load them
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            if "history_hits" in metrics:
                return jsonify(metrics)
        except Exception as e:
            pass
            
    # Otherwise run a quick evaluation on the fly if model exists
    if game == "sportka":
        status = get_sportka_db_status()
        if not status["model_trained"]:
            return jsonify({"error": "Model is not trained yet."}), 400
            
        try:
            model = tf.keras.models.load_model("sportka_lstm_model.keras")
            scaler_x = joblib.load("sportka_scaler_x.joblib")
            scaler_sum = joblib.load("sportka_scaler_sum.joblib")
            scaler_counts = joblib.load("sportka_scaler_counts.joblib")
            
            draws = sportka_database.get_all_draws()
            df_features = sportka_features.compute_draw_features(draws)
            data_dict = sportka_features.generate_sequences(df_features, window_size=10)
            
            metrics = sportka_backtest.run_backtest(
                model, data_dict, scaler_x, scaler_sum, scaler_counts, val_split=50
            )
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
                
            return jsonify(metrics)
        except Exception as e:
            return jsonify({"error": f"Failed to compute Sportka metrics: {str(e)}"}), 500
    else:
        status = get_db_status()
        if not status["model_trained"]:
            return jsonify({"error": "Model is not trained yet."}), 400
            
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler_x = joblib.load(train.SCALER_X_PATH)
            scaler_sum = joblib.load(train.SCALER_SUM_PATH)
            scaler_counts = joblib.load(train.SCALER_COUNTS_PATH)
            
            draws = database.get_all_draws()
            df_features = features.compute_draw_features(draws)
            data_dict = features.generate_sequences(df_features, window_size=10)
            
            metrics = backtest.run_backtest(
                model, data_dict, scaler_x, scaler_sum, scaler_counts, val_split=50
            )
            
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
                
            return jsonify(metrics)
        except Exception as e:
            return jsonify({"error": f"Failed to compute metrics: {str(e)}"}), 500

def generate_sportka_bets(main_probs, pred_sum, pred_counts, temperature=1.0, count=5):
    temp = max(temperature, 1e-6)
    x = np.log(np.clip(main_probs, 1e-12, 1.0)) / temp
    e_x = np.exp(x - np.max(x))
    p = e_x / np.sum(e_x)
    
    target_even = int(np.clip(np.round(pred_counts[0]), 0, 6))
    target_low = int(np.clip(np.round(pred_counts[2]), 0, 6))
    
    bets = []
    attempts = 0
    max_attempts = 5000
    relaxed = False
    
    while len(bets) < count and attempts < max_attempts:
        attempts += 1
        choices = np.arange(1, 50)
        nums = sorted(list(np.random.choice(choices, size=6, replace=False, p=p)))
        
        if nums in bets:
            continue
            
        cand_sum = sum(nums)
        cand_even = sum(1 for x in nums if x % 2 == 0)
        cand_low = sum(1 for x in nums if 1 <= x <= 24)
        
        sum_tolerance = 15 if not relaxed else 25
        sum_ok = abs(cand_sum - pred_sum) <= sum_tolerance
        
        if not relaxed:
            even_ok = cand_even == target_even
            low_ok = cand_low == target_low
        else:
            even_ok = abs(cand_even - target_even) <= 1
            low_ok = abs(cand_low - target_low) <= 1
            
        if sum_ok and even_ok and low_ok:
            bets.append(nums)
            
        if attempts == max_attempts // 2:
            relaxed = True
            
    while len(bets) < count:
        choices = np.arange(1, 50)
        nums = sorted(list(np.random.choice(choices, size=6, replace=False, p=p)))
        if nums not in bets:
            bets.append(nums)
            
    return bets

@app.route("/api/predictions/hybrid", methods=["GET"])
def api_predictions_hybrid():
    """Returns the ensemble Hybrid LSTM-Quantum predictions for Eurojackpot."""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "crypto"))
        import quantum_lstm_hybrid
        import importlib
        importlib.reload(quantum_lstm_hybrid)
        res = quantum_lstm_hybrid.run_hybrid_predictions(count=6)
        if "error" in res:
            return jsonify(res), 500
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    game = request.args.get("game", "eurojackpot")
    bets_count = request.args.get("count", default=5, type=int)
    
    if game == "sportka":
        status = get_sportka_db_status()
        if not status["model_trained"]:
            return jsonify({"error": "Model is not trained yet."}), 400
            
        try:
            model = tf.keras.models.load_model("sportka_lstm_model.keras")
            scaler_x = joblib.load("sportka_scaler_x.joblib")
            scaler_sum = joblib.load("sportka_scaler_sum.joblib")
            scaler_counts = joblib.load("sportka_scaler_counts.joblib")
            
            draws = sportka_database.get_all_draws()
            df_features = sportka_features.compute_draw_features(draws)
            
            window_size = 10
            feature_cols = [
                'mean', 'std', 'median', 'sum', 'product_diff',
                'even_count', 'odd_count', 'low_count', 'high_count',
                'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
            ]
            num_features = df_features[feature_cols].values
            main_nums = df_features[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
            
            X_num_last = np.expand_dims(num_features[-window_size:], axis=0).astype(np.float32)
            X_main_last = np.expand_dims(main_nums[-window_size:], axis=0).astype(np.int32)
            
            X_num_scaled = sportka_train.transform_3d(scaler_x, X_num_last)
            
            pred_sum_scaled, pred_counts_scaled, pred_main_probs = model.predict(
                [X_num_scaled, X_main_last],
                verbose=0
            )
            
            next_pred_sum = float(scaler_sum.inverse_transform(pred_sum_scaled)[0, 0])
            next_pred_counts = scaler_counts.inverse_transform(pred_counts_scaled)[0]
            next_main_probs = pred_main_probs[0]
            
            import hashlib
            latest_date = status["latest_draw_date"]
            seed_src = latest_date if latest_date else "default_seed_key"
            seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            
            bets_c = generate_sportka_bets(next_main_probs, next_pred_sum, next_pred_counts, temperature=0.2, count=2)
            bets_b = generate_sportka_bets(next_main_probs, next_pred_sum, next_pred_counts, temperature=1.0, count=2)
            bets_u = generate_sportka_bets(next_main_probs, next_pred_sum, next_pred_counts, temperature=2.0, count=2)
            
            raw_bets = [
                (bets_c[0], "Conservative"),
                (bets_c[1], "Conservative"),
                (bets_b[0], "Balanced"),
                (bets_b[1], "Balanced"),
                (bets_u[0], "Unique"),
                (bets_u[1], "Unique")
            ]
            
            max_p_m = float(np.max(next_main_probs))
            
            bets_list = []
            for idx, (comb, profile) in enumerate(raw_bets):
                m_nums = [int(x) for x in comb]
                mean_p_m = float(np.mean([next_main_probs[x-1] for x in m_nums]))
                confidence = (mean_p_m / max_p_m) * 100
                bets_list.append({
                    "id": idx + 1,
                    "profile": profile,
                    "main_nums": m_nums,
                    "euro_nums": [],
                    "confidence": round(confidence, 1)
                })
                
            # System bets: 8-number wheel
            top8_idx = np.argsort(next_main_probs)[-8:]
            top8_nums = sorted([int(x + 1) for x in top8_idx])
            
            wheel_indices = [
                [1, 2, 3, 4, 6, 7],
                [0, 1, 3, 4, 5, 6],
                [0, 1, 2, 3, 5, 7],
                [0, 2, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 3, 4, 6, 7]
            ]
            
            system_bets_list = []
            for idx, row in enumerate(wheel_indices):
                m_nums = sorted([top8_nums[k] for k in row])
                mean_p_m = float(np.mean([next_main_probs[x-1] for x in m_nums]))
                confidence = (mean_p_m / max_p_m) * 100
                system_bets_list.append({
                    "id": idx + 1,
                    "profile": f"System Row {idx+1}",
                    "main_nums": m_nums,
                    "euro_nums": [],
                    "confidence": round(confidence, 1)
                })
                
            return jsonify({
                "estimated_sum": next_pred_sum,
                "estimated_counts": {
                    "even": float(next_pred_counts[0]),
                    "odd": float(next_pred_counts[1]),
                    "low": float(next_pred_counts[2]),
                    "high": float(next_pred_counts[3])
                },
                "bets": bets_list,
                "system_bets": system_bets_list
            })
        except Exception as e:
            return jsonify({"error": f"Failed to generate predictions for Sportka: {str(e)}"}), 500
            
    else:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "crypto"))
            import quantum_lstm_hybrid
            import importlib
            importlib.reload(quantum_lstm_hybrid)
            res = quantum_lstm_hybrid.run_hybrid_predictions(count=6)
            if "error" in res:
                return jsonify(res), 500
                
            # Add estimated fields to make it fully compatible with legacy consumers if any
            res["estimated_sum"] = 125.0
            res["estimated_counts"] = {"even": 2.5, "odd": 2.5, "low": 2.5, "high": 2.5}
            res["system_bets"] = res["bets"] # fallback
            
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": f"Failed to generate hybrid predictions: {str(e)}"}), 500

@app.route("/api/update", methods=["POST"])
def api_update():
    """Triggers an incremental scraper sync to fetch the latest draw results."""
    try:
        inserted = scraper.update_database(force_all=False)
        # Scan and evaluate any pending tickets with the updated data
        evaluate_pending_tickets()
        # Clear metrics cache on database change to force re-evaluation
        if inserted > 0 and os.path.exists(METRICS_PATH):
            os.remove(METRICS_PATH)
            
        # Update Sportka database from CSV and evaluate pending tickets
        sportka_inserted = sportka_database.load_csv_data() or 0
        evaluate_sportka_pending_tickets()
        if os.path.exists("sportka_metrics.json"):
            try:
                os.remove("sportka_metrics.json")
            except Exception:
                pass
                
        # Trigger automatic background retraining if new draws were imported
        if inserted > 0:
            if not training_status["is_training"]:
                thread = threading.Thread(target=background_training_thread)
                thread.daemon = True
                thread.start()
                print("Eurojackpot auto-retrain triggered.")
                
        if sportka_inserted > 0:
            if not training_status["is_training"]:
                thread = threading.Thread(target=background_sportka_training_thread)
                thread.daemon = True
                thread.start()
                print("Sportka auto-retrain triggered.")
            
        return jsonify({"success": True, "inserted": inserted, "sportka_inserted": sportka_inserted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/lotto/quantum", methods=["GET"])
def api_lotto_quantum():
    game = request.args.get("game", "eurojackpot")
    if game not in ["eurojackpot", "sportka"]:
        return jsonify({"error": "Invalid game type."}), 400
        
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "crypto"))
        import quantum_lotto
        # Reload to prevent stale cache if database updates
        import importlib
        importlib.reload(quantum_lotto)
        res = quantum_lotto.run_quantum_prediction(game)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/crypto/live", methods=["GET"])
def api_crypto_live():
    prediction_path = os.path.join("crypto", "crypto_live_prediction.json")
    if not os.path.exists(prediction_path):
        import subprocess
        try:
            subprocess.run([sys.executable, "crypto/output_bridge.py"], check=True)
        except Exception as e:
            return jsonify({"error": f"Failed to generate prediction: {str(e)}"}), 500
            
    try:
        with open(prediction_path, "r") as f:
            data = json.load(f)
            
        updated_at_str = data.get("updated_at")
        if updated_at_str:
            try:
                updated_at = datetime.strptime(updated_at_str, "%Y-%m-%d %H:%M:%S")
                # Refresh if older than 10 minutes (600 seconds)
                if (datetime.now() - updated_at).total_seconds() > 600:
                    def refresh_in_bg():
                        import subprocess
                        try:
                            subprocess.run([sys.executable, "crypto/output_bridge.py"])
                            subprocess.run([sys.executable, "crypto/executor.py"])
                        except Exception:
                            pass
                    threading.Thread(target=refresh_in_bg).start()
            except Exception:
                pass
                
        # Trigger trade monitoring and execution asynchronously
        def run_executor_bg():
            import subprocess
            try:
                subprocess.run([sys.executable, "crypto/executor.py"])
            except Exception:
                pass
        threading.Thread(target=run_executor_bg).start()
                
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/crypto/plan", methods=["GET"])
def api_crypto_plan():
    plan_path = "crypto_10y_plan.json"
    if not os.path.exists(plan_path):
        return jsonify({"error": "Plan file not found."}), 404
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/crypto/plan/update", methods=["POST"])
def api_crypto_plan_update():
    plan_path = "crypto_10y_plan.json"
    if not os.path.exists(plan_path):
        return jsonify({"error": "Plan file not found."}), 404
    try:
        req_data = request.get_json() or {}
        index = req_data.get("index")
        actual_val = req_data.get("actual_balance")
        
        if index is None:
            return jsonify({"error": "Missing index parameter."}), 400
            
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
            
        updated = False
        for item in plan:
            if item.get("index") == int(index):
                item["actual_balance"] = float(actual_val) if actual_val is not None else None
                updated = True
                break
                
        if not updated:
            return jsonify({"error": f"Item with index {index} not found."}), 404
            
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4, ensure_ascii=False)
            
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def background_training_thread():
    """Background training worker to avoid blocking Flask web process."""
    global training_status
    try:
        training_status["is_training"] = True
        training_status["error"] = None
        
        print("Background training started...")
        # Scrape and update DB
        scraper.update_database(force_all=False)
        
        # Compute features
        draws = database.get_all_draws()
        df_features = features.compute_draw_features(draws)
        
        # Sequences (w=10)
        data_dict = features.generate_sequences(df_features, window_size=10)
        
        # Train & Prune (using 30 epochs for fast online execution)
        model = train.train_and_prune(
            data_dict=data_dict,
            window_size=10,
            lstm_units=64,
            learning_rate=1e-3,
            epochs=30,
            verbose=0
        )
        
        # Save model
        model.save(MODEL_PATH)
        
        # Force metric cache clearing
        if os.path.exists(METRICS_PATH):
            os.remove(METRICS_PATH)
            
        training_status["last_success"] = datetime.now().isoformat()
        print("Background training completed successfully.")
    except Exception as e:
        training_status["error"] = str(e)
        print(f"Background training failed: {e}")
    finally:
        training_status["is_training"] = False

def background_sportka_training_thread():
    """Background training worker to avoid blocking Flask web process for Sportka."""
    global training_status
    try:
        training_status["is_training"] = True
        training_status["error"] = None
        
        print("Background Sportka training started...")
        # Scrape and update DB
        sportka_database.load_csv_data()
        
        # Compute features
        draws = sportka_database.get_all_draws()
        df_features = sportka_features.compute_draw_features(draws)
        
        # Sequences (w=10)
        data_dict = sportka_features.generate_sequences(df_features, window_size=10)
        
        # Train & Prune (using 25 epochs)
        sportka_train.train_and_prune(
            data_dict=data_dict,
            window_size=10,
            epochs=25,
            verbose=0
        )
        
        # Force metric cache clearing
        if os.path.exists("sportka_metrics.json"):
            try:
                os.remove("sportka_metrics.json")
            except Exception:
                pass
            
        training_status["last_success"] = datetime.now().isoformat()
        print("Background Sportka training completed successfully.")
    except Exception as e:
        training_status["error"] = str(e)
        print(f"Background Sportka training failed: {e}")
    finally:
        training_status["is_training"] = False

@app.route("/api/train", methods=["POST"])
def api_train():
    """Launches the background training thread if not already running."""
    global training_status
    if training_status["is_training"]:
        return jsonify({"error": "Training is already in progress."}), 400
        
    game = request.args.get("game", "eurojackpot")
    if game == "sportka":
        thread = threading.Thread(target=background_sportka_training_thread)
    else:
        thread = threading.Thread(target=background_training_thread)
        
    thread.daemon = True
    thread.start()
    return jsonify({"success": True, "message": f"Model training for {game} initiated in the background."})

@app.route("/api/tickets", methods=["GET"])
def api_tickets():
    game = request.args.get("game", "eurojackpot")
    try:
        if game == "sportka":
            tickets = sportka_database.get_all_tickets()
        else:
            tickets = database.get_all_tickets()
        return jsonify(tickets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tickets/register", methods=["POST"])
def api_tickets_register():
    req_data = request.get_json(silent=True) or {}
    game = req_data.get("game", "eurojackpot")
    ticket_type = req_data.get("type", "standard")
    
    if game == "sportka":
        status = get_sportka_db_status()
        if not status["model_trained"]:
            return jsonify({"error": "Model is not trained yet."}), 400
            
        try:
            model = tf.keras.models.load_model("sportka_lstm_model.keras")
            scaler_x = joblib.load("sportka_scaler_x.joblib")
            scaler_sum = joblib.load("sportka_scaler_sum.joblib")
            scaler_counts = joblib.load("sportka_scaler_counts.joblib")
            
            latest_date = status["latest_draw_date"]
            next_draw_date = status["next_draw_date"]
            
            draws = sportka_database.get_all_draws()
            df_features = sportka_features.compute_draw_features(draws)
            
            window_size = 10
            feature_cols = [
                'mean', 'std', 'median', 'sum', 'product_diff',
                'even_count', 'odd_count', 'low_count', 'high_count',
                'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
            ]
            num_features = df_features[feature_cols].values
            main_nums = df_features[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values
            
            X_num_last = np.expand_dims(num_features[-window_size:], axis=0).astype(np.float32)
            X_main_last = np.expand_dims(main_nums[-window_size:], axis=0).astype(np.int32)
            
            X_num_scaled = sportka_train.transform_3d(scaler_x, X_num_last)
            
            pred_sum_scaled, pred_counts_scaled, pred_main_probs = model.predict(
                [X_num_scaled, X_main_last],
                verbose=0
            )
            
            next_pred_sum = float(scaler_sum.inverse_transform(pred_sum_scaled)[0, 0])
            next_pred_counts = scaler_counts.inverse_transform(pred_counts_scaled)[0]
            next_main_probs = pred_main_probs[0]
            
            import hashlib
            seed_src = latest_date if latest_date else "default_seed_key"
            seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            
            if ticket_type == "system":
                top8_idx = np.argsort(next_main_probs)[-8:]
                top8_nums = sorted([int(x + 1) for x in top8_idx])
                
                wheel_indices = [
                    [1, 2, 3, 4, 6, 7],
                    [0, 1, 3, 4, 5, 6],
                    [0, 1, 2, 3, 5, 7],
                    [0, 2, 4, 5, 6, 7],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 3, 4, 6, 7]
                ]
                
                raw_bets = []
                for idx, row in enumerate(wheel_indices):
                    m_nums = sorted([top8_nums[k] for k in row])
                    raw_bets.append((m_nums, f"System Row {idx+1}"))
            else:
                bets_c = generate_sportka_bets(next_main_probs, next_pred_sum, next_pred_counts, temperature=0.2, count=2)
                bets_b = generate_sportka_bets(next_main_probs, next_pred_sum, next_pred_counts, temperature=1.0, count=2)
                bets_u = generate_sportka_bets(next_main_probs, next_pred_sum, next_pred_counts, temperature=2.0, count=2)
                
                raw_bets = [
                    (bets_c[0], "Conservative"),
                    (bets_c[1], "Conservative"),
                    (bets_b[0], "Balanced"),
                    (bets_b[1], "Balanced"),
                    (bets_u[0], "Unique"),
                    (bets_u[1], "Unique")
                ]
                
            max_p_m = float(np.max(next_main_probs))
            
            for idx, (comb, profile) in enumerate(raw_bets):
                m_nums = [int(x) for x in comb]
                mean_p_m = float(np.mean([next_main_probs[x-1] for x in m_nums]))
                confidence = (mean_p_m / max_p_m) * 100
                
                sportka_database.save_ticket(next_draw_date, idx + 1, profile, m_nums, round(confidence, 1))
                
            return jsonify({"success": True, "draw_date": next_draw_date})
        except Exception as e:
            return jsonify({"error": f"Failed to register Sportka ticket: {str(e)}"}), 500
            
    else:
        status = get_db_status()
        if not status["model_trained"]:
            return jsonify({"error": "Model is not trained yet."}), 400
            
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "crypto"))
            import quantum_lstm_hybrid
            import importlib
            importlib.reload(quantum_lstm_hybrid)
            
            res = quantum_lstm_hybrid.run_hybrid_predictions(count=6)
            if "error" in res:
                return jsonify(res), 500
                
            next_draw_date = status["next_draw_date"]
            
            # Save the 6 hybrid bets to database
            for b in res["bets"]:
                database.save_ticket(
                    next_draw_date,
                    b["row_id"],
                    f"Hybrid Row {b['row_id']}",
                    b["main_nums"],
                    b["euro_nums"],
                    b["confidence"]
                )
                
            return jsonify({"success": True, "draw_date": next_draw_date})
        except Exception as e:
            return jsonify({"error": f"Failed to register hybrid tickets: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5080))
    app.run(host="0.0.0.0", port=port)
