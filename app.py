import os
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

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global state for background training task
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "error": None,
    "last_success": None
}

MODEL_PATH = "eurojackpot_lstm_model.keras"
METRICS_PATH = "metrics.json"

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
    
    return {
        "total_draws": len(draws),
        "latest_draw_date": latest_date if latest_date else "N/A",
        "model_trained": model_exists and scalers_exist,
        "training_state": "Training" if training_status["is_training"] else ("Ready" if (model_exists and scalers_exist) else "Untrained"),
        "error": training_status["error"]
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    status = get_db_status()
    return jsonify(status)

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    # If pre-computed metrics exist, load them
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            return jsonify(metrics)
        except Exception as e:
            pass
            
    # Otherwise run a quick evaluation on the fly if model exists
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
        # Default w=10
        data_dict = features.generate_sequences(df_features, window_size=10)
        
        metrics = backtest.run_backtest(
            model, data_dict, scaler_x, scaler_sum, scaler_counts, val_split=50
        )
        
        # Cache them
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f)
            
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Failed to compute metrics: {str(e)}"}), 500

@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    bets_count = request.args.get("count", default=5, type=int)
    status = get_db_status()
    if not status["model_trained"]:
        return jsonify({"error": "Model is not trained yet."}), 400
        
    try:
        # Load assets
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler_x = joblib.load(train.SCALER_X_PATH)
        scaler_sum = joblib.load(train.SCALER_SUM_PATH)
        scaler_counts = joblib.load(train.SCALER_COUNTS_PATH)
        
        # Compute features
        draws = database.get_all_draws()
        df_features = features.compute_draw_features(draws)
        
        # Last window input (w=10)
        window_size = 10
        feature_cols = [
            'mean', 'std', 'median', 'sum', 'product_diff',
            'even_count', 'odd_count', 'low_count', 'high_count',
            'cpr_pp', 'cpr_bc', 'cpr_tc', 'vwap'
        ]
        num_features = df_features[feature_cols].values
        main_nums = df_features[['num1', 'num2', 'num3', 'num4', 'num5']].values
        euro_nums = df_features[['euro1', 'euro2']].values
        
        X_num_last = np.expand_dims(num_features[-window_size:], axis=0).astype(np.float32)
        X_main_last = np.expand_dims(main_nums[-window_size:], axis=0).astype(np.int32)
        X_euro_last = np.expand_dims(euro_nums[-window_size:], axis=0).astype(np.int32)
        
        # Scale
        X_num_scaled = train.transform_3d(scaler_x, X_num_last)
        
        # Predict
        pred_sum_scaled, pred_counts_scaled, pred_main_probs, pred_euro_probs = model.predict(
            [X_num_scaled, X_main_last, X_euro_last],
            verbose=0
        )
        
        # Inverse scale
        next_pred_sum = float(scaler_sum.inverse_transform(pred_sum_scaled)[0, 0])
        next_pred_counts = scaler_counts.inverse_transform(pred_counts_scaled)[0]
        
        # Probabilities
        next_main_probs = pred_main_probs[0]
        next_euro_probs = pred_euro_probs[0]
        
        # Seed generator deterministically based on latest draw date to prevent re-generation on refresh
        import hashlib
        latest_date = database.get_latest_draw_date()
        seed_src = latest_date if latest_date else "default_seed_key"
        seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        
        # Generate exactly 6 bets: 2 Conservative (T=0.2), 2 Balanced (T=1.0), 2 Unique (T=2.0)
        bets_c = generator.generate_bets(next_main_probs, next_euro_probs, next_pred_sum, next_pred_counts, temperature=0.2, count=2)
        bets_b = generator.generate_bets(next_main_probs, next_euro_probs, next_pred_sum, next_pred_counts, temperature=1.0, count=2)
        bets_u = generator.generate_bets(next_main_probs, next_euro_probs, next_pred_sum, next_pred_counts, temperature=2.0, count=2)
        
        # Merge lists
        raw_bets = [
            (bets_c[0], "Conservative"),
            (bets_c[1], "Conservative"),
            (bets_b[0], "Balanced"),
            (bets_b[1], "Balanced"),
            (bets_u[0], "Unique"),
            (bets_u[1], "Unique")
        ]
        
        # Calculate max bounds for relative normalization
        max_p_m = float(np.max(next_main_probs))
        max_p_e = float(np.max(next_euro_probs))
        
        bets_list = []
        for idx, (comb, profile) in enumerate(raw_bets):
            m_nums = [int(x) for x in comb[0]]
            e_nums = [int(y) for y in comb[1]]
            
            # Confidence metric: compares values to absolute hottest numbers
            mean_p_m = float(np.mean([next_main_probs[x-1] for x in m_nums]))
            mean_p_e = float(np.mean([next_euro_probs[y-1] for y in e_nums]))
            
            confidence = (mean_p_m / max_p_m) * 60 + (mean_p_e / max_p_e) * 40
            
            bets_list.append({
                "id": idx + 1,
                "profile": profile,
                "main_nums": m_nums,
                "euro_nums": e_nums,
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
            "bets": bets_list
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate predictions: {str(e)}"}), 500

@app.route("/api/update", methods=["POST"])
def api_update():
    """Triggers an incremental scraper sync to fetch the latest draw results."""
    try:
        inserted = scraper.update_database(force_all=False)
        # Clear metrics cache on database change to force re-evaluation
        if inserted > 0 and os.path.exists(METRICS_PATH):
            os.remove(METRICS_PATH)
        return jsonify({"success": True, "inserted": inserted})
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

@app.route("/api/train", methods=["POST"])
def api_train():
    """Launches the background training thread if not already running."""
    global training_status
    if training_status["is_training"]:
        return jsonify({"error": "Training is already in progress."}), 400
        
    thread = threading.Thread(target=background_training_thread)
    thread.daemon = True
    thread.start()
    return jsonify({"success": True, "message": "Model training initiated in the background."})

if __name__ == "__main__":
    # Render binds to PORT environment variable, locally we default to 5080 to avoid port conflicts
    port = int(os.environ.get("PORT", 5080))
    # We disable debug mode for production deployments
    app.run(host="0.0.0.0", port=port)
