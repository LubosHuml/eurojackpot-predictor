import os
import sys
import time
import hmac
import hashlib
import requests
import json

def get_bybit_signature(api_key, api_secret, params_str, timestamp):
    val = timestamp + api_key + "5000" + params_str
    return hmac.new(api_secret.encode('utf-8'), val.encode('utf-8'), hashlib.sha256).hexdigest()

def test_auth():
    api_key = "NhyIWck9muRVkEIf1I"
    api_secret = "hKAHzNihIOmSveESysf6xpVYpbhLVUljjIZ0"
    
    timestamp = str(int(time.time() * 1000))
    # Query Wallet Balance for SPOT/Unified account
    url = "https://api.bybit.com/v5/account/wallet-balance"
    
    # Query parameters
    params = "accountType=UNIFIED"
    
    signature = get_bybit_signature(api_key, api_secret, params, timestamp)
    
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{url}?{params}", headers=headers)
        res_json = response.json()
        print("Bybit Wallet Balance Response:")
        print(json.dumps(res_json, indent=2))
    except Exception as e:
        print(f"Error testing Bybit Auth: {e}")

if __name__ == "__main__":
    test_auth()
