# Binance Altcoin Portfolio Strategy

This document outlines the quantitative research and backtesting results for expanding the AI Quantum Reservoir Trading Bot to a secondary exchange (Binance) to diversify capital and assets.

## 📊 Altcoin Backtest Performance (1,000 Hours)

Evaluated using the Dual 4-qubit Quantum Reservoir Computing (QRC) model with a Reward-to-Risk ratio of **1.33** (Take Profit = 2.0 * ATR, Stop Loss = 1.5 * ATR). 

*Note: The mathematical break-even win rate for this configuration is **42.9%**.*

| Asset (Symbol) | Win Rate (%) | Total Trades | Profit Factor | Max Loss Streak | Total Return (ATR units) | Role / Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **RUNEUSDT** | **63.83%** | 141 | **2.35** | 14 | **+103.50** | 🏆 **Primary Profit Generator (Sword)** |
| **ADAUSDT** | **54.10%** | 122 | **1.57** | 7 | **+48.00** | 🛡️ **Risk Stabilizer (Shield)** |
| **SUIUSDT** | **50.59%** | 170 | **1.37** | 18 | **+46.00** | 🚀 **Growth Accelerator (Rocket)** |
| **AVAXUSDT** | **48.67%** | 150 | **1.26** | 15 | **+30.50** | 👍 **Solid Diversifier (Alternative)** |
| **LINKUSDT** | **47.20%** | 125 | **1.19** | 14 | **+19.00** | 🆗 **Conservative Trend-Follower** |

---

## 🎯 Proposed Binance Portfolio Allocation (3 Coins)

To diversify risk away from the Bybit instance (running BTC, ETH, and SOL), the Binance account will trade a complementary set of high-performing altcoins.

### Asset Composition
1. **ADAUSDT (Cardano) - 16.67% Allocation**
   * *Rationale*: Highly stable under the QRC model. Extremely low drawdown streak (max 7 losses), ensuring smooth portfolio equity curve behavior.
2. **RUNEUSDT (THORChain) - 16.67% Allocation**
   * *Rationale*: Phenomenal trend momentum. Highest win rate (63.83%) and profit factor (2.35). Acts as the primary growth engine.
3. **SUIUSDT (Sui Network) - 16.67% Allocation**
   * *Rationale*: High-beta layer-1 blockchain. Highly active (170 trades) with clean wave trends, capturing significant directional swings.

*Remaining **50%** of the Binance USDT balance will remain in cash/free margin as a safety cushion against liquidation.*

---

## 🛠️ Step-by-Step Activation Guide (Phase 2)

When Bybit trading proves its multi-week profitability and you are ready to launch on Binance:

### 1. API Credentials Setup
* Secure API Key and API Secret from Binance with **Futures Trading enabled**.
* Add the credentials to the local environment variables or configuration file on the notebook.

### 2. Model Training for New Assets
* Run the training pipeline for the new symbols to generate Keras models and scalers:
  ```bash
  python crypto/train.py --symbol ADAUSDT
  python crypto/train.py --symbol RUNEUSDT
  python crypto/train.py --symbol SUIUSDT
  ```

### 3. Deploy Multi-Exchange Scheduler
* Update the background scheduler to manage two separate exchange clients (`BybitClient` and `BinanceClient`).
* Configure `executor.py` to route orders dynamically based on the exchange profile.
