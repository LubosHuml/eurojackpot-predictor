# MetaTrader 5 (Pepperstone) Portfolio Strategy

This document outlines the quantitative research, backtesting results, and deployment plan for expanding the AI Quantum Reservoir Trading Bot to traditional markets via MetaTrader 5.

## 📊 Traditional Asset Backtest Performance (2 Years)

Evaluated using the Dual 4-qubit Quantum Reservoir Computing (QRC) model with a Reward-to-Risk ratio of **1.33** (Take Profit = 2.0 * ATR, Stop Loss = 1.5 * ATR). 

*Note: The mathematical break-even win rate for this configuration is **42.9%**.*

| Asset Class (Symbol) | Win Rate (%) | Total Trades | Profit Factor | Max Loss Streak | Total Return (ATR units) | Recommendation / Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **USD/JPY Forex** | **50.45%** | 1,005 | **1.36** | 25 | **+267.00** | 🏆 **Primary Yield Engine (Forex King)** |
| **Nasdaq 100 (QQQ)** | **47.52%** | 825 | **1.21** | 21 | **+134.50** | 🥇 **Primary Growth Engine (Tech Index)** |
| **EUR/USD Forex** | 45.70% | 3,385 | 1.12 | 60 | **+337.00** | 🚀 High-Frequency Generator (Backup) |
| **GBP/USD Forex** | 46.10% | 1,078 | 1.14 | 31 | **+122.50** | 👍 Solid Trend Follower |
| **Gold (XAUUSD)** | 45.01% | 1,293 | 1.09 | 22 | **+97.50** | 🛡️ **Risk Stabilizer (Safe Haven)** |
| **Natural Gas (NG)** | 44.79% | 1,536 | 1.08 | 29 | **+104.00** | 💨 Volatile Commodity Alternative |
| **Silver (SI)** | 44.42% | 1,317 | 1.07 | 17 | **+72.00** | 🥈 Secondary Precious Metal |
| **AUD/USD Forex** | 42.04% | 980 | 0.97 | 43 | -28.00 | ❌ Unprofitable (Do Not Trade) |
| **Crude Oil (CL)** | 40.08% | 1,472 | 0.89 | 32 | -143.00 | ❌ Unprofitable (Do Not Trade) |

---

## 🎯 Selected MT5 Portfolio (3 Assets)

To diversify risk and optimize capital efficiency, we will trade the three highest-performing uncorrelated assets.

### Asset Composition
1. **USD/JPY Forex (1:30 Leverage)**
   * *Rationale*: Outstanding profit factor (1.36) and win rate (50.45%). Capitalizes on long-term macroeconomic interest rate trends.
2. **Nasdaq 100 (QQQ) (1:20 Leverage)**
   * *Rationale*: Tech growth momentum. Highest performing equity index under the QRC model, offering clean trend waves.
3. **Gold (XAUUSD) (1:20 Leverage)**
   * *Rationale*: Inflation and geopolitical hedge. Zero correlation to USD/JPY and equities, smoothing out the portfolio drawdown.

---

## 🛡️ Risk Management & Capital Allocation (Starting: $200)

Because traditional brokers have fixed leverage and minimum order sizes (0.01 lot), a starting capital of **$200** requires strict margin management.

### Dynamic Asset Unlock Rule
To prevent margin calls or capital over-allocation during the initial phase:
* **Stage 1 (Balance < $350)**:
  * The bot trades **ONLY S&P 500 / Nasdaq 100** and **EUR/USD or USD/JPY**.
  * Total required margin for 0.01 lot is only **~$35**, leaving a massive safety buffer of **$165+** (82.5% cash).
* **Stage 2 (Balance >= $350)**:
  * The bot **automatically unlocks Gold (XAUUSD)** (which requires ~$115 margin for 0.01 lot).
  * Safe compounding begins, increasing position sizes by 0.01 lot for every $200 in account balance.

---

## 🛠️ Step-by-Step Launch Guide (Phase 4)

### 1. MT5 Terminal Setup
* Install MetaTrader 5 (Pepperstone Version) on the Windows station.
* Log into your Pepperstone Demo account.
* Allow Algorithmic Trading:
  * Go to `Tools` -> `Options` -> `Expert Advisors`.
  * Check the box `Allow Algorithmic Trading`.
  * Leave the MT5 terminal running on the desktop.

### 2. Run Connection Test
* Open terminal and verify that Python can connect to MT5 and retrieve gold prices:
  ```bash
  python scratch/test_mt5_connect.py
  ```

### 3. Model Training
* Once connection is verified, we will run the training scripts to download historical data directly from Pepperstone via MT5 API and build the local models for `USDJPY`, `QQQ`, and `XAUUSD`.
