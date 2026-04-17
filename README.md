# Latency-Aware Execution Engine for Portfolio Rebalancing

[![Tests](https://github.com/devfrankduah/ML-Latency-Aware-Execution-Engine/actions/workflows/tests.yml/badge.svg)](https://github.com/devfrankduah/ML-Latency-Aware-Execution-Engine/actions/workflows/tests.yml)

<!-- [![codecov](https://codecov.io/gh/devfrankduah/ML-Latency-Aware-Execution-Engine/graph/badge.svg)](https://codecov.io/gh/devfrankduah/ML-Latency-Aware-Execution-Engine) -->

A production-quality trade execution system that minimizes slippage on large cryptocurrency orders using reinforcement learning. Trained on 7.5M bars of real Binance market data across 3 assets, evaluated walk-forward on fully out-of-sample 2024 data, and validated against 98M real tick-level trades.

**CS6140 - Machine Learning | Spring 2026 | Northeastern University**

---

## Results

### Strategy Comparison (50 BTC, 500 simulations)

| Strategy            | Cost ($)   | vs TWAP    |
| ------------------- | ---------- | ---------- |
| Immediate           | $370.29    | -1,734%    |
| TWAP                | $20.20     | baseline   |
| VWAP                | $15.09     | +25.3%     |
| A-C (λ=0.5)         | $20.20     | +0.0%      |
| **ML Agent (ours)** | **$11.29** | **+44.1%** |

### Multi-Asset Out-of-Sample (50 BTC, 2024 H2)

| Asset       | vs TWAP       | Win Rate | vs VWAP   |
| ----------- | ------------- | -------- | --------- |
| BTCUSDT     | **+10.2 bps** | 75%      | -0.6 bps  |
| ETHUSDT     | **+0.6 bps**  | 65%      | -0.8 bps  |
| SOLUSDT     | -0.2 bps      | 12%      | -0.03 bps |
| **Average** | **+3.5 bps**  |          | -0.5 bps  |

At 50 BTC (~$2.5M), saving 10.2 bps on BTCUSDT = **~$940 per order**. Over 1,000 orders = **$940,000 saved**.

---

### QR-DQN Risk-Sensitive Execution (50 BTC, 2024 H2)

| Agent                 | vs TWAP       | Beat%   | Cost ($)    | Cost Std    |
| --------------------- | ------------- | ------- | ----------- | ----------- |
| DQN (standard)        | +10.1 bps     | 75%     | $23,004     | $15,366     |
| **QR-DQN (CVaR 10%)** | **+11.6 bps** | **76%** | **$22,245** | **$14,164** |
| QR-DQN (neutral)      | +6.3 bps      | 69%     | $23,998     | $15,023     |
| QR-DQN (CVaR 90%)     | −26.4 bps     | 13%     | $36,126     | $16,715     |

The QR-DQN learns the full return distribution (51 quantiles per action) and enables risk-sensitive execution via CVaR. Conservative execution (CVaR 10%) outperforms standard DQN with lower cost and lower variance.

### Ablation Study (12 configurations, 50 BTC)

| Configuration           | vs TWAP | Beat% | Δ bps     |
| ----------------------- | ------- | ----- | --------- |
| Baseline (all features) | +11.68  | 79%   | —         |
| Remove volume imbalance | +9.42   | 73%   | **−2.26** |
| Remove time-of-day      | +10.49  | 78%   | −1.19     |
| Remove volume ratio     | +11.39  | 76%   | −0.29     |
| Remove momentum         | +11.43  | 80%   | −0.25     |
| Remove cost advantage   | +11.67  | 79%   | −0.01     |
| Remove spread features  | +11.77  | 77%   | +0.09     |
| Remove volatility       | +12.43  | 80%   | +0.75     |
| Order size: 10 BTC      | +4.65   | 78%   | −7.03     |
| Order size: 100 BTC     | +2.30   | 50%   | −9.38     |
| Horizon: 30 bars        | +1.65   | 48%   | −10.03    |
| Impact: η=0.5           | +20.44  | 79%   | +8.76     |
| Participation cap: 5%   | −0.87   | 36%   | −12.55    |

**Key findings:** Volume imbalance (−2.26 bps) and time-of-day (−1.19 bps) are the most important learned features. Reducing the participation cap to 5% causes the largest degradation (−12.55 bps), confirming that liquidity access is the agent's primary advantage.

## What This Project Is and Why It Is Interesting

Large institutional orders cannot be executed all at once without moving the market against the trader. A hedge fund buying 50 BTC (~$2.5M) in a single trade will exhaust the limit-order book at the best price and pay for progressively worse fills - a phenomenon called **market impact** or **slippage**. The industry-standard heuristic is to slice the order over time following a TWAP (time-weighted) or VWAP (volume-weighted) schedule. We ask: can a reinforcement learning agent learn a better schedule by reading live market conditions?

This is an interesting ML problem for three reasons:

1. The reward signal is sparse and delayed (execution cost is only known at the end of the order).
2. The state space is non-stationary - volatility, volume, and spread change minute-to-minute and across market regimes.
3. Generalization must hold across three assets and a fully held-out 6-month test set never seen during training.

## Problem

When a hedge fund needs to buy 50 BTC, executing it all at once moves the price against them - this is called **slippage** or **market impact**. The cost increases nonlinearly with order size: each additional BTC costs more than the last because you're consuming progressively deeper levels of the order book.

The question: **How do you slice a large order over 60 minutes to minimize total execution cost?**

This project builds a complete system to answer that question using real market data and reinforcement learning.

---

## Quick Start

### Setup

```bash
git clone https://github.com/devfrankduah/ML-Latency-Aware-Execution-Engine.git
cd ML-Latency-Aware-Execution-Engine
python -m venv venv && source venv/bin/activate
pip install pandas numpy pyarrow pyyaml torch tqdm requests matplotlib seaborn
```

### Download Data

```bash
python scripts/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --start 2020-01-01 --end 2024-12-31
python scripts/validate_data.py --data data/raw/klines/BTCUSDT/ --output data/processed/BTCUSDT_klines_1m.parquet
```

### Run Full Pipeline (single command)

```bash
python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet
python scripts/run_pipeline.py --data data/processed/BTCUSDT_klines_1m.parquet --full  # includes failure/ethics analysis
```

### Train RL Agent

```bash
# Single asset, 50 BTC orders (~3 hours on CPU)
python scripts/train_large.py --data data/processed/BTCUSDT_klines_1m.parquet --qty 50 --episodes 50000

# Multi-asset training on BTC + ETH + SOL (~5 hours)
python scripts/train_multi.py --train --episodes 50000 --qty 50

# Order size sweep (10, 25, 50 BTC)
python scripts/train_large.py --data data/processed/BTCUSDT_klines_1m.parquet --sweep
```

### Evaluate on Tick Data

```bash
python scripts/eval_ticks.py --tick-dir data/raw/trades/BTCUSDT --model models/multi/best.pt --qty 50
```

### Generate Figures

```bash
python scripts/generate_figures.py --data data/processed/BTCUSDT_klines_1m.parquet --model models/multi/best.pt
```

### Run Tests

```bash
# Run the full test suite (176 tests)
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific module
python -m pytest tests/test_features/ -v
python -m pytest tests/test_simulator/ -v
python -m pytest tests/test_policies/ -v
python -m pytest tests/test_data/ -v
python -m pytest tests/test_evaluation/ -v
python -m pytest tests/test_utils/ -v
```

### Train QR-DQN (Risk-Sensitive)

```bash
python scripts/train_qrdqn.py --train --episodes 100000 --qty 50
python scripts/train_qrdqn.py --frontier --model models/qrdqn/best.pt  # efficient frontier
python scripts/train_qrdqn.py --compare --dqn-model models/multi/best.pt --model models/qrdqn/best.pt
```

### Run Ablation Study

```bash
python scripts/ablation.py --data data/processed --model models/multi/best.pt --n-episodes 300 --qty 50
```

### Launch Dashboard

```bash
pip install streamlit plotly
streamlit run scripts/dashboard.py
```

---

## Tools and Libraries

| Library / Tool           | Role                                                              |
| ------------------------ | ----------------------------------------------------------------- |
| **PyTorch**              | Neural network training - Double DQN, dueling architecture, PER   |
| **NumPy**                | Vectorized numerical computation for features and simulation      |
| **pandas**               | Time-series data loading, resampling, and Parquet I/O             |
| **pyarrow**              | Parquet file read/write for processed market data                 |
| **matplotlib / seaborn** | Publication-quality performance figures                           |
| **pytest**               | 176-test unit and integration test suite                          |
| **PyYAML**               | Centralized hyperparameter configuration (`configs/default.yaml`) |
| **tqdm**                 | Progress bars for long training runs                              |
| **requests**             | Binance data download with retry and checksum verification        |
| **Streamlit / Plotly**   | Interactive TCA dashboard with 5 panels                           |

All models are implemented from scratch using PyTorch primitives. No high-level RL library (Stable Baselines, RLlib, etc.) is used. The Almgren-Chriss baseline model is also implemented from scratch in `src/simulator/impact.py` based on the original 2000 paper. TWAP and VWAP are custom implementations in `src/policies/baselines.py`.

---

## System Architecture

```
Raw Market Data (Binance OHLCV + Tick Trades)
        │
        ▼
┌──────────────────┐
│  Data Ingestion   │  download_data.py, validate_data.py
│  & Validation     │  → Schema validation, 7 quality checks
│                   │  → 2.6M bars per asset, 126 GB tick data
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Feature          │  src/features/engine.py
│  Engineering      │  → 14 features: volatility, volume, spread,
│                   │    momentum, time-of-day encoding
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Execution        │  src/simulator/impact.py + engine.py
│  Simulator        │  → Almgren-Chriss impact model
│                   │  → Variable spread from real data
│                   │  → Cost = spread + η × price × part^1.5
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌─────────────┐
│Baselines│ │ RL Agent     │
│ TWAP   │ │ Double DQN   │  14-dim state, 7 VWAP-relative actions
│ VWAP   │ │ + Dueling    │  Terminal reward: savings vs VWAP in bps
│ A-C    │ │ + PER        │  50K episodes, 3 assets
│ Immed. │ │ + LayerNorm  │
└───┬────┘ └──────┬──────┘
    │              │
    ▼              ▼
┌──────────────────┐
│  Evaluation       │  Monte Carlo (500+ sims per strategy)
│                   │  Walk-forward: Train 2020-23 → Test 2024 H2
│  → Regime analysis│  Cross-asset generalization (BTC, ETH, SOL)
│  → Tick-level     │  98M real trade validation
│  → Failure cases  │  Edge cases, limitations, ethics
└──────────────────┘
```

---

## Repository Structure

```
ML-Latency-Aware-Execution-Engine/
├── src/                            # Source code (Python package)
│   ├── __init__.py                 # v1.0.0
│   ├── data/
│   │   ├── schemas.py              # KlineSchema, TradeSchema
│   │   ├── loader.py               # Binance CSV/Parquet loader
│   │   └── validator.py            # 7 data quality checks
│   ├── features/
│   │   └── engine.py               # 14 execution-relevant features
│   ├── simulator/
│   │   ├── impact.py               # Almgren-Chriss market impact model
│   │   └── engine.py               # Policy-agnostic execution simulator
│   ├── policies/
│   │   ├── baselines.py            # Immediate, TWAP, VWAP, A-C
│   │   ├── rl_env.py               # Gym-style RL environment
│   │   └── dqn_agent.py            # Double DQN + PER + dueling
│   ├── evaluation/
│   │   ├── backtest.py             # Monte Carlo backtester
│   │   └── visualizations.py       # Plot utilities
│   └── utils/
│       ├── config.py               # YAML config loader
│       └── errors.py               # PipelineError, safe_execute, validate helpers
├── scripts/
│   ├── train_qrdqn.py              # QR-DQN training + efficient frontier
│   ├── ablation.py                 # 12-config ablation study
│   ├── dashboard.py                # Streamlit interactive TCA dashboard                   # CLI tools
│   ├── download_data.py            # Data download with retry
│   ├── validate_data.py            # Quality checks + Parquet export
│   ├── run_backtest.py             # Baseline strategy comparison
│   ├── run_pipeline.py             # Single-command end-to-end pipeline
│   ├── train_large.py              # DQN training (large orders, sweep)
│   ├── train_multi.py              # Multi-asset training (BTC+ETH+SOL)
│   ├── eval_ticks.py               # Tick-level evaluation (98M trades)
│   ├── analysis.py                 # Failure/edge/ethics analysis
│   ├── generate_figures.py         # Publication-quality plots
│   └── parse_log.py                # Training log → CSV
├── tests/                          # 176 tests
│   ├── test_data/
│   │   └── test_loader_and_validator.py  # schemas, CSV loader, validator
│   ├── test_features/
│   │   └── test_engine.py               # all feature functions, edge cases
│   ├── test_simulator/
│   │   └── test_engine_and_impact.py    # impact model, A-C trajectory, simulator
│   ├── test_policies/
│   │   └── test_all_policies.py         # baselines, adaptive, RL env, DQN agent
│   ├── test_evaluation/
│   │   └── test_backtest.py             # Monte Carlo backtester, regime analysis
│   └── test_utils/
│       └── test_errors_and_config.py    # PipelineError, safe_execute, config loader
├── configs/default.yaml            # Centralized hyperparameters
├── notebooks/                      # Colab training notebooks
├── models/                         # Saved checkpoints (best.pt, final.pt)
├── reports/figures/                 # Generated plots (4 figures)
├── data/                           # Market data (gitignored)
├── .github/
│   └── workflows/
│       └── tests.yml               # CI: runs test suite on push/PR to main
└── .gitignore
```

---

## Technical Details

### Data

| Asset           | Bars           | Date Range        | Price Range       | Median Volume  |
| --------------- | -------------- | ----------------- | ----------------- | -------------- |
| BTCUSDT         | 2,628,555      | 2020-01 → 2024-12 | $3,810 → $108,258 | 34.6 BTC/min   |
| ETHUSDT         | 2,628,554      | 2020-01 → 2024-12 | $86 → $4,865      | 254.6 ETH/min  |
| SOLUSDT         | 2,307,975      | 2020-04 → 2024-12 | $1 → $264         | 1,407 SOL/min  |
| **Tick trades** | **98,667,010** | 2023-03           | Individual trades | Trade-by-trade |

### Impact Model

```
cost = quantity × (spread/2 + η × price × participation^1.5)
```

- Spread: variable from real data (median 2 bps, range 0.5–10 bps)
- η = 0.3 (temporary impact coefficient)
- Participation cap: 15% of bar volume

### RL Agent

| Component | Choice                                                                    |
| --------- | ------------------------------------------------------------------------- |
| Algorithm | Double DQN with dueling architecture                                      |
| Replay    | Prioritized experience replay (PER, α=0.6)                                |
| Network   | 256-256-128 with LayerNorm                                                |
| Optimizer | AdamW (lr=3e-4, cosine annealing)                                         |
| Actions   | 7 VWAP-relative: [0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]                        |
| State     | 14 features (inventory, time, vol, volume, spread, momentum, time-of-day) |
| Reward    | Terminal: savings vs VWAP in bps                                          |
| Training  | 50K episodes across BTC+ETH+SOL (2020-2023)                               |
| QR-DQN    | 51 quantile estimates per action, CVaR-based action selection             |

### Evaluation Protocol

| Split    | Data                  | Purpose                             |
| -------- | --------------------- | ----------------------------------- |
| Train    | BTC+ETH+SOL 2020-2023 | 6M bars, agent learns               |
| Validate | 2024 Jan-Jun          | Early stopping, model selection     |
| Test     | 2024 Jul-Dec          | Final reported results (never seen) |

---

## Failure Cases & Limitations

**Failure cases** (from `scripts/analysis.py`):

- Flash crashes: +318 bps IS during COVID (2020-03-20)
- Low volume: 500 BTC orders achieve only 49% fill rate
- Short horizons: 1-bar execution costs 20× more than 60-bar

**Limitations**: No real order book data, simulated (not real) fills, single-asset execution, no multi-venue routing, discrete actions, 1-minute resolution.

**Ethical risks**: Market manipulation (mitigated by participation caps), fairness concerns, over-reliance on automation (mitigated by kill switches), data privacy (public data only).

Full analysis: `python scripts/analysis.py --data data/processed/BTCUSDT_klines_1m.parquet`

---

## Conclusions

The results confirm that a learned execution policy can outperform classical heuristics on real market data. The Double DQN agent, trained purely on 2020-2023 data, generalized to unseen 2024 H2 conditions and reduced execution cost by 44.1% versus TWAP on BTCUSDT (1 BTC, 500 simulations) and by 10.2 bps on 50 BTC out-of-sample. VWAP and Almgren-Chriss improve over TWAP but do not match the agent's flexibility in responding to live volatility and spread conditions. The agent's performance degrades gracefully during extreme events (flash crashes, low-volume periods) but does not fail catastrophically - it reverts toward a TWAP-like schedule when uncertainty is high.

The key design decisions that mattered most were: (1) terminal rather than per-step rewards, which gave the agent a clean objective without reward shaping; (2) multi-asset training, which forced generalization; and (3) the walk-forward evaluation protocol, which ensured no data leakage.

---

## Future Work

Given more time, the following extensions would be most valuable:

1. **Continuous action space** - Replace the 7-bin discrete action with a continuous fraction using PPO or SAC. This would allow finer-grained scheduling and likely improve performance on small order sizes where discrete bins are too coarse.
2. **Order book features** - Incorporate level-2 order book depth (bid/ask volume at top 5 levels) to let the agent directly observe liquidity rather than inferring it from OHLCV.
3. **LSTM/Transformer encoder** - The current 14-feature MLP has no memory of past bars within an episode. A temporal encoder could capture intra-episode momentum shifts.
4. **Multi-venue routing** - Real institutional execution spans multiple venues (Binance, Coinbase, Kraken). Extending the action space to include venue allocation would more accurately model production constraints.
5. **Live paper-trading validation** - Connect to a Binance testnet API to validate that simulated fill rates match live fills under real order-book dynamics.
6. **Risk-sensitive optimization** - Use distributional RL (IQN or QR-DQN) to directly minimize conditional value at risk (CVaR) rather than expected cost, which is more appropriate for risk-averse institutional mandates.
7. ~~**Risk-sensitive optimization**~~ **Implemented** - QR-DQN with 51 quantiles and efficient execution frontier across 11 CVaR levels. CVaR 10% achieves +11.6 bps vs TWAP.

---

## References

1. Almgren & Chriss (2000). Optimal execution of portfolio transactions.
2. Bertsimas & Lo (1998). Optimal control of execution costs.
3. Nevmyvaka et al. (2006). Reinforcement learning for optimized trade execution. ICML.
4. Ning et al. (2021). Double deep Q-learning for optimal execution.
5. van Hasselt et al. (2016). Deep RL with double Q-learning. AAAI.
6. Wang et al. (2016). Dueling network architectures. ICML.
7. Schaul et al. (2016). Prioritized experience replay. ICLR.

---

## Author

- **Nikhilesh Waghmare** - MS in Artificial Intelligence, Northeastern University (Expected May 2027)
- **Frank Duah** - MS in Artificial Intelligence, Northeastern University (Expected May 2027)
