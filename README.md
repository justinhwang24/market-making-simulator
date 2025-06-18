# Adaptive Market Making Simulator

This project implements a realistic market making simulator comparing an **adaptive inventory-skewed quoting strategy** against a **constant spread benchmark** under a regime-switching stochastic price process.

## Project Overview

- Simulates 10,000 trader arrivals, resulting in ~4,200 actual trades with 96% noise traders and 4% informed traders
- Mid-price evolves via **regime-switching volatility** (low- and high-vol regimes)
- Informed traders receive a noisy signal about future price and trade directionally
- Market maker earns spread revenue but faces inventory and adverse selection risks

## Adaptive Strategy Features

- Bid/ask prices shift based on current inventory
- Spreads widen during high-volatility periods
- Spread size increases with inventory magnitude
- Informed traders operate on noisy signals, not future prices

## Key Results

| Metric                      | Adaptive MM | Constant Spread |
|----------------------------|-------------|-----------------|
| Sharpe Ratio               | **2.21**    | 0.43            |
| Average PnL per Trade      | 0.2064      | 0.0828          |
| Max Drawdown               | 82.7        | 217.0           |
| Informed Trades Faced      | 162         | 381             |
| Hit Rate (%)               | 95.9%       | 91.1%           |

> Adaptive strategy reduces risk, increases PnL, and deters informed trading.

## Grid Search

Ran grid search over 9 spread-skew combinations to maximize Sharpe:

| Spread | Skew=0.005 | Skew=0.010 | Skew=0.020 |
|--------|------------|------------|------------|
| 0.1    | 0.9612     | 1.5235     | 1.9594     |
| 0.2    | 0.7963     | 1.4053     | **2.2085** |
| 0.3    | 0.6928     | 1.2119     | 2.1703     |

Optimal configuration: **Spread = 0.2**, **Skew = 0.02**
