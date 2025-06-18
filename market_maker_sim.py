import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_market_maker(T=10000, mu=0, sigma=0.1, initial_price=100, spread=0.3, skew_factor=0.01, transaction_cost=0.05, seed=42, p_noise=0.4, max_inventory=50):
    np.random.seed(seed)

    regime_switches = np.random.choice([0.1, 0.2], size=T, p=[0.6, 0.4])
    mid_price = [initial_price]
    for t in range(T):
        sigma_t = regime_switches[t]
        delta = np.random.normal(mu, sigma_t)
        mid_price.append(mid_price[-1] + delta)

    traders = np.random.choice(["noise", "informed"], size=T, p=[p_noise, 1 - p_noise])
    trade_directions = np.random.choice(["buy", "sell"], size=T)

    inventory = 0
    cash = 0
    trade_log = []
    realized_pnl = []
    inventory_list = []
    bid_list = []
    ask_list = []

    for t in range(T):
        mid = mid_price[t]
        skew = inventory * skew_factor
        rolling_vol = np.std(mid_price[max(0, t-50):t+1]) if t > 0 else 0

        dynamic_spread = spread + 0.5 * rolling_vol + 0.005 * abs(inventory)
        bid = mid - dynamic_spread / 2 - skew
        ask = mid + dynamic_spread / 2 - skew

        trader_type = traders[t]
        direction = trade_directions[t]

        traded = False
        if trader_type == "noise":
            if direction == "buy":
                cash += ask
                inventory -= 1
                traded = True
                pnl = ask - mid - transaction_cost
            elif direction == "sell":
                cash -= bid
                inventory += 1
                traded = True
                pnl = mid - bid - transaction_cost

        elif trader_type == "informed":
            edge = np.random.normal(0, 0.1)  # realistic signal with noise
            true_value = mid + edge
            if direction == "buy" and true_value > ask:
                cash += ask
                inventory -= 1
                traded = True
                pnl = ask - true_value - transaction_cost
            elif direction == "sell" and true_value < bid:
                cash -= bid
                inventory += 1
                traded = True
                pnl = true_value - bid - transaction_cost

        if traded:
            trade_log.append({
                't': t,
                'type': trader_type,
                'direction': direction,
                'price': ask if direction == "buy" else bid,
                'mid': mid,
                'pnl': pnl,
                'inventory': inventory,
            })

        bid_list.append(bid)
        ask_list.append(ask)
        inventory_list.append(inventory)
        total_pnl = cash + inventory * mid
        realized_pnl.append(total_pnl)
        if traded:
            trade_log[-1]['total_pnl'] = total_pnl

    pnl_series = pd.Series(realized_pnl)
    df_trades = pd.DataFrame(trade_log)

    return {
        'pnl_series': pnl_series,
        'inventory': inventory_list,
        'df_trades': df_trades,
        'bid': bid_list,
        'ask': ask_list
    }

def run_constant_spread_maker(T=10000, mu=0, sigma=0.1, initial_price=100, spread=0.3, transaction_cost=0.05, seed=42, p_noise=0.4):
    np.random.seed(seed)
    mid_price = [initial_price]
    for _ in range(T):
        delta = np.random.normal(mu, sigma)
        mid_price.append(mid_price[-1] + delta)

    traders = np.random.choice(["noise", "informed"], size=T, p=[p_noise, 1 - p_noise])
    trade_directions = np.random.choice(["buy", "sell"], size=T)

    inventory = 0
    cash = 0
    realized_pnl = []
    trade_log = []

    for t in range(T):
        mid = mid_price[t]
        bid = mid - spread / 2
        ask = mid + spread / 2
        trader_type = traders[t]
        direction = trade_directions[t]

        traded = False
        if trader_type == "noise":
            if direction == "buy":
                cash += ask
                inventory -= 1
                traded = True
                pnl = ask - mid - transaction_cost
            else:
                cash -= bid
                inventory += 1
                traded = True
                pnl = mid - bid - transaction_cost
        elif trader_type == "informed":
            edge = np.random.normal(0, 0.1)  # same realistic signal as adaptive
            true_value = mid + edge
            if direction == "buy" and true_value > ask:
                cash += ask
                inventory -= 1
                traded = True
                pnl = ask - true_value - transaction_cost
            elif direction == "sell" and true_value < bid:
                cash -= bid
                inventory += 1
                traded = True
                pnl = true_value - bid - transaction_cost

        if traded:
            trade_log.append({'t': t, 'type': trader_type, 'direction': direction, 'pnl': pnl})

        total_pnl = cash + inventory * mid
        realized_pnl.append(total_pnl)

    return pd.Series(realized_pnl), pd.DataFrame(trade_log)

def compute_metrics(pnl_series, df_trades):
    returns = pnl_series.diff().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_drawdown = (pnl_series.cummax() - pnl_series).max()
    df_trades['is_win'] = df_trades['pnl'] > 0
    hit_rate = df_trades['is_win'].mean()

    return {
        'Total Trades': len(df_trades),
        'Noise Trades': len(df_trades[df_trades['type'] == 'noise']),
        'Informed Trades': len(df_trades[df_trades['type'] == 'informed']),
        'PnL (Noise)': round(df_trades[df_trades['type'] == 'noise']['pnl'].sum(), 4),
        'PnL (Informed)': round(df_trades[df_trades['type'] == 'informed']['pnl'].sum(), 4),
        'Average PnL per Trade': round(df_trades['pnl'].mean(), 4),
        'Sharpe Ratio': round(sharpe, 4),
        'Max Drawdown': round(max_drawdown, 4),
        'Hit Rate (%)': round(hit_rate * 100, 2)
    }

def plot_pnl_and_quotes(pnl_series, bid_list, ask_list):
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_series, label="Total PnL", linewidth=2)
    plt.plot(bid_list, label="Bid", alpha=0.3)
    plt.plot(ask_list, label="Ask", alpha=0.3)
    plt.title("Market Making PnL with Inventory-Based Quoting")
    plt.xlabel("Time")
    plt.ylabel("PnL / Quote")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_inventory(inventory_list):
    plt.figure(figsize=(10, 4))
    plt.plot(inventory_list, label="Inventory")
    plt.title("Inventory Over Time")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.grid(True)
    plt.show()

def grid_search():
    spreads = [0.1, 0.2, 0.3]
    skews = [0.005, 0.01, 0.02]
    results = []

    for s in spreads:
        for k in skews:
            out = run_market_maker(spread=s, skew_factor=k)
            sharpe = compute_metrics(out['pnl_series'], out['df_trades'])['Sharpe Ratio']
            results.append((s, k, sharpe))

    return pd.DataFrame(results, columns=["Spread", "Skew", "Sharpe"])

if __name__ == "__main__":
    results_adaptive = run_market_maker()
    metrics_adaptive = compute_metrics(results_adaptive['pnl_series'], results_adaptive['df_trades'])

    pnl_constant, trades_constant = run_constant_spread_maker()
    metrics_constant = compute_metrics(pnl_constant, trades_constant)

    print("\n--- Adaptive Market Maker Metrics ---")
    for k, v in metrics_adaptive.items():
        print(f"{k}: {v}")

    print("\n--- Constant Spread Benchmark Metrics ---")
    for k, v in metrics_constant.items():
        print(f"{k}: {v}")

    plot_pnl_and_quotes(results_adaptive['pnl_series'], results_adaptive['bid'], results_adaptive['ask'])
    plot_inventory(results_adaptive['inventory'])

    results_adaptive['df_trades'].to_csv("market_maker_trade_log_adaptive.csv", index=False)
    trades_constant.to_csv("market_maker_trade_log_constant.csv", index=False)

    print("\n--- Grid Search Over Spread and Skew ---")
    grid_df = grid_search()
    print(grid_df.pivot(index='Spread', columns='Skew', values='Sharpe'))
