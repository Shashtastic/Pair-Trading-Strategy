import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import scipy.optimize as sco
warnings.filterwarnings("ignore")

# === Constants ===
forced_exit_thresh = 3
capital_per_leg = 5000
transaction_cost_rate, slippage_rate = 0.001, 0.0005
capital = 2 * capital_per_leg
window = 365

# === Read pairs and thresholds from Excel ===
# Excel format: ticker1, ticker2, entry_thresh, exit_thresh
pairs_df = pd.read_excel("/Users/shashwatgupta/Desktop/Finance/pairs_for_portfolio.xlsx")
pairs_df[['ticker1', 'ticker2']] = pairs_df['Pair'].str.split('-', expand=True)
else_data = ["^TNX", "^SPX"]
data_for_else = yf.download(else_data, start="2015-01-01", end="2025-01-01", auto_adjust=False)["Close"]

results = []
pair_pnl_dict = {}

for idx, row in pairs_df.iterrows():
    ticker1 = row['ticker1']
    ticker2 = row['ticker2']
    entry_thresh = row['EntryThreshold']
    exit_thresh = row['ExitThreshold']

    # === Download historical price data ===
    tickers = [ticker1, ticker2]
    data = yf.download(tickers, start="2015-01-01", end="2025-01-01", auto_adjust=False)["Close"]
    
    # Extract relevant series
    S1, S2 = data[ticker1], data[ticker2]
    TNX = data_for_else["^TNX"] / 100  # Convert 10Y yield to decimal
    SPX = data_for_else["^SPX"]

    # === Calculate rolling hedge ratio and spread ===
    hedge_ratio = S1.rolling(window).cov(S2) / S1.rolling(window).var()
    spread = S1 - hedge_ratio * S2
    zscore = (spread - np.mean(spread)) / np.std(spread)

    # === Define trade signals based on z-score ===
    signals = pd.DataFrame(index=data.index)
    signals['zscore'] = zscore
    signals['long'] = zscore < -entry_thresh
    signals['short'] = zscore > entry_thresh
    signals['exit'] = abs(zscore) < exit_thresh
    signals['forced_exit'] = abs(zscore) > forced_exit_thresh

    # === Initialize position and cost tracking ===
    positions = pd.DataFrame(index=data.index)
    positions[["long_S1", "short_S2", "short_S1", "long_S2"]] = 0.0
    positions["transaction_costs"] = 0.0
    positions['actual_long_entry'] = 0
    positions['actual_short_entry'] = 0
    positions['actual_exit'] = 0
    positions['actual_forced_exit'] = 0

    position = None  # No open trade at the start

    # === Simulate Trading Logic ===
    for i in range(1, len(signals)):
        date, prev_date = signals.index[i], signals.index[i-1]
        s1_price, s2_price = S1.iloc[i], S2.iloc[i]
        z = zscore.iloc[i]

        # Carry forward previous positions
        for col in ["long_S1", "short_S2", "short_S1", "long_S2"]:
            positions.loc[date, col] = positions.loc[prev_date, col]

        # === Entry Conditions ===
        if position is None:
            if signals['long'].iloc[i]:
                # Go long S1, short S2
                positions.loc[date, 'long_S1'] = capital_per_leg / s1_price
                positions.loc[date, 'short_S2'] = capital_per_leg / s2_price
                position = 'long'
                positions.loc[date, 'transaction_costs'] = 2 * capital_per_leg * (transaction_cost_rate + slippage_rate)
                positions.loc[date, 'actual_long_entry'] = 1

            elif signals['short'].iloc[i]:
                # Go short S1, long S2
                positions.loc[date, 'short_S1'] = capital_per_leg / s1_price
                positions.loc[date, 'long_S2'] = capital_per_leg / s2_price
                position = 'short'
                positions.loc[date, 'transaction_costs'] = 2 * capital_per_leg * (transaction_cost_rate + slippage_rate)
                positions.loc[date, 'actual_short_entry'] = 1

        # === Exit Conditions ===
        elif position:
            forced_exit = signals['forced_exit'].iloc[i]
            normal_exit = signals['exit'].iloc[i] and not forced_exit

            if forced_exit or normal_exit:
                # Calculate total exposure to apply exit transaction cost
                notional = 0
                if position == 'long':
                    notional += positions.loc[date, 'long_S1'] * s1_price
                    notional += positions.loc[date, 'short_S2'] * s2_price
                elif position == 'short':
                    notional += positions.loc[date, 'short_S1'] * s1_price
                    notional += positions.loc[date, 'long_S2'] * s2_price

                positions.loc[date, "transaction_costs"] = notional * (transaction_cost_rate + slippage_rate)

                # Log exit type
                if forced_exit:
                    positions.loc[date, 'actual_forced_exit'] = 1
                else:
                    positions.loc[date, 'actual_exit'] = 1

                # Close positions
                for col in ["long_S1", "short_S2", "short_S1", "long_S2"]:
                    positions.loc[date, col] = 0
                position = None

    # === Calculate Daily PnL ===
    positions['position_value_S1'] = (positions["long_S1"] - positions["short_S1"]) * S1
    positions['position_value_S2'] = (positions["long_S2"] - positions["short_S2"]) * S2
    positions['total_position_value'] = positions['position_value_S1'].abs() + positions['position_value_S2'].abs()

    # Compute daily price differences
    price_diff = data[[ticker1, ticker2]].diff()

    # PnL from both legs
    positions['pnl'] = (
        positions["long_S1"].shift(1) * price_diff[ticker1] +
        positions["short_S2"].shift(1) * -price_diff[ticker2] +
        positions["short_S1"].shift(1) * -price_diff[ticker1] +
        positions["long_S2"].shift(1) * price_diff[ticker2]
    )

    # Subtract transaction costs
    positions['pnl_after_costs'] = positions['pnl'] - positions['transaction_costs']
    positions['cumulative_pnl_after_costs'] = positions['pnl_after_costs'].cumsum()

    # === Store Daily PnL for Portfolio Building ===
    pair_key = f"{ticker1}-{ticker2}"
    if 'pair_pnl_dict' not in globals():
        pair_pnl_dict = {}
    pair_pnl_dict[pair_key] = positions['pnl_after_costs']

    # === Sharpe Ratio Calculation ===
    equity = capital + positions['cumulative_pnl_after_costs']
    daily_returns = equity.pct_change().fillna(0)
    daily_riskfree = TNX / 252
    daily_riskfree = daily_riskfree.reindex(data.index).fillna(method='ffill')
    daily_excess_returns = daily_returns - daily_riskfree

    if np.std(daily_excess_returns) != 0:
        sharpe = (np.mean(daily_excess_returns) / np.std(daily_excess_returns)) * np.sqrt(252)
    else:
        sharpe = np.nan

    # === Save Final Results ===
    total_pnl = positions['pnl_after_costs'].sum()
    results.append({
        'ticker1': ticker1,
        'ticker2': ticker2,
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe
    })

# Convert results to DataFrame and show
results_df = pd.DataFrame(results)
top_4_pnl_df = results_df.nlargest(4, 'total_pnl')

print(top_4_pnl_df)

# === Optimising the Pair Portfolio by Minimising Variance ===
# === Step 1: Extract top 4 pair keys (in the format 'TICKER1-TICKER2') ===
top_4_keys = [f"{row['ticker1']}-{row['ticker2']}" for _, row in top_4_pnl_df.iterrows()]

# === Step 2: Build daily PnL DataFrame for the top 4 pairs ===
# This constructs a new DataFrame where each column is a pair's PnL time series
daily_pnls_df = pd.DataFrame({pair: pair_pnl_dict[pair] for pair in top_4_keys})

# Create a returns DataFrame (filling missing values with 0 to avoid NaNs in calc)
returns_df = daily_pnls_df.fillna(0)

# === Step 3: Compute the sample covariance matrix of returns ===
# This matrix quantifies how each pair's PnLs co-move with others
cov_matrix = returns_df.cov().values

# === Step 4: Define the objective function: portfolio variance ===
# This will be minimised to get optimal weights for lowest risk
def portfolio_variance(weights):
    return weights @ cov_matrix @ weights.T  # Matrix multiplication: wᵀ * Σ * w

# === Step 5: Constraints and bounds for optimization ===

# Constraint: Sum of weights must be 1 (fully invested portfolio)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds: Each weight should be between 1% and 100% (no short-selling, no 0 allocation)
bounds = [(0.01, 1) for _ in top_4_keys]

# Initial guess: Equal allocation to all 4 pairs
init_guess = np.array([1 / len(top_4_keys)] * len(top_4_keys))

# === Step 6: Handle numerical issues in covariance matrix ===
# Replace any NaN or Inf values with 0s to prevent optimization failures
if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)

# === Step 7: Run the optimization using Sequential Least Squares Programming (SLSQP) ===
opt_result = sco.minimize(
    portfolio_variance,       # Objective function: minimise portfolio variance
    init_guess,               # Starting guess for weights
    method='SLSQP',           # Optimization method suitable for constrained problems
    bounds=bounds,            # Bounds on weights
    constraints=constraints   # Ensure weights sum to 1
)

# === Step 8: Extract optimal results ===

# Optimal portfolio weights that minimise risk
optimal_weights = opt_result.x

# Portfolio variance using optimal weights
pair_portfolio_variance_value = portfolio_variance(optimal_weights)

# === Calculate Pair portfolio PnL & Sharpe ===
daily_pnls_df = pd.DataFrame({pair: pair_pnl_dict[pair] for pair in top_4_keys}).fillna(0)
weighted_pnl = daily_pnls_df @ optimal_weights
pair_pnl_scaler = 0.4 #Input variable in final code
pair_portfolio_daily_pnl = weighted_pnl * pair_pnl_scaler  # scale by $4k

# Equity curve & % returns
pair_equity = (capital * pair_pnl_scaler) + pair_portfolio_daily_pnl.cumsum()
pair_daily_returns = pair_equity.pct_change().fillna(0)

# Risk-free rate
daily_rf = data_for_else["^TNX"] / 100 / 252
daily_rf = daily_rf.reindex(daily_returns.index).fillna(method='ffill')

# Sharpe
pair_daily_excess = pair_daily_returns - daily_rf
if pair_daily_excess.std() > 1e-8:
    pair_sharpe_ratio = (pair_daily_excess.mean() / pair_daily_excess.std()) * np.sqrt(252)
else:
    pair_sharpe_ratio = np.nan

total_pair_pnl = pair_portfolio_daily_pnl.sum()

print(f"\n=== Pair Portfolio ===")
print(f"Total pair trading PnL: ${total_pnl:.2f}")
print(f"Total pair trading Sharpe: {pair_sharpe_ratio:.4f}")

# === S&P 500 Buy-and-Hold Pnl and Sharpe Ratio ===
#Sharpe
spx_returns = SPX.pct_change().dropna() * (1-pair_pnl_scaler)
spx_excess_returns = spx_returns - daily_riskfree.reindex(spx_returns.index).fillna(method='ffill')
spx_excess_returns = spx_excess_returns.dropna()
if spx_excess_returns.std() != 0:
    sharpe_spx_buyhold = (spx_excess_returns.mean() / spx_excess_returns.std()) * np.sqrt(252)
else:
    sharpe_spx_buyhold = np.nan
print(f"\n=== Combined Portfolio ===")
print(f"S&P 500 Buy-and-Hold Annualized Sharpe: {sharpe_spx_buyhold:.2f}")

#PnL Value
spx_initial_value = SPX.iloc[0]
spx_final_value = SPX.iloc[-1]
spx_shares = (capital * (1-pair_pnl_scaler)) / spx_initial_value
spx_final_portfolio_value = spx_shares * spx_final_value
spx_total_return = spx_final_portfolio_value - (capital * (1-pair_pnl_scaler))
spx_total_return_full = spx_total_return*(1/0.6)
print(f"Total S&P500 Buy and Hold Return: ${spx_total_return:,.2f}")
print(f"Total S&P500 Buy and Hold Return if 10K: ${spx_total_return_full:,.2f}")

# === Combined Portfolio PnL Sharpe ===
# Rebase SPX to simulate $6000 buy and hold
spx_normalized = SPX / SPX.iloc[0]   # starts at 1
spx_capital = capital * (1 - pair_pnl_scaler)  # $6000
spx_portfolio_value = spx_normalized * spx_capital
spx_daily_pnl = spx_portfolio_value.diff().fillna(0)

# Align dates
combined_portfolio_df = pd.DataFrame({
    'pair_pnl': pair_portfolio_daily_pnl,
    'spx_pnl': spx_daily_pnl
}).dropna()

# Total PnL per day
combined_portfolio_df['total_pnl'] = combined_portfolio_df['pair_pnl'] + combined_portfolio_df['spx_pnl']

# Recreate equity curve
combined_portfolio_df['equity'] = capital + combined_portfolio_df['total_pnl'].cumsum()

# Daily % returns
combined_portfolio_df['daily_return'] = combined_portfolio_df['equity'].pct_change().fillna(0)

# Risk-free rate aligned
rf_combined = daily_rf.reindex(combined_portfolio_df.index).fillna(method='ffill')

# Excess returns
combined_portfolio_df['excess_return'] = combined_portfolio_df['daily_return'] - rf_combined

# Sharpe
if combined_portfolio_df['excess_return'].std() > 1e-8:
    combined_sharpe = (combined_portfolio_df['excess_return'].mean() /
                       combined_portfolio_df['excess_return'].std()) * np.sqrt(252)
else:
    combined_sharpe = np.nan

print(f"\n=== Combined Portfolio ===")
print(f"Total Portfolio Sharpe: {combined_sharpe:.4f}")
print(f"Total Portfolio PnL: ${combined_portfolio_df['total_pnl'].cumsum().iloc[-1]:,.2f}")
print(f"Total Portfolio Final Value: ${combined_portfolio_df['equity'].iloc[-1]:,.2f}")

pair_equity = capital * pair_pnl_scaler + combined_portfolio_df['pair_pnl'].cumsum()
spx_equity = capital * (1 - pair_pnl_scaler) + combined_portfolio_df['spx_pnl'].cumsum()

# Plot equity curve comparison
plt.figure(figsize=(12, 6))
plt.plot(combined_portfolio_df.index, combined_portfolio_df['equity'], label='Total Combined Portfolio', color='dodgerblue', linewidth=1)
plt.plot(combined_portfolio_df.index, pair_equity, label='Cumulative Pair Value', color='green', linewidth=1, linestyle='--' )
plt.plot(combined_portfolio_df.index, spx_equity, label='Cumulative S&P 500 Value', color='black', linewidth=1, linestyle='--')

plt.title("Combined Portfolio Value", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

