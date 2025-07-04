import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = yf.download(["UAL", "DAL", "^TNX","^SPX"], start="2015-01-01", end="2024-12-01", auto_adjust=False)["Close"]
UAL = data["UAL"]
DAL = data["DAL"]
TNX = data["^TNX"]/100
SPX = data["^SPX"]

airline_hedge_ratio, airline_intercept = np.polyfit(data["UAL"], data["DAL"], 1)

airline_spread = UAL - airline_hedge_ratio*DAL

airline_zscore = (airline_spread - np.mean(airline_spread)) /np.std(airline_spread)


entry_thresh = 1.
exit_thresh = 0.5
exit_thresh2 = -0.5
forced_exit_thresh = 2.5

signals = pd.DataFrame(index=data.index)
signals['zscore'] = airline_zscore
signals['long'] = airline_zscore < -entry_thresh
signals['short'] = airline_zscore > entry_thresh
signals['exit'] = abs(airline_zscore) < exit_thresh
signals['forced_exit'] = abs(airline_zscore) > forced_exit_thresh

capital = 10000
capital_per_leg = capital / 2

positions = pd.DataFrame(index=data.index)
positions["long_UAL"] = 0
positions["short_DAL"] = 0
positions["short_UAL"] = 0
positions["long_DAL"] = 0

position = None  # None, 'long', or 'short'

for i in range(1, len(signals)):
    date = signals.index[i]
    ual_price = data["UAL"].iloc[i]
    dal_price = data["DAL"].iloc[i]
    z = airline_zscore.iloc[i]

    if position is None:
        if signals['long'].iloc[i]:
            # Enter long UAL, short DAL
            positions.loc[date, 'long_UAL'] = capital_per_leg / ual_price
            positions.loc[date, 'short_DAL'] = capital_per_leg / dal_price
            position = 'long'

        elif signals['short'].iloc[i]:
            # Enter short UAL, long DAL
            positions.loc[date, 'short_UAL'] = capital_per_leg / ual_price
            positions.loc[date, 'long_DAL'] = capital_per_leg / dal_price
            position = 'short'

    elif position is not None:
        if abs(z) > forced_exit_thresh or signals['exit'].iloc[i]:
            # Forced exit or normal exit
            position = None
            # Don't open a new trade today
        else:
            # Maintain current position
            if position == 'long':
                positions.loc[date, 'long_UAL'] = capital_per_leg / ual_price
                positions.loc[date, 'short_DAL'] = capital_per_leg / dal_price
            elif position == 'short':
                positions.loc[date, 'short_UAL'] = capital_per_leg / ual_price
                positions.loc[date, 'long_DAL'] = capital_per_leg / dal_price

positions['position_value_UAL'] = (positions["long_UAL"] + positions["short_UAL"]) * data["UAL"]

positions['position_value_DAL'] = (positions["long_DAL"] + positions["short_DAL"]) * data["DAL"]

positions['total_position_value'] = (positions['position_value_UAL'].abs() + positions['position_value_DAL'].abs())

price_diff = data.diff()

positions['pnl'] = (
    positions["long_UAL"].shift(1) * price_diff["UAL"] +
    positions["short_DAL"].shift(1) * -price_diff["DAL"] +
    positions["short_UAL"].shift(1) * -price_diff["UAL"] +
    positions["long_DAL"].shift(1) * price_diff["DAL"]
)

monthly_position = positions['total_position_value'].resample('M').mean()

yearly_position = positions['total_position_value'].resample('Y').mean()

overall_total_position = positions['total_position_value'].sum()

print("---- Average Position Size Per Month ----")
print(monthly_position)
print("\n---- Average Position Size Per Year ----")
print(yearly_position)

print("\n--- Summary Stats --")
print (f"Total Average Position Value by Year: {np.mean(yearly_position)}")
print (f"Total Average Position Value by Month: {np.mean(monthly_position)}")
clean_spx = SPX.dropna()

initial_spx_price = clean_spx.iloc[0]
final_spx_price = clean_spx.iloc[-1]

spx_units = capital / initial_spx_price

spx_final_value = spx_units * final_spx_price
spx_pnl = spx_final_value - capital

positions['cumulative_pnl'] = positions['pnl'].cumsum()

total_pnl = positions['cumulative_pnl'].iloc[-1]
print(f"Total Pirs Strat PnL: ${total_pnl:.2f}")
print(f"S&P 500 Buy-and-Hold PnL: ${spx_pnl:.2f}")

daily_returns = positions["pnl"]/capital
daily_riskfree = TNX / 252
daily_riskfree = daily_riskfree.reindex(data.index).fillna(method='ffill')
daily_excess_returns = daily_returns - daily_riskfree
daily_std_excess = np.std(daily_excess_returns)
daily_sharpe = (np.mean(daily_excess_returns) / daily_std_excess) * np.sqrt(252)

returns_sp = data["^SPX"].pct_change()
excess_returns_sp = returns_sp - daily_riskfree
excess_returns_sp = excess_returns_sp.dropna()

if excess_returns_sp.std() != 0:
    sharpe_sp = (excess_returns_sp.mean() / excess_returns_sp.std()) * np.sqrt(252)
else:
    sharpe_sp = np.nan

print(f"Sharpe Ratio: {daily_sharpe}")
print(f"Sharpe Ratio for SP500: {sharpe_sp}")


num_long_entries = signals['long'].astype(int).diff().fillna(0).eq(1).sum()
num_short_entries = signals['short'].astype(int).diff().fillna(0).eq(1).sum()
num_exits = signals['exit'].astype(int).diff().fillna(0).eq(1).sum()
num_forced_exits = signals['forced_exit'].astype(int).diff().fillna(0).eq(1).sum()

total_trades = int(num_long_entries + num_short_entries +num_exits + num_forced_exits)

print(f"Total number of trades entered: {total_trades}")
print(f" - Long trades: {int(num_long_entries)}")
print(f" - Short trades: {int(num_short_entries)}")
print(f" - Exit trades: {int(num_exits)}")
print(f" - Forced Exit trades: {int(num_forced_exits)}")


plt.figure(figsize=(14, 8))
plt.plot(positions['cumulative_pnl'], label="Cumulative PnL", color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Cumulative PnL of UAL-DAL Pair Strategy")
plt.xlabel("Date")
plt.ylabel("PnL ($)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,8))
plt.plot(airline_zscore, label="Spread Z-Score")
plt.axhline(entry_thresh, color='red', linestyle='--', label="Entry Z")
plt.axhline(-entry_thresh, color='green', linestyle='--', label="Entry Z")
plt.axhline(exit_thresh, color='black', linestyle=':',label="Exit Z" )
plt.axhline(exit_thresh2, color='black', linestyle=':',label="Exit Z" )
plt.title("Z-Score of Adjusted Spread")
plt.legend()
plt.show()