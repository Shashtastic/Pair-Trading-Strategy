import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

capital = 10000
forced_exit_thresh = 3.0

# === List of pairs ===
pair_list = [
    ("CAT", "MA"),
    ("BKNG", "MA"),
    ("HD", "SPG"),
    ("MO", "PM"),
    ("BKNG", "V"),
    ("HON", "NFLX"),
    ("AVGO", "MA"),
    ("MA", "V"),
    ("CVX", "DIS"),
    ("GE", "MA"),
    ("IBM", "MA"),
    ("ABT", "USB"),
    ("AIG", "PG"),
    ("AVGO", "BKNG"),
    ("AIG", "GD"),
    ("LOW", "SPG"),
    ("MA", "NVDA"),
    ("GM", "MMM"),
    ("HON", "ISRG"),
    ("DE", "GILD"),
    ("BRK-B", "PG"),
    ("CVX", "PYPL"),
    ("EMR", "MA"),
    ("EMR", "V"),
    ("ABT", "GM"),
    ("EMR", "JPM"),
    ("CVX", "MDT"),
    ("COP", "DIS"),
    ("KO", "LMT"),
    ("JPM", "NOW"),
    ("JPM", "MA"),
    ("ISRG", "JPM"),
    ("DE", "LMT"),
    ("GE", "NVDA"),
    ("ABBV", "KO"),
    ("ABT", "BAC"),
    ("MS", "PM"),
    ("ABBV", "LMT"),
    ("ACN", "HD"),
    ("EMR", "NVDA"),
    ("CAT", "V"),
    ("HON", "PLTR"),
    ("COP", "MDT"),
    ("EMR", "WFC"),
    ("BLK", "NFLX"),
    ("DE", "MCD"),
    ("DIS", "PYPL"),
    ("HON", "META"),
    ("EMR", "WMT"),
    ("ABT", "COF"),
    ("HON", "T"),
    ("AIG", "LLY"),
    ("EMR", "ISRG"),
    ("CRM", "NFLX"),
    ("SPG", "WFC"),
    ("DE", "NKE"),
    ("DE", "INTC"),
    ("PM", "RTX"),
    ("AVGO", "IBM"),
    ("HD", "WFC"),
    ("HON", "WMT"),
    ("EMR", "NOW"),
    ("ABBV", "GD"),
    ("TXN", "WMT"),
    ("DUK", "RTX"),
    ("JPM", "V"),
    ("LOW", "PG"),
    ("HON", "TXN"),
    ("COP", "PYPL"),
    ("HON", "NOW"),
    ("HON", "JPM"),
    ("AMD", "UNP"),
    ("AIG", "COST"),
    ("DE", "DIS"),
    ("DE", "MDT"),
    ("DE", "LOW"),
    ("DIS", "PEP"),
    ("DUK", "TXN"),
    ("AIG", "AMGN"),
    ("ABT", "C"),
    ("HON", "MMM"),
    ("ACN", "INTU"),
    ("EMR", "GS"),
    ("AMD", "INTU"),
    ("DUK", "PM"),
    ("DE", "PG"),
    ("MA", "WMT"),
    ("ABT", "MMM"),
    ("DE", "XOM"),
    ("MO", "RTX"),
    ("ACN", "UNP"),
    ("EMR", "ORCL"),
    ("JPM", "NVDA"),
    ("CAT", "NVDA"),
    ("DE", "VZ"),
    ("DE", "MRK"),
    ("ABT", "VZ"),
    ("ABBV", "PG"),
    ("DE", "ORCL"),
    ("EMR", "IBM"),
    ("GE", "V"),
    ("AXP", "NVDA"),
    ("LOW", "WFC"),
    ("HON", "ORCL"),
    ("DE", "PYPL"),
    ("DE", "MO"),
    ("DE", "PM"),
    ("DE", "MDLZ"),
    ("JPM", "META"),
    ("MDT", "PYPL"),
    ("PEP", "PYPL"),
    ("EMR", "GOOG"),
    ("LOW", "ORCL"),
    ("HON", "NVDA"),
    ("MS", "WMT"),
    ("DE", "MA"),
    ("AMGN", "GD"),
    ("AAPL", "PG"),
    ("AIG", "MSFT"),
    ("MET", "PM"),
    ("EMR", "GOOGL"),
    ("DE", "WMT"),
    ("USB", "VZ"),
    ("BRK-B", "MA"),
    ("JNJ", "NFLX"),
    ("COST", "PG"),
    ("BK", "ISRG"),
    ("LIN", "LLY"),
    ("DE", "IBM"),
    ("HON", "TMUS"),
    ("CHTR", "XOM"),
    ("EMR", "META"),
    ("ACN", "LOW"),
    ("EMR", "SPG"),
    ("UPS", "WMT"),
    ("GOOG", "GOOGL"),
    ("UPS", "V"),
    ("NVDA", "V"),
    ("HON", "V"),
    ("ABT", "TGT"),
    ("IBM", "NVDA"),
    ("DE", "NVDA"),
    ("COF", "CRM"),
    ("DE", "LLY"),
    ("CAT", "GE"),
    ("BLK", "CRM"),
    ("MSFT", "PG"),
    ("IBM", "V"),
    ("AIG", "NVDA"),
    ("DUK", "MO"),
    ("MET", "MO"),
    ("BRK-B", "COST"),
    ("MET", "RTX"),
    ("TXN", "WFC"),
    ("AVGO", "V"),
    ("DE", "SO"),
    ("DE", "RTX"),
    ("BK", "HD"),
    ("HON", "MA"),
    ("CAT", "LLY"),
    ("AMGN", "LLY"),
    ("JNJ", "PLTR"),
    ("DE", "KO"),
    ("MET", "TMUS"),
    ("GS", "TXN"),
    ("GM", "VZ"),
    ("MS", "RTX"),
    ("DE", "UNH"),
    ("DE", "TMUS"),
    ("DE", "LIN"),
    ("MDT", "XOM"),
    ("AXP", "WMT"),
    ("CRM", "META"),
    ("DE", "V"),
    ("JNJ", "PFE"),
    ("CAT", "IBM"),
    ("BKNG", "WMT"),
    ("TMUS", "WMT"),
    ("V", "WMT"),
    ("INTU", "SPG"),
    ("DE", "TGT"),
    ("MET", "WMT"),
    ("HON", "PFE"),
    ("KO", "RTX"),
    ("AVGO", "NVDA"),
    ("DIS", "XOM"),
    ("HON", "IBM"),
    ("ABT", "NFLX"),
    ("ABT", "AMZN"),
    ("DE", "MET"),
    ("DE", "PEP"),
    ("HD", "INTU"),
    ("JNJ", "VZ"),
    ("ABT", "BLK"),
    ("ABBV", "SO"),
    ("MET", "NVDA"),
    ("ABT", "CRM"),
    ("INTU", "UNP"),
    ("COST", "IBM"),
    ("GS", "ISRG"),
    ("JNJ", "MDT"),
    ("DE", "GD"),
    ("MET", "NKE"),
    ("JNJ", "META"),
    ("CHTR", "MDT"),
    ("META", "NOW"),
    ("JNJ", "PYPL"),
    ("TMO", "UNP"),
    ("DE", "ISRG"),
    ("CVX", "PEP"),
    ("AIG", "LOW"),
    ("JNJ", "SBUX"),
    ("AIG", "BRK-B"),
    ("FDX", "META"),
    ("ABT", "SCHW"),
    ("AAPL", "ORCL"),
    ("HON", "PM"),
    ("CL", "GE"),
    ("LOW", "MS"),
    ("ABT", "TSLA"),
    ("HD", "JPM"),
    ("HON", "INTU"),
    ("HD", "ISRG"),
    ("PEP", "VZ"),
    ("UNH", "XOM"),
    ("DE", "HD"),
    ("GS", "WMT"),
    ("JNJ", "MMM"),
    ("DE", "GE"),
    ("MET", "PG"),
    ("AMZN", "COF"),
    ("CMCSA", "CVX"),
    ("JNJ", "TMO"),
    ("ABT", "ACN"),
    ("MA", "ORCL"),
    ("DE", "NOW"),
    ("GOOG", "HD"),
    ("MDLZ", "MRK"),
    ("COF", "NFLX"),
    ("COP", "PEP"),
    ("ABT", "CMCSA"),
    ("JPM", "WMT"),
    ("BK", "GS"),
    ("HON", "WFC"),
    ("DE", "MSFT"),
    ("HD", "MS"),
    ("JNJ", "NOW"),
    ("BLK", "NOW"),
    ("EMR", "TMUS"),
    ("TMO", "UNH"),
    ("CL", "NVDA"),
    ("TXN", "V"),
    ("ABT", "DIS"),
    ("C", "NFLX"),
    ("JNJ", "QCOM"),
    ("JNJ", "UNH"),
    ("JNJ", "NKE"),
    ("IBM", "TMUS"),
    ("EMR", "GE"),
    ("BKNG", "NVDA"),
    ("RTX", "SO"),
    ("BKNG", "NOW"),
    ("MO", "PG")
]

# === Threshold grid ===
entry_grid = np.linspace(1.1, 1.8, 21)
exit_grid = np.linspace(0.35, 0.7, 21)

# === Result storage ===
qualified_pairs = []

# === Download data for all tickers at once ===
all_tickers = list(set([t for pair in pair_list for t in pair] + ["^TNX"]))
df = yf.download(all_tickers, start="2015-01-01", end="2025-01-01", auto_adjust=False)["Close"]
TNX = df["^TNX"] / 100

# === Backtest function ===
def backtest_pair(prices, ticker1, ticker2, entry_thresh, exit_thresh, forced_exit_thresh=3.0,
                  tc_rate=0.001, slippage=0.0005, capital_per_leg=5000):
    p1 = prices[ticker1]
    p2 = prices[ticker2]
    capital = 2 * capital_per_leg
    window = 365
    beta = p1.rolling(window).cov(p2) / p1.rolling(window).var()
    spread = p1 - beta * p2
    zscore = (spread - np.mean(spread)) / np.std(spread)

    go_long = zscore < -entry_thresh
    go_short = zscore > entry_thresh
    normal_exit = np.abs(zscore) < exit_thresh
    forced_exit = np.abs(zscore) > forced_exit_thresh

    position_1, position_2 = 0.0, 0.0
    pnl_list = []

    for i in range(1, len(prices)):
        price_1, price_2 = p1.iat[i], p2.iat[i]
        price_1_0, price_2_0 = p1.iat[i-1], p2.iat[i-1]
        cost = 0.0

        if position_1 == 0 and position_2 == 0:
            if go_long.iat[i]:
                position_1 = capital_per_leg / price_1
                position_2 = -capital_per_leg / price_2
                cost = 2 * capital_per_leg * (tc_rate + slippage)
            elif go_short.iat[i]:
                position_1 = -capital_per_leg / price_1
                position_2 = capital_per_leg / price_2
                cost = 2 * capital_per_leg * (tc_rate + slippage)

        elif normal_exit.iat[i] or forced_exit.iat[i]:
            cost = (abs(position_1) * price_1 + abs(position_2) * price_2) * (tc_rate + slippage)
            position_1, position_2 = 0.0, 0.0

        pnl = position_1 * (price_1 - price_1_0) + position_2 * (price_2 - price_2_0) - cost
        pnl_list.append(pnl)

    unrealised_equity = capital + pd.Series(pnl_list).cumsum()
    daily_returns = unrealised_equity.pct_change().fillna(0)

    # Match index
    daily_returns.index = prices.index[1:]
    daily_riskfree = TNX.reindex(daily_returns.index).fillna(method='ffill') / 252
    daily_excess_returns = daily_returns - daily_riskfree

    mean_excess = daily_excess_returns.mean()
    std_excess = daily_excess_returns.std()
    sharpe1 = (mean_excess / std_excess) * np.sqrt(252) if std_excess != 0 else np.nan
    return sharpe1

# === Run loop over all pairs ===
for ticker1, ticker2 in pair_list:
    try:
        prices = df[[ticker1, ticker2]].dropna()
        max_sharpe = -np.inf
        best_entry, best_exit = None, None

        for entry_th in entry_grid:
            for exit_th in exit_grid:
                sharpe = backtest_pair(prices, ticker1, ticker2, entry_th, exit_th, forced_exit_thresh)
                if sharpe > max_sharpe:
                    max_sharpe = sharpe
                    best_entry = entry_th
                    best_exit = exit_th

        if max_sharpe > 0.55:
            qualified_pairs.append({
                "ticker1": ticker1,
                "ticker2": ticker2,
                "sharpe": round(max_sharpe, 3),
                "entry_threshold": round(best_entry, 2),
                "exit_threshold": round(best_exit, 2)
            })

    except Exception as e:
        print(f"Error with pair {ticker1}-{ticker2}: {e}")

# === Output qualified pairs ===
print("\nQualified Pairs with Sharpe > 0.6:")
for pair in qualified_pairs:
    print(f'{pair["ticker1"]}-{pair["ticker2"]}: Sharpe={pair["sharpe"]}, Entry={pair["entry_threshold"]}, Exit={pair["exit_threshold"]}')

