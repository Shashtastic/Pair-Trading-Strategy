import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
import itertools

# Load tickers from Excel
sp100_df = pd.read_excel("/Users/shashwatgupta/Desktop/Finance/sp100_tickers_industries.xlsx")
tickers = sp100_df['Ticker'].dropna().unique().tolist()

# Download price data
price_data = yf.download(tickers, start="2015-01-01", end="2025-01-01")['Close']
price_data = price_data.dropna()

results = []

# === Check all possible pairs (across industries too) ===
for stock1, stock2 in itertools.combinations(tickers, 2):
    if stock1 not in price_data.columns or stock2 not in price_data.columns:
        continue

    series1 = price_data[stock1]
    series2 = price_data[stock2]
    combined = pd.concat([series1, series2], axis=1).dropna()
    if len(combined) == 0:
        print(f"No overlapping data for {stock1}-{stock2}")
        continue

    # Perform cointegration test
    score, pvalue, _ = coint(combined.iloc[:, 0], combined.iloc[:, 1])

    industry1 = sp100_df.loc[sp100_df['Ticker'] == stock1, 'Industry'].values[0]
    industry2 = sp100_df.loc[sp100_df['Ticker'] == stock2, 'Industry'].values[0]

    results.append({
        'Pair': f"{stock1}-{stock2}",
        'Ticker1 Industry': industry1,
        'Ticker2 Industry': industry2,
        'Test Statistic': score,
        'P-Value': pvalue
    })

# Create results DataFrame
initial_pair_results_df = pd.DataFrame(results)

# Filter cointegrated pairs
cointegrated_pairs = initial_pair_results_df[initial_pair_results_df['P-Value'] < 0.05].sort_values(['P-Value'])

print("Cointegrated pairs across all industries:")
print(cointegrated_pairs.to_string(index=False))

# Generate pair list
final_pair_list = []
for pair_name in cointegrated_pairs['Pair'].dropna():
    parts = pair_name.split('-', maxsplit=1)
    if len(parts) == 2:
        final_pair_list.append(tuple(parts))

# Output to be copied onto the input pairs of Threshold File
print("\npair_list = [")
for t1, t2 in final_pair_list:
    print(f'    ("{t1}", "{t2}"),')
print("]")
