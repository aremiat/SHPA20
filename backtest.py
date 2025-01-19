import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the tickers and weights for the portfolio
tickers = ['MA', 'TSLA', 'CMG', 'RMS', 'NOVO-B.CO', 'CRM', 'PGHN.SW', 'REGN', 'V', 'WSP',
           'AHT', 'ASML', 'LII', 'ASM', 'INTU', 'WST', 'SNPS', 'TSCO', 'FI', 'MTD']
weights = [0.076160, 0.069118, 0.058923, 0.057336, 0.056844, 0.052907, 0.050583, 0.048601,
           0.048058, 0.047396, 0.046568, 0.046477, 0.046106, 0.044993, 0.043754, 0.043250,
           0.041264, 0.041211, 0.040239, 0.040211]

# Download the historical data from Yahoo Finance (from 2015)
start_date = '2017-01-01'
end_date = '2025-01-01'
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate the daily returns for each stock
returns = data.pct_change().dropna()

# Calculate the portfolio returns assuming constant weights each year
portfolio_returns = (returns * weights).sum(axis=1)

# Additional index for MSCI ACWI Low Carbon Target
index_ticker = 'CRBN'  # Ticker for the MSCI ACWI Low Carbon Target Index
index_data = yf.download(index_ticker, start=start_date, end=end_date)['Close']

# Calculate the returns of the index
index_returns = index_data.pct_change().dropna()

# Align both series on the same date range
common_dates = portfolio_returns.index.intersection(index_returns.index)
portfolio_returns = portfolio_returns.loc[common_dates]
index_returns = index_returns.loc[common_dates]

# Add the index returns to the portfolio returns for comparison
combined_returns = pd.concat([
    portfolio_returns,
    index_returns
], axis=1)

combined_returns.columns = ['SHAP 20', 'MSCI ACWI Low Carbon Target']
# Calculate the cumulative returns of both the portfolio and the MSCI ACWI Low Carbon Target Index
combined_cumulative_returns = (1 + combined_returns).cumprod()

# Plot the comparison of cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(combined_cumulative_returns['SHAP 20'], label="SHAP 20 Cumulative Return", color='blue')
plt.plot(combined_cumulative_returns['MSCI ACWI Low Carbon Target'], label="MSCI ACWI Low Carbon Target Cumulative Return", color='green')
plt.title("SHAP 20 vs. MSCI ACWI Low Carbon Target Index: Cumulative Returns (2015-2025)")
plt.xlabel("Year")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Calculate the annualized returns (CAGR) for both the portfolio and the MSCI ACWI Low Carbon Target Index
years = (common_dates[-1] - common_dates[0]).days / 365.25
portfolio_annualized_return = (combined_cumulative_returns['SHAP 20'][-1]) ** (1/years) - 1
index_annualized_return = (combined_cumulative_returns['MSCI ACWI Low Carbon Target'][-1]) ** (1/years) - 1

print(f"Annualized Return (CAGR) from 2016 to 2025 for SHAP 20: {portfolio_annualized_return:.2%}")
print(f"Annualized Return (CAGR) from 2016 to 2025 for MSCI ACWI Low Carbon Target Index: {index_annualized_return:.2%}")

# Calculate cumulative returns for both portfolio and index
portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
index_cumulative_returns = (1 + index_returns).cumprod()

# Resample to yearly frequency, getting the last value for each year
portfolio_annualized_returns = portfolio_cumulative_returns.resample('Y').last().pct_change().add(1)
index_annualized_returns = index_cumulative_returns.resample('Y').last().pct_change().add(1)

# Combine the annualized returns of both the portfolio and index
annualized_returns_per_year = pd.concat({
    'SHAP 20 Annualized Return': portfolio_annualized_returns,
    'MSCI ACWI Low Carbon Target Annualized Return': index_annualized_returns
}, axis=1).dropna()

annualized_returns_per_year = (annualized_returns_per_year -1) * 100  # Convert to percentage
annualized_returns_per_year.columns = ['SHAP 20', 'MSCI ACWI Low Carbon Target']
# Display the result
print(annualized_returns_per_year)