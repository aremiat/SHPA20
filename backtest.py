import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = ['MA', 'TSLA', 'CMG', 'RMS', 'NOVO-B.CO', 'CRM', 'PGHN.SW', 'REGN', 'V', 'WSP',
           'AHT', 'ASML', 'LII', 'ASM', 'INTU', 'WST', 'SNPS', 'TSCO', 'FI', 'MTD']
WEIGHTS = [0.076160, 0.069118, 0.058923, 0.057336, 0.056844, 0.052907, 0.050583, 0.048601,
           0.048058, 0.047396, 0.046568, 0.046477, 0.046106, 0.044993, 0.043754, 0.043250,
           0.041264, 0.041211, 0.040239, 0.040211]

START_DATE = '2017-01-01'
END_DATE = '2025-01-01'


if __name__ == '__main__':
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)['Close']
    returns = data.pct_change().dropna()
    portfolio_returns = (returns * WEIGHTS).sum(axis=1)

    # Additional index for MSCI ACWI Low Carbon Target
    index_ticker = 'CRBN'  # Ticker for the MSCI ACWI Low Carbon Target Index
    index_data = yf.download(index_ticker, start=START_DATE, end=END_DATE)['Close']
    index_returns = index_data.pct_change().dropna()
    common_dates = portfolio_returns.index.intersection(index_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    index_returns = index_returns.loc[common_dates]
    combined_returns = pd.concat([
        portfolio_returns,
        index_returns
    ], axis=1)

    combined_returns.columns = ['SHPA 20', 'MSCI ACWI Low Carbon Target']
    combined_cumulative_returns = (1 + combined_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(combined_cumulative_returns['SHPA 20'], label="SHPA 20 Cumulative Return", color='blue')
    plt.plot(combined_cumulative_returns['MSCI ACWI Low Carbon Target'], label="MSCI ACWI Low Carbon Target Cumulative Return", color='green')
    plt.title("SHPA 20 vs. MSCI ACWI Low Carbon Target Index: Cumulative Returns (2017-2025)")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the annualized returns (CAGR) for both the portfolio and the MSCI ACWI Low Carbon Target Index
    years = (common_dates[-1] - common_dates[0]).days / 365.25
    portfolio_annualized_return = (combined_cumulative_returns['SHPA 20'][-1]) ** (1/years) - 1
    index_annualized_return = (combined_cumulative_returns['MSCI ACWI Low Carbon Target'][-1]) ** (1/years) - 1

    print(f"Annualized Return (CAGR) from 2017 to 2025 for SHPA 20: {portfolio_annualized_return:.2%}")
    print(f"Annualized Return (CAGR) from 2017 to 2025 for MSCI ACWI Low Carbon Target Index: {index_annualized_return:.2%}")


    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    index_cumulative_returns = (1 + index_returns).cumprod()
    portfolio_annualized_returns = portfolio_cumulative_returns.resample('Y').last().pct_change()
    index_annualized_returns = index_cumulative_returns.resample('Y').last().pct_change()
    annualized_returns_per_year = pd.concat({
        'SHPA 20 Annualized Return': portfolio_annualized_returns,
        'MSCI ACWI Low Carbon Target Annualized Return': index_annualized_returns
    }, axis=1).dropna()

    annualized_returns_per_year = (annualized_returns_per_year) * 100
    annualized_returns_per_year.columns = ['SHPA 20', 'MSCI ACWI Low Carbon Target']
    print(annualized_returns_per_year)