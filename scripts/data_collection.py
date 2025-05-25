import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start_date, end_date):
    return yf.download(tickers=symbol, start=start_date, end=end_date, auto_adjust=False)

def collect_portfolio_data(portfolio_symbols, start_date, end_date):
    portfolio_data = pd.DataFrame()

    for symbol in portfolio_symbols:
        print(f"\n📥 Downloading data for {symbol}...")
        asset_data = yf.download(tickers=symbol, start=start_date, end=end_date, auto_adjust=False)

        # Flatten MultiIndex columns if necessary
        if isinstance(asset_data.columns, pd.MultiIndex):
            asset_data.columns = [' '.join(col).strip() for col in asset_data.columns.values]

        # Find the actual 'Adj Close' column (may be 'Adj Close AAPL', etc.)
        adj_close_col = next((col for col in asset_data.columns if col.lower().startswith('adj close')), None)

        if not adj_close_col:
            print(f"⚠️ 'Adj Close' column not found for {symbol}. Available: {asset_data.columns.tolist()}")
            continue

        # Extract and rename the column, reset index to get Date as column
        asset_data = asset_data[[adj_close_col]].rename(columns={adj_close_col: symbol})
        asset_data = asset_data.reset_index()  # Ensures Date is a column, not index

        # Merge with portfolio_data on 'Date'
        if portfolio_data.empty:
            portfolio_data = asset_data
        else:
            portfolio_data = pd.merge(portfolio_data, asset_data, on='Date', how='outer')

    return portfolio_data

if __name__ == "__main__":
    portfolio_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    start_date = "2018-01-01"
    end_date = "2025-01-31"

    portfolio_data = collect_portfolio_data(portfolio_symbols, start_date, end_date)

    if 'Date' in portfolio_data.columns:
        portfolio_data['Date'] = pd.to_datetime(portfolio_data['Date'])
        portfolio_data = portfolio_data.sort_values('Date')

        output_path = "data/portfolio_data.csv"
        portfolio_data.to_csv(output_path, index=False)
        print(f"\n✅ Data saved to {output_path}")
    else:
        print("❌ No data collected. Please check symbol availability or column structure.")
