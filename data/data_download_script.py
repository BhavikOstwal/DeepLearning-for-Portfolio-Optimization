import yfinance as yf
import pandas as pd

tickers = ['^VIX', 'VTI', 'AGG', 'DBC']
start_date = '2006-02-06'
end_date = '2025-05-19'

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker', auto_adjust=True) #type: ignore
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    data.reset_index(inplace=True)
    data.to_csv(f'{ticker.replace("^", "")}.csv', index=False)
    print(f'\033[92m[Success]\033[0m {ticker.replace("^", "")} Dataset downloaded and saved')
