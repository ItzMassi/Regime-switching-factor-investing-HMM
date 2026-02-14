import yfinance as yf
import numpy as np
import pandas as pd

def fetch_data(ticker='SPY', start='2007-01-01', end='2025-12-31'):
    print('Fetching data for {ticker} from {start} to {end}...')
    data = yf.download(ticker, start=start, end=end, progress= False)
    # Handling of multi-index cols
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    return data

def calculate_features(df):
    df = df.copy()

    # Daily Returns
    df['Daily_return'] = df['Close'].pct_change()

    # Volatility (possibility to try with the standard deviation) 
    ma_10 = df['Daily_return'].rolling(window = 10).mean()
    squared_diff = (df['Daily_return'] - ma_10)**2
    df['MSE_Vol'] = squared_diff.rolling(window = 10).mean()
    
    return df

if __name__ == '__main__':
    df = fetch_data()
    if not df.empty:
        df = calculate_features(df)
        print(df.head())
        print(df.tail())
        print('\n','NAs:')
        print(df.isna().sum())
