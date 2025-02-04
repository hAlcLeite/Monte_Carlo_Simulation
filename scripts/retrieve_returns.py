import json
import yfinance as yf
import pandas as pd

def retrieve_historical_returns(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    tickers = config['tickers']
    params = config['dataParameters']
    data = yf.download(tickers, start=params['sample_period_end'], period=params['total_sample_period'])["Adj Close"]
    returns = data.pct_change().dropna()
    return returns

if __name__ == "__main__":
    returns_df = retrieve_historical_returns("../data/ETF_data_info.json")
    print(returns_df.head())

def construct_key_metrics(returns_df):
    cov_matrix = returns_df.cov()
    volatilities = returns_df.std()
    return returns_df, volatilities, cov_matrix