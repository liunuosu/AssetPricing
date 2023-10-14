import pandas as pd


def dataloader():

    # Load double sorted portfolio
    returns = pd.read_csv("data/size_bm_data.csv")

    # Load market data
    market_data = pd.read_csv("data/market_data.csv")

    # Load volatility data
    realized_vol = pd.read_csv("data/realized_vol.csv")

    start_date = 196401  # January 1964
    end_date = 202109  # September 2021

    returns = returns[(returns['Month'] >= start_date) & (returns['Month'] <= end_date)]
    market_data = market_data[(market_data['Month'] >= start_date) & (market_data['Month'] <= end_date)]
    realized_vol = realized_vol[(realized_vol['Month'] >= start_date) & (realized_vol['Month'] <= end_date)]

    # Set month as new index#
    if 'Month' in returns.columns:
        returns.set_index('Month', inplace=True)
    if 'Month' in market_data.columns:
        market_data.set_index('Month', inplace=True)
    if 'Month' in realized_vol.columns:
        realized_vol.set_index('Month', inplace=True)

    # Subtract the risk-free rate from portfolio returns
    returns = returns.sub(market_data['RF'], axis=0)

    return returns, market_data['Mkt-RF'], realized_vol
