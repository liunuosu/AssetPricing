import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def dataloader():
    #returns = pd.read_csv("data/size_inv_data.csv")
    #market_data = pd.read_csv("data/market_data.csv")

    returns = pd.read_csv("data/size_bm_data.csv")
    market_data = pd.read_csv("data/market_data.csv")
    realized_vol = pd.read_csv("data/realized_vol.csv")

    start_date = 196404  # April 1964
    end_date = 202106  # June 2021

    returns = returns[(returns['Month'] >= start_date) & (returns['Month'] <= end_date)]
    market_data = market_data[(market_data['Month'] >= start_date) & (market_data['Month'] <= end_date)]
    realized_vol = realized_vol[(realized_vol['Month'] >= start_date) & (realized_vol['Month'] <= end_date)]

    #print(returns)
    #print(market_data)
    #print(market_data['RF'])

    # Set month as new index#
    if 'Month' in returns.columns:
        returns.set_index('Month', inplace=True)
    if 'Month' in market_data.columns:
        market_data.set_index('Month', inplace=True)
    if 'Month' in realized_vol.columns:
        realized_vol.set_index('Month', inplace=True)


    returns = returns.sub(market_data['RF'], axis=0)
    #print(market_data['RF'])
    #print(returns)

    return returns, market_data['Mkt-RF'], realized_vol
