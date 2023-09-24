import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def dataloader():
    returns = pd.read_csv("data//size_inv_data.csv")
    market_data = pd.read_csv("data//market_data.csv")

    start_date = 196404  # April 1964
    end_date = 202106  # June 2021

    returns = returns[(returns['Month'] >= start_date) & (returns['Month'] <= end_date)]
    market_data = market_data[(market_data['Month'] >= start_date) & (market_data['Month'] <= end_date)]
    #print(returns)
    #print(market_data)
    #print(market_data['RF'])

    # Set month as new index
    if 'Month' in returns.columns:
        returns.set_index('Month', inplace=True)
    if 'Month' in market_data.columns:
        market_data.set_index('Month', inplace=True)

    returns = returns.sub(market_data['RF'], axis=0)
    #print(market_data['RF'])
    #print(returns)

    return returns, market_data['Mkt-RF'],market_data['RF']

def save_results_csv(file_path, results):
    # Write the DataFrame to a CSV file
    results.to_csv(file_path, index=False)  # Set index=False to exclude the index column

    # Optionally, print a message to confirm the save
    print(f'Results saved to {file_path}')
