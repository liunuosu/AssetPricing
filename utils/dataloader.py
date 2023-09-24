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
def tangency_portfolio(market_returns,other_excess_returns,risk_free):
    #calculate standard deviation of portfolios' excess returns
    portfolio_stats = pd.DataFrame(columns=['Portfolio', 'Standard Deviation', 'Sharpe Ratio'])

    for column in other_excess_returns.columns:
        excess = other_excess_returns[column]
        sd_dev = excess.std()
        mean_return = excess.mean()
        sharpe_ratio = mean_return/sd_dev
        
        portfolio = pd.DataFrame({
            'Portfolio': [column],
            'Standard Deviation': [sd_dev],
            'Sharpe Ratio': [sharpe_ratio]
        })

        portfolio_stats = pd.concat([portfolio_stats,portfolio], ignore_index=True)

    
    # Find the portfolio with the highest Sharpe ratio
    tangency_portfolio = portfolio_stats.loc[portfolio_stats['Sharpe Ratio'].idxmax()]

    # Extract the standard deviation and expected excess return for the tangency portfolio
    tangency_portfolio_std = tangency_portfolio['Standard Deviation']
    tangency_portfolio_return = tangency_portfolio['Sharpe Ratio'] * tangency_portfolio_std
    cov_matrix = np.cov(other_excess_returns, rowvar=False)
    market_std = market_returns.std()
    market_mean = market_returns.mean()
    rf_mean = risk_free.mean()
    efficient_returns, efficient_volatilities = efficient_frontier(cov_matrix,other_excess_returns,risk_free)
    
     # Plot the tangency portfolio, market portfolio, Capital Market Line, and Mean-Variance Frontier
    plt.scatter(0, rf_mean, color='blue', marker='o', s=100, label='Risk-Free Rate')
    plt.scatter(tangency_portfolio_std, tangency_portfolio_return, color='red', marker='o', s=100, label='Tangency Portfolio')
    plt.plot(market_std, market_mean, marker='o', markersize=8, label='Market Portfolio', color='purple')
    plt.plot([0, tangency_portfolio_std], [rf_mean, tangency_portfolio_return], color='blue', linestyle='-', label='Capital Market Line')
    plt.plot(efficient_volatilities, efficient_returns, marker='o', linestyle='-', label='Mean-Variance Frontier')


    # Add labels and title to the plot
    plt.xlabel('Portfolio Volatility (Standard Deviation)')
    plt.ylabel('Expected Portfolio Return')
    plt.title('Mean-Variance Frontier and Tangency Portfolio with Risk-Free Rate')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

def efficient_frontier(cov_matrix,other_excess_returns):
     # Define your data: μ, Σ, Rf, and other parameters
    mean_excess_return = other_excess_returns.mean()
    min_expected_excess_return = min(mean_excess_return) - 0.005  # Adjust the lower limit
    max_expected_excess_return = max(mean_excess_return) + 0.005  # Adjust the upper limit

    # Define a range of target expected excess returns (μe) for the frontier
    target_returns = np.linspace(min_expected_excess_return, max_expected_excess_return,num=200)

    # Initialize empty lists to store portfolio properties
    efficient_returns = []
    efficient_volatilities = []

    for target_return in target_returns:
        # Define the optimization problem to minimize portfolio variance
        # Define the objective function to minimize portfolio variance
        def objective_function(weights):
            portfolio_variance = weights @ cov_matrix @ weights
            return portfolio_variance

        # Define the equality constraint for the target expected excess return
        def constraint_function(weights):
            return weights @ mean_excess_return - target_return
        # Initial guess for portfolio weights (e.g., equal-weighted portfolio)
        initial_weights = np.ones(len(mean_excess_return)) / len(mean_excess_return)
        # Define equality constraint for the target expected excess return
        constraint = {'type': 'eq', 'fun': constraint_function}
        # Use an optimization solver to find efficient portfolio weights
        result = minimize(objective_function, initial_weights, constraints=constraint)

        # Extract efficient portfolio weights
        efficient_weights = result.x

        # Calculate portfolio expected return and standard deviation
        portfolio_return = efficient_weights @ mean_excess_return
        portfolio_std = np.sqrt(efficient_weights @ cov_matrix @ efficient_weights)

        efficient_returns.append(portfolio_return)
        efficient_volatilities.append(portfolio_std)
    return efficient_returns,efficient_volatilities
    
