from utils.dataloader import dataloader,save_results_csv,tangency_portfolio
import statsmodels.api as sm
import pandas as pd

data, market_data, rf = dataloader()

## Part 2##
print(data.mean())
print(data.var())
print(data.corr())

print(market_data.mean())
print(market_data.var())


## Part 5 ##

# Create an empty DataFrame to store regression results
results_table = pd.DataFrame(columns=['Portfolio', 'Coefficient', 'Standard_Error', 'Intercept', 'Intercept_Std_Err', 'R-squared'])

# Extract the market excess return from market_data
market_return = market_data
# Loop through each portfolio column in portfolio_data
for portfolio_column in data.columns:
    portfolio_return = data[portfolio_column]
    
    # Extract the corresponding market excess return from market_data
    market_return = market_data
    
    # Add a constant (intercept) to the market return
    X = sm.add_constant(market_return)
    
    # Fit the linear regression model
    model = sm.OLS(portfolio_return, X).fit()

   # Get coefficients, intercept, standard errors, p-values, and R-squared
    coefficient = model.params[1]  # Index 1 corresponds to the coefficient of the market excess return
    intercept = model.params[0]    # Index 0 corresponds to the intercept
    coefficient_std_err = model.bse[1]  # Standard error of the coefficient
    p_value_coeff = model.pvalues[1]    # P-value of the coefficient
    intercept_std_err = model.bse[0]    # Standard error of the intercept
    p_value_intercept = model.pvalues[0]  # P-value of the intercept
    r_squared = model.rsquared

    # Create a DataFrame for the current portfolio's results
    portfolio_result = pd.DataFrame({'Portfolio': [portfolio_column],
                                     'Intercept': [intercept],
                                     'P-Value_Intercept': [p_value_intercept],
                                     'Coefficient': [coefficient],
                                     'Standard_Error': [coefficient_std_err],
                                     'P-Value_Coeff': [p_value_coeff],
                                     'R-squared': [r_squared]})

    # Concatenate the portfolio_result DataFrame with results_table
    results_table = pd.concat([results_table, portfolio_result], ignore_index=True)


# Print the table of results
print(results_table)

save_results_csv('part_5.csv',results_table)

tangency_portfolio(market_data,data,rf)
