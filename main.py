from utils.dataloader import dataloader
from utils.volatility import volatility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data, market_data, realized_vol = dataloader()


mean_gross_returns = data.mean() + 0.15
ones = np.ones(25, )
# print(mean_gross_returns.shape)
print(data.mean())
# print(mean_gross_returns)
print(data.var())
variance = np.diag(data.var())
# print(variance)
covariance_matrix = data.cov()
inv_cov = np.linalg.inv(covariance_matrix)
A = np.matmul(np.matmul(np.transpose(mean_gross_returns), inv_cov), mean_gross_returns)
B = np.matmul(np.matmul(np.transpose(mean_gross_returns), inv_cov), ones)
C = np.matmul(np.matmul(np.transpose(ones), inv_cov), ones)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(data.corr())

#print(market_data.mean())
#print(market_data.var())

pi_mu = 1/B * (np.matmul(inv_cov, mean_gross_returns))
pi_gmv = 1/C * (np.matmul(inv_cov, ones))
print(pi_mu)
print(pi_gmv)
print(mean_gross_returns)

pi_mu_mean = np.matmul(pi_mu, np.transpose(mean_gross_returns))
pi_gmv_mean = np.matmul(pi_gmv, np.transpose(mean_gross_returns))
print(pi_mu_mean)
print(pi_gmv_mean)
pi_mu_vol = np.sqrt(np.matmul(np.matmul(np.transpose(pi_mu), covariance_matrix), pi_mu))
pi_gmv_vol = np.sqrt(np.matmul(np.matmul(np.transpose(pi_gmv), covariance_matrix), pi_gmv))
print(pi_mu_vol)
print(pi_gmv_mean)

mu = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
      0, 0.1, 0.2, 0.3, 0.4, 0.5,
      0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75,
      0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85,
      0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
      0.96, 0.97, 0.98, 0.99]

mu2 = [1, 1.01, 1.02, 1.03, 1.04, 1.05,
       1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15,
       1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25,
       1.26, 1.27, 1.28, 1.29, 1.30, 1.31, 1.32, 1.33, 1.34, 1.35,
       1.36, 1.37, 1.38, 1.39, 1.40, 1.41, 1.42, 1.43, 1.44, 1.45,
       1.46, 1.47, 1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55,
       1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.95]

vol = []
vol2 = []


for i in mu:
    vol_i = volatility(A, B, C, i)
    # print(vol_i)
    vol.append(vol_i)

for j in mu2:
    vol_j = volatility(A, B, C, j)
    vol2.append(vol_j)

plt.plot(vol, mu, color='blue')
plt.plot(vol2, mu2, color='green')
plt.plot(pi_mu_vol, pi_mu_mean, 'r*')
plt.text(pi_mu_vol-0.125, pi_mu_mean+0.15, 'π_mu')

plt.plot(pi_gmv_vol, pi_gmv_mean, 'r*')
plt.text(pi_gmv_vol+0.125, pi_gmv_mean, 'π_gmv')

plt.xlabel('Volatility')
plt.ylabel('Gross excess returns')
plt.title('Mean-Variance Frontier')
# plt.show()

X = pd.concat([market_data, realized_vol], axis=1)
print(market_data)
print(realized_vol)
print(X)
y = data

reg = LinearRegression()
reg.fit(X, y)
intercept = reg.intercept_
coefficients = reg.coef_

print(f'Intercept: {intercept}')
print('Coefficients:')
for feature, coef in zip(X.columns, coefficients):
    print(f'{feature}: {coef}')

