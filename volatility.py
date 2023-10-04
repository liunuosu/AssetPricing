import numpy as np


def volatility(A, B, C, mu):
    numerator = A - 2*B*mu + C*np.power(mu, 2)
    denominator = A*C - np.power(B, 2)
    vol = np.sqrt(numerator/denominator)
    return vol
