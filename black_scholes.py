import numpy as np
from scipy.stats import norm

def black_scholes_call(SO, K, T, r, sigma):
    d1 = (np.log(SO/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    call_price = SO * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price
