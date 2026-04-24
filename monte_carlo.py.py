import numpy as np

def monte_carlo_call(SO, K, T, r, sigma, simulations=10000):
    Z = np.random.standard_normal(simulations)
    
    ST = SO * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    
    payoff = np.maximum(ST - K, 0)
    
    price = np.exp(-r*T) * np.mean(payoff)
    
    return price