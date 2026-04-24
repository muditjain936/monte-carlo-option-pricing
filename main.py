from monte_carlo import monte_carlo_call
from black_scholes import black_scholes_call

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

np.random.seed(42)

#Fetch real market data
ticker = "AAPL"   # you can change to NIFTY, RELIANCE.NS etc.
data = yf.download(ticker, start="2022-01-01", end="2024-01-01", auto_adjust=True)

#Compute log returns
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

#Estimate volatility (annualized)
sigma = float(returns.std() * np.sqrt(252))

#Current stock price
S0 = float(data['Close'].iloc[-1])

#Option parameters
K = S0   # ATM option
T = 1
r = 0.05

print(f"Using real data for {ticker}")
print(f"Current Price (S0): {S0:.2f}")
print(f"Estimated Volatility (sigma): {sigma:.4f}")

#Averaging function
def mc_avg(S0, K, T, r, sigma, simulations, runs=10):
    prices = []
    for _ in range(runs):
        prices.append(monte_carlo_call(S0, K, T, r, sigma, simulations))
    return np.mean(prices)

#Stock path simulation
def simulate_paths(S0, T, r, sigma, steps=100, simulations=5):
    dt = T / steps
    paths = np.zeros((steps, simulations))
    paths[0] = S0

    for t in range(1, steps):
        Z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

    return paths

#Pricing
mc_price = monte_carlo_call(S0, K, T, r, sigma, simulations=10000)
bs_price = black_scholes_call(S0, K, T, r, sigma)

print(f"Monte Carlo Price: {mc_price:.4f}")
print(f"Black-Scholes Price: {bs_price:.4f}")

#Convergence
simulations_list = [1000, 5000, 10000, 20000, 50000, 100000]
prices = []

for sim in simulations_list:
    price = mc_avg(S0, K, T, r, sigma, sim, runs=10)
    prices.append(price)

#Plot everything
plt.figure(figsize=(10, 12))

#Plot 1: Convergence
plt.subplot(3, 1, 1)
plt.plot(simulations_list, prices, marker='o', label="Monte Carlo")
plt.axhline(y=bs_price, linestyle='--', label="Black-Scholes")
plt.title(f"Monte Carlo Convergence ({ticker})")
plt.xlabel("Simulations")
plt.ylabel("Option Price")
plt.legend()

#Plot 2: Stock Paths
plt.subplot(3, 1, 2)
paths = simulate_paths(S0, T, r, sigma)

for i in range(paths.shape[1]):
    plt.plot(paths[:, i])

plt.title("Simulated Stock Price Paths (GBM)")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")

#Plot 3: Payoff Histogram
plt.subplot(3, 1, 3)

Z = np.random.standard_normal(10000)
ST = S0 * np.exp((r - 0.5 * sigma**2)*T + sigma*np.sqrt(T)*Z)
payoffs = np.maximum(ST - K, 0)

plt.hist(payoffs, bins=50, range=(0, np.percentile(payoffs, 95)))
plt.axvline(np.mean(payoffs), linestyle='--', label='Mean')
plt.legend()
plt.title("Distribution of Option Payoffs")
plt.xlabel("Payoff")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
