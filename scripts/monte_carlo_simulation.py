import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.retrieve_returns import retrieve_historical_returns, construct_key_metrics

ASSETS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
NUM_PORTFOLIOS = 100_000
RISK_FREE_RATE = 0

returns_df = retrieve_historical_returns("data/ETF_data_info.json")
returns_df, volatilities, cov_matrix = construct_key_metrics(returns_df)

results = np.zeros((4, NUM_PORTFOLIOS))
weights_record = np.zeros((len(ASSETS), NUM_PORTFOLIOS))

for i in range(NUM_PORTFOLIOS):
    weights = np.random.random(len(ASSETS))
    weights /= np.sum(weights)
    weights_record[:, i] = weights

    port_return = np.sum(weights * returns_df.mean()) * 12
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
    sharpe_ratio = (port_return - RISK_FREE_RATE) / port_volatility

    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe_ratio
    results[3, i] = i

simulated_portfolios = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio', 'Simulation'])

optimal_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
optimal_portfolio = simulated_portfolios.loc[optimal_idx]
optimal_weights = weights_record[:, optimal_idx]

print("Optimal Portfolio Weights:")
for asset, weight in zip(ASSETS, optimal_weights):
    print(f"{asset}: {weight * 100:.2f}%")

print(f"\nOptimized Portfolio Return: {optimal_portfolio['Return'] * 100:.2f}%")
print(f"Optimized Portfolio Volatility: {optimal_portfolio['Volatility'] * 100:.2f}%")
print(f"Optimized Portfolio Sharpe Ratio: {optimal_portfolio['Sharpe Ratio']:.2f}")

plt.figure(figsize=(12, 8))
plt.scatter(simulated_portfolios['Volatility'], simulated_portfolios['Return'], c=simulated_portfolios['Sharpe Ratio'], cmap='YlGnBu')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')

plt.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Return'], color='red', marker='*', s=200, label='Optimal Portfolio')
plt.legend()
plt.show()
