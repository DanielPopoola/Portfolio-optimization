import pathlib
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci_opt

from pprint import pprint
from fake_useragent import UserAgent
from src.data_fetcher.client import PriceHistory


def get_metrics(weights: list) -> np.array:
    """
    Overview:
    ----
    With a given set of weights, return the portfolio returns,
    the portfolio volatility, and the portfolio sharpe ratio.

    Arguments:
    ----
    weights (list): An array of portfolio weights.

    Returns:
    ----
    (np.array): An array containing return value, a volatility value,
        and a sharpe ratio.
    """
    weights = np.array(weights)

    ret = np.dot(log_return.mean(), weights) * 252

    vol = np.sqrt(
        np.dot(weights.T, np.dot(log_return.cov() * 252, weights))
    )

    sr = ret / vol

    return np.array([ret, vol, sr])


def neg_sharpe(weights: list) -> np.array:
    """The function used to minimize the Sharpe Ratio.

    Arguments:
    ----
    weights (list): The weights, we are testing to see
        if it's the minimum.

    Returns:
    ----
    (np.array): An numpy array of the portfolio metrics.
    """
    return get_metrics(weights)[2] * -1


def check_sum(weights: list) -> float:
    """Ensure the allocations of the "weights", sums to 1 (100%)

    Arguments:
    ----
    weights (list): The weights we want to check to see
        if they sum to 1.

    Returns:
    ----
    float: The different between 1 and the sum of the weights.
    """
    return np.sum(weights) - 1


# Define the symbols
symbols = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']

# Grab number of stocks in portfolio
number_of_symbols = len(symbols)

# If no data is available, grab it from the NASDAQ.
if not pathlib.Path('src/data/stock_data.csv').exists():
    # Initialize PriceHistory Client
    price_history_client = PriceHistory(symbols=symbols, user_agent=UserAgent().chrome)
    # Grab and dump data in to a csv file.
    price_history_client.price_data_frame.to_csv('src/data/stock_data.csv', index=False)
    pprint(price_history_client.price_data_frame)

    # store dataframe in variable
    price_df: pd.DataFrame = price_history_client.price_data_frame

else:
    # load the existing CSV file
    price_df: pd.DataFrame = pd.read_csv('src/data/stock_data.csv')

# Grab important columns
price_df = price_df[['date', 'symbol', 'close']]

# Pivot dataframe to make symbols headers
price_df = price_df.pivot(index='date', columns='symbol', values='close')

# Calculate the Log of returns.
log_return = np.log(1 + price_df.pct_change())

# Generate Random Weights.
random_weights = np.array(np.random.random(number_of_symbols))

# Generate the Rebalance Weights, these should equal 1.
rebalance_weights = random_weights / np.sum(random_weights)

# Calculate the Expected Returns, annualize it by multiplying it by `252`.
exp_ret = np.sum((log_return.mean() * rebalance_weights) * 252)

# Calculate the Expected Volatility, annualize it by multiplying it by `252`.
exp_vol = np.sqrt(
    np.dot(
        rebalance_weights.T,
        np.dot(
            log_return.cov() * 252,
            rebalance_weights
        )
    )
)

# Calculate the Sharpe Ratio.
sharpe_ratio = exp_ret / exp_vol

# Put the weights into a data frame to see them better.
weights_df = pd.DataFrame(data={
    'random_weights': random_weights,
    'rebalance_weights': rebalance_weights
})
print(weights_df)
print('')

metrics_df = pd.DataFrame(data={
    'Expected Portfolio Returns': exp_ret,
    'Expected Portfolio Volatility': exp_vol,
    'Portfolio Sharpe Ratio': sharpe_ratio
}, index=[0])
print(metrics_df)

# Initialize the components, to run a Monte Carlo Simulation.
# We will run 5000 iterations.
num_of_portfolios = 5000

# Prep an array to store the weights as they are generated.
all_weights = np.zeros((num_of_portfolios, number_of_symbols))

# Prep an array to store the returns as they are generated.
ret_arr = np.zeros(num_of_portfolios)

# Prep an array to store the volatilities as they are generated.
vol_arr = np.zeros(num_of_portfolios)

# Prep an array to store the sharpe ratios as they are generated.
sharpe_arr = np.zeros(num_of_portfolios)

# Start the simulations.
for ind in range(num_of_portfolios):

    # First, calculate the weights.
    weights = np.array(np.random.random(number_of_symbols))
    weights = weights / np.sum(weights)

    # Add the weights, to the `weights_arrays`.
    all_weights[ind, :] = weights

    # Calculate the expected returns, and add them to the `returns_array`.
    ret_arr[ind] = np.dot(log_return.mean(), weights) * 252

    # Calculate the volatility, and add them to the `volatility_array`.
    vol_arr[ind] = np.sqrt(
        np.dot(weights.T, np.dot(log_return.cov() * 252, weights))
    )

    # Calculate the Sharpe Ratio and Add it to the `sharpe_ratio_array`.
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

# Let's add all the different parts together.
simulations_data = [ret_arr, vol_arr, sharpe_arr, all_weights]

# Create a DataFrame from it.
simulations_df = pd.DataFrame(data=simulations_data).T
simulations_df.columns = [
    'Returns',
    'Volatility',
    'Sharpe Ratio',
    'Portfolio Weights'
]
simulations_df = simulations_df.infer_objects()
print(simulations_df.head())

# Return the Max Sharpe Ratio from the run.
max_sharpe_ratio = simulations_df.loc[simulations_df['Sharpe Ratio'].idxmax()]

# Return the Min Volatility from the run.
min_volatility = simulations_df.loc[simulations_df['Volatility'].idxmin()]

print('')
print('='*80)
print('MAX SHARPE RATIO:')
print('-'*80)
print(max_sharpe_ratio)
print('')
print('='*80)
print('MIN VOLATILITY:')
print('-'*80)
print(min_volatility)

# Plot the data on a Scatter plot.
plt.scatter(
    y=simulations_df['Returns'],
    x=simulations_df['Volatility'],
    c=simulations_df['Sharpe Ratio'],
    cmap='RdYlBu'
)

# Plot some details.
plt.title('Portfolio Returns Vs. Risk')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')

# Plot the Max Sharpe Ratio.
plt.scatter(
    max_sharpe_ratio[1],
    max_sharpe_ratio[0],
    marker=(5, 1, 0),
    color='r',
    s=600
)

# Plot the Min Volatility.
plt.scatter(
    min_volatility[1],
    min_volatility[0],
    marker=(5, 1, 0),
    color='b',
    s=600
)

plt.show()

# Define the boundaries.
bounds = tuple((0, 1) for symbol in range(number_of_symbols))

# Define the constraints.
constraints = ({'type': 'eq', 'fun': check_sum})

# We need to create an initial guess to start with,
# and usually the best initial guess is just an
# even distribution
init_guess: list[float] = number_of_symbols * [1 / number_of_symbols]

# Perform the operation to minimize the risk.
optimized_sharpe = sci_opt.minimize(
    neg_sharpe,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Print the results.
print(optimized_sharpe)

# Grab the metrics.
optimized_metrics = get_metrics(weights=optimized_sharpe.x)
print(optimized_metrics)