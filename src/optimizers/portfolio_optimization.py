import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sci_opt
import io
import base64


class PortfolioOptimizer:
    def __init__(self, price_data: pd.DataFrame):
        """Initialize with price data dataframe."""
        self.price_df = price_data
        # Process data
        self.price_df = self.price_df[['date', 'symbol', 'close']]
        self.price_df = self.price_df.pivot(index='date', columns='symbol', values='close')
        self.log_return = np.log(1 + self.price_df.pct_change())
        self.symbols = list(self.price_df.columns)
        self.number_of_symbols = len(self.symbols)

    def get_metrics(self, weights: list) -> np.array:
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
        (np.array): An array containg return value, a volatility value,
            and a sharpe ratio.
        """

        # Convert to a Numpy Array.
        weights = np.array(weights)

        # Calculate the returns, remember to annualize them (252).
        ret = np.dot(self.log_return.mean(), weights) * 252

        # Calculate the volatility, remember to annualize them (252).
        vol = np.sqrt(
            np.dot(weights.T, np.dot(self.log_return.cov() * 252, weights))
        )

        # Calculate the Sharpe Ratio.
        sr = ret / vol

        return np.array([ret, vol, sr])

    def neg_sharpe(self, weights: list) -> np.array:
        """The function used to minimize the Sharpe Ratio.

        Arguments:
        ----
        weights (list): The weights, we are testing to see
            if it's the minimum.

        Returns:
        ----
        (np.array): An numpy array of the portfolio metrics.
        """
        return self.get_metrics(weights)[2] - 1

    def check_sum(self, weights: list) -> float:
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

    def random_portfolio(self):
        """Generate a random portfolio weights"""
        random_weights = np.array(np.random.rand(self.number_of_symbols))
        rebalance_weights = random_weights / np.sum(random_weights)

        metrics = self.get_metrics(rebalance_weights)

        return {
            'weights': rebalance_weights,
            'returns': metrics[0],
            'volatility': metrics[1],
            'sharpe_ratio': metrics[2],
            'weights_by_symbol': dict(zip(self.symbols, rebalance_weights))
        }

    def monte_carlo_simulations(self, num_portfolios=5000):
        """Run Monte Carlo simulation to find optimal portfolios."""
        all_weights = np.zeros((num_portfolios, self.number_of_symbols))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        for idx in range(num_portfolios):
            weights = np.array(np.random.random(self.number_of_symbols))
            weights = weights / np.sum(weights)
            all_weights[idx, :] = weights

            ret_arr[idx] = np.dot(self.log_return.mean(), weights) * 252
            vol_arr[idx] = np.sqrt(
                np.dot(weights.T, np.dot(self.log_return.cov() * 252, weights))
            )
            sharpe_arr[idx] = ret_arr[idx] / vol_arr[idx]

        # Create results DataFrame
        results = pd.DataFrame({
            'Returns': ret_arr,
            'Volatility': vol_arr,
            'Sharpe Ratio': sharpe_arr
        })

        # Add weight columns
        for i, symbol in enumerate(self.symbols):
            results[f'Weight_{symbol}'] = all_weights[:, i]

        # Find max sharpe ratio and min volatility portfolio
        max_sharpe_idx = results['Sharpe Ratio'].idxmax()
        min_vol_idx = results['Volatility'].idxmin()

        max_sharpe_portfolio = {
            'returns': results.loc[max_sharpe_idx, 'Returns'],
            'volatility': results.loc[max_sharpe_idx, 'Volatility'],
            'sharpe_ratio': results.loc[max_sharpe_idx, 'Sharpe Ratio'],
            'weights_by_symbol': {symbol: results.loc[max_sharpe_idx, f'Weight_{symbol}'] for symbol in self.symbols}
        }

        min_vol_portfolio = {
            'returns': results.loc[min_vol_idx, 'Returns'],
            'volatility': results.loc[min_vol_idx, 'Volatility'],
            'sharpe_ratio': results.loc[min_vol_idx, 'Sharpe Ratio'],
            'weights_by_symbol': {symbol: results.loc[min_vol_idx, f'Weight_{symbol}'] for symbol in self.symbols}
        }

        return {
            'simulation': results,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'min_vol_portfolio': min_vol_portfolio
        }

    @property
    def optimize_portfolio(self):
        """Use scipy to find optimal portfolio weights"""
        bounds = tuple((0, 1) for _ in range(self.number_of_symbols))
        constraints = ({'type': 'eq', 'fun': self.check_sum})
        init_guess = self.number_of_symbols * [1 / self.number_of_symbols]

        optimized_sharpe = sci_opt.minimize(
            self.neg_sharpe,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        metrics = self.get_metrics(optimized_sharpe.x)

        return {
            'success': optimized_sharpe.success,
            'weights': optimized_sharpe.x,
            'returns': metrics[0],
            'volatility': metrics[1],
            'sharpe_ratio': metrics[2],
            'weights_by_symbol': dict(zip(self.symbols, optimized_sharpe.x))
        }

    def generate_efficient_frontier_plot(self, results):
        """Generate efficient frontier plot and return as base64 image."""
        plt.figure(figsize=(10, 8))

        # Create scatter plot
        plt.scatter(
            x=results['simulations']['Volatility'],
            y=results['simulations']['Returns'],
            c=results['simulations']['Sharpe Ratio'],
            cmap='RdYlBu',
            alpha=0.7
        )

        # Plot max sharpe and min vol portfolios
        plt.scatter(
            results['max_sharpe_portfolio']['volatility'],
            results['max_sharpe_portfolio']['returns'],
            marker='*',
            color='r',
            s=300,
            label='Max Sharpe Ratio'
        )

        plt.scatter(
            results['min_volatility_portfolio']['volatility'],
            results['min_volatility_portfolio']['returns'],
            marker='*',
            color='b',
            s=300,
            label='Min Volatility'
        )

        plt.title('Portfolio Optimization - Efficient Frontier')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Returns')
        plt.colorbar(label='Sharpe Ratio')
        plt.legend()
        plt.grid(True)

        # Convert plot to base64 string for HTML embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64decode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f'data:image/png;base64,{image_base64}'
