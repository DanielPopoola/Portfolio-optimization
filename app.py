from  flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pathlib
import numpy as np
from src.optimizers.portfolio_optimization import PortfolioOptimizer
from src.data_fetcher.client import PriceHistory
from fake_useragent import UserAgent
import json

app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

@app.route('/')
def index():
    """Main page with form for portfolio optimization."""
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    """Process form data and run the portfolio optimization."""
    # Get form data
    symbols = request.form.get('symbols', '').replace(' ', '').split(',')

    # Handle empty input
    if not symbols or symbols == ['']:
        symbols = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']

    # Get data (from file or API)
    data_file = pathlib.Path('data/stock_data.csv')

    if not data_file.exists() or request.form.get('refresh_data') == 'yes':
        # Fetch fresh data
        try:
            price_history_client = PriceHistory(symbols=symbols, user_agent=UserAgent().chrome)
            price_df = price_history_client.price_data_frame
            price_df.to_csv('data/stock_data.csv', index=False)
        except Exception as e:
            return  render_template('index.html', error=f"Error fetching data: {str(e)}")
    else:
        # Use existing data
        price_df = pd.read_csv('data/stock_data.csv')
        # Filter for requested symbols
        price_df = price_df[price_df['symbol'].isin(symbols)]

    # Check if we have enough data
    if len(price_df['symbol'].unique()) < 2:
        return render_template('index.html',
                               error="Not enough stock data available. Please try different symbols.")

    # Create optimizer
    optimizer = PortfolioOptimizer(price_df)

    # Run Monte Carlo simulation
    num_portfolios = int(request.form.get('num_portfolios', 5000))
    mc_results = optimizer.monte_carlo_simulations(num_portfolios)

    # Get optimized portfolio
    optimized = optimizer.optimize_portfolio()

    # Generate efficient frontier plot
    plot_img = optimizer.generate_efficient_frontier_plot(mc_results)

    # Prepare results for template
    return render_template(
        'results.html',
        symbols=symbols,
        plot_img=plot_img,
        max_sharpe=mc_results['max_sharpe_portfolio'],
        min_vol=mc_results['min_vol_portfolio'],
        optimized=optimized
    )

if __name__ == '__main__':
    pathlib.Path('data').mkdir(exist_ok=True)
    app.run(debug=True)
