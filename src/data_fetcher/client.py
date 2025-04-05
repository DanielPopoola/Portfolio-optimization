import requests
import pandas as pd

from typing import  List, Dict, Union
from datetime import date, datetime, timedelta
from dateutil.relativedelta import *

from fake_useragent import UserAgent

class PriceHistory():
    " ""This is a simple class for scraping price data from NASDAQ website """
    def __init__(self, symbols: List[str], user_agent: UserAgent) -> None:
        """
        Initalizes the PriceHistory client.

        Arguments:
        ----
        symbols (List[str]): A list of ticker symbols to pull
        quotes for.
        """
        self._api_url = 'https://api.nasdaq.com/api/quote'
        self._api_service = 'historical'
        self._symbols = symbols
        self._user_agent = user_agent
        self.price_data_frame = self._build_data_frame()

    def _build_url(self,symbol: str) -> str:
        """Builds a Full URL.

        ### Arguments:
        ----
        symbol (str): The symbol you want to build a URL for.

        ### Returns:
        ----
        str: A URL to the Ticker symbol provided.
        """
        parts = [self._api_url, symbol, self._api_service]
        return '/'.join(parts)

    @property
    def symbols(self) -> List[str]:
        """Returns all the symbols currently being pulled.

        ### Returns:
        ----
        List[str]: A list of ticker symbols.
        """
        return self._symbols

    def _build_data_frame(self) -> pd.DataFrame:
        """Builds a data frame with all the price data.

        ### Returns:
        ----
        pd.DataFrame: A Pandas DataFrame with the data cleaned
            and sorted.
        """
        all_data = []
        to_date = datetime.today().date()

        #Calculate the start and end point
        from_date = to_date - relativedelta(months=6)

        for symbol in self._symbols:
            all_data = self._grab_prices(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
            ) + all_data

        price_data_frame =  pd.DataFrame(all_data)
        price_data_frame['date'] = pd.to_datetime(price_data_frame['date'])

        return price_data_frame

    def _grab_prices(self,symbol: str, from_date: date, to_date: date) -> List[Dict]:
        """Grabs the prices.

        ### Arguments:
        ----
        symbol (str): The symbol to pull prices for.

        from_date (date): The starting date to pull prices.

        to_date (date): The ending data to pull prices for.

        ### Returns:
        ----
        List[Dict]: A list of candle dictionaries.
        """
        price_url = self._build_url(symbol)

        # Calculate the limit
        limit: timedelta = to_date - from_date

        # Define parameters
        params = {
            'fromdate': from_date.isoformat(),
            'todate': to_date.isoformat(),
            'assetclass':'stocks',
            'limit':limit.days
        }

        # Fake the headers
        headers ={'user-agent': self._user_agent}

        # Grab historical price data
        historical_data = requests.get(
            url=price_url,
            params=params,
            headers=headers,
            verify=True
        )

        # If it's okay parse it.
        if historical_data.ok:
            historical_data = historical_data.json()
            historical_data = historical_data['data']['tradesTable']['rows']

            # Clean the data.
            for table_row in historical_data:
                table_row['symbol'] = symbol
                table_row['close'] = float(table_row['close'].replace('$', ''))
                table_row['open'] = float(table_row['open'].replace('$', ''))
                table_row['high'] = float(table_row['high'].replace('$', ''))
                table_row['low'] = float(table_row['low'].replace('$', ''))
                table_row['volume'] = int(table_row['volume'].replace(',', ''))

            return historical_data