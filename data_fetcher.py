import yfinance as yf
import pandas as pd
from config import PORTFOLIO, START_DATE, END_DATE, BENCHMARK_TICKER, DATA_CACHE_PATH, CACHE_EXPIRATION_DAYS
import os
import time
from logger import logger

class DataFetcher:
    def __init__(self):
        self.portfolio = PORTFOLIO + [BENCHMARK_TICKER]
        self.start_date = START_DATE
        self.end_date = END_DATE
        self.data = None

    def _is_cache_valid(self):
        """Check if the cache file exists and is not expired."""
        if not os.path.exists(DATA_CACHE_PATH):
            return False
        
        cache_age = time.time() - os.path.getmtime(DATA_CACHE_PATH)
        return cache_age < (CACHE_EXPIRATION_DAYS * 86400)

    def fetch_data(self):
        """Fetch historical data for the portfolio, using cache if available."""
        if self._is_cache_valid():
            logger.info(f"Loading data from cache: {DATA_CACHE_PATH}")
            self.data = pd.read_pickle(DATA_CACHE_PATH)
            return self.data

        logger.info("Fetching new data from Yahoo Finance...")
        try:
            # Download all data at once
            logger.info(f"Downloading data for {', '.join(self.portfolio)}...")
            self.data = yf.download(self.portfolio, self.start_date, self.end_date, auto_adjust=True)
            
            if self.data.empty:
                logger.error("No data downloaded. Check tickers and date range.")
                return None
            
            # Extract just the Close prices
            self.data = self.data['Close']
            
            logger.info(f"Saving data to cache: {DATA_CACHE_PATH}")
            self.data.to_pickle(DATA_CACHE_PATH)
            return self.data
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None

    def get_close_data(self):
        """Get close prices."""
        if self.data is None:
            self.fetch_data()
        if self.data is None:
            return None
        return self.data.dropna()

    def get_returns(self, data=None):
        """Calculate returns. Can be used for subsets of data (e.g., training/testing)."""
        if data is None:
            data = self.get_close_data()
        if data is None:
            return None
        
        returns = data.pct_change().dropna()
        return returns

    def get_covariance_matrix(self):
        """Calculate covariance matrix from returns."""
        returns = self.get_returns()
        return returns.cov()

    def get_mean_returns(self):
        """Calculate mean returns."""
        returns = self.get_returns()
        return returns.mean()

    def get_latest_prices(self):
        """Get latest prices for portfolio optimization."""
        if self.data is None:
            self.fetch_data()
        return self.data.iloc[-1] 