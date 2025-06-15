import pandas as pd
import numpy as np
import logging
from portfolio_optimizer import PortfolioOptimizer
import config

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, price_data):
        """
        Initializes the Backtester.
        
        Args:
            price_data (pd.DataFrame): DataFrame of historical prices for all assets.
        """
        self.price_data = price_data
        self.start_date = pd.to_datetime(config.BACKTEST_START_DATE)
        self.initial_capital = config.INITIAL_CAPITAL
        self.window = config.ROLLING_WINDOW_YEARS * 252  # In trading days
        self.rebalance_freq = config.REBALANCE_FREQUENCY

    def run_backtest(self):
        """
        Runs the rolling-window backtest.
        """
        logger.info("Starting rolling-window backtest...")
        
        # Get rebalancing dates based on the specified frequency
        rebalance_dates = self.price_data[self.start_date:].resample(self.rebalance_freq).first().index
        
        portfolio_value = pd.Series(index=self.price_data.index)
        weights_over_time = pd.DataFrame(index=rebalance_dates, columns=self.price_data.columns)
        
        capital = self.initial_capital
        last_rebalance_end = rebalance_dates[0]

        for i in range(len(rebalance_dates)):
            rebal_date = rebalance_dates[i]
            
            # Define the training period for optimization
            train_start = rebal_date - pd.DateOffset(years=config.ROLLING_WINDOW_YEARS)
            training_data = self.price_data.loc[train_start:rebal_date]
            
            if training_data.shape[0] < self.window * 0.9: # Ensure enough data for a robust optimization
                logger.warning(f"Skipping rebalance at {rebal_date}: Not enough training data ({training_data.shape[0]} days).")
                continue

            # Optimize portfolio
            returns = training_data.pct_change().dropna()
            optimizer = PortfolioOptimizer(
                data=returns,
                config=config,
                risk_free_rate=config.RISK_FREE_RATE
            )
            optimal_weights = optimizer.optimize()
            weights_over_time.loc[rebal_date] = optimal_weights
            
            # Define holding period
            holding_start = last_rebalance_end
            holding_end = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.price_data.index[-1]
            
            # Simulate
            period_prices = self.price_data.loc[holding_start:holding_end]
            period_returns = period_prices.pct_change().dropna()
            
            if period_returns.empty:
                continue

            num_shares = (capital * optimal_weights) / period_prices.iloc[0]
            portfolio_daily_value = (num_shares * period_prices).sum(axis=1)
            
            portfolio_value.loc[portfolio_daily_value.index] = portfolio_daily_value
            capital = portfolio_daily_value.iloc[-1]
            last_rebalance_end = holding_end

        # --- Final Metrics Calculation ---
        portfolio_value = portfolio_value.dropna()
        
        if portfolio_value.empty:
            logger.error("Backtest failed to produce any results, likely due to insufficient data for all periods.")
            # Return a dictionary of NaNs or zeros to prevent crashes
            nan_results = {k: 0 for k in ['final_value', 'total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']}
            nan_results['portfolio_values'] = pd.Series()
            nan_results['weights_over_time'] = pd.DataFrame()
            return nan_results

        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_value)) - 1
        daily_returns = portfolio_value.pct_change().dropna()
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - config.RISK_FREE_RATE) / volatility if volatility != 0 else 0
        
        # Calculate maximum drawdown
        peak = portfolio_value.cummax()
        drawdown = (peak - portfolio_value) / peak
        max_drawdown = np.max(drawdown)

        logger.info("Backtest complete.")
        return {
            'portfolio_values': portfolio_value,
            'weights_over_time': weights_over_time.dropna(),
            'final_value': portfolio_value.iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        } 