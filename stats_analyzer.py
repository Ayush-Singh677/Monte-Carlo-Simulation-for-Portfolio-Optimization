import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import logging
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from logger import logger

class PortfolioStatsAnalyzer:
    def __init__(self, returns: pd.DataFrame, benchmark_returns: pd.Series = None):
        """
        Initialize the analyzer with portfolio returns and optional benchmark returns.
        
        Args:
            returns (pd.DataFrame): Portfolio returns
            benchmark_returns (pd.Series, optional): Benchmark returns (e.g., SPY)
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.portfolio_returns = returns.mean(axis=1)  # Equal-weighted portfolio returns
        
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic portfolio metrics."""
        metrics = {}
        
        # Annualized returns
        metrics['annualized_return'] = (1 + self.portfolio_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
        
        # Volatility (annualized)
        metrics['annualized_volatility'] = self.portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Sharpe Ratio
        excess_returns = self.portfolio_returns - RISK_FREE_RATE/TRADING_DAYS_PER_YEAR
        metrics['sharpe_ratio'] = np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()
        
        # Sortino Ratio
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        metrics['sortino_ratio'] = np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / downside_returns.std()
        
        return metrics
    
    def calculate_advanced_metrics(self) -> Dict:
        """Calculate advanced portfolio metrics."""
        metrics = {}
        
        # Maximum Drawdown
        cum_returns = (1 + self.portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()
        
        # Calmar Ratio
        metrics['calmar_ratio'] = self.calculate_basic_metrics()['annualized_return'] / abs(metrics['max_drawdown'])
        
        # Value at Risk (95% and 99%)
        metrics['var_95'] = np.percentile(self.portfolio_returns, 5)
        metrics['var_99'] = np.percentile(self.portfolio_returns, 1)
        
        # Expected Shortfall (CVaR)
        metrics['cvar_95'] = self.portfolio_returns[self.portfolio_returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = self.portfolio_returns[self.portfolio_returns <= metrics['var_99']].mean()
        
        # Skewness and Kurtosis
        metrics['skewness'] = stats.skew(self.portfolio_returns)
        metrics['kurtosis'] = stats.kurtosis(self.portfolio_returns)
        
        # Information Ratio (if benchmark provided)
        if self.benchmark_returns is not None:
            active_returns = self.portfolio_returns - self.benchmark_returns
            metrics['information_ratio'] = np.sqrt(TRADING_DAYS_PER_YEAR) * active_returns.mean() / active_returns.std()
        
        return metrics
    
    def calculate_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling metrics for the portfolio.
        
        Args:
            window (int): Rolling window size in days (default: 252 for annual)
        """
        rolling_metrics = pd.DataFrame(index=self.portfolio_returns.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = self.portfolio_returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = self.portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Rolling Sharpe
        rolling_metrics['rolling_sharpe'] = (rolling_metrics['rolling_return'] - RISK_FREE_RATE) / rolling_metrics['rolling_volatility']
        
        # Rolling drawdown
        cum_returns = (1 + self.portfolio_returns).cumprod()
        rolling_max = cum_returns.rolling(window).max()
        rolling_metrics['rolling_drawdown'] = (cum_returns / rolling_max - 1)
        
        return rolling_metrics
    
    def calculate_factor_analysis(self) -> Dict:
        """Calculate factor analysis metrics."""
        if self.benchmark_returns is None:
            logger.warning("Benchmark returns not provided. Skipping factor analysis.")
            return {}
        
        metrics = {}
        
        # Beta calculation
        covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0,1]
        benchmark_variance = np.var(self.benchmark_returns)
        metrics['beta'] = covariance / benchmark_variance
        
        # Alpha calculation
        metrics['alpha'] = (self.portfolio_returns.mean() - 
                          RISK_FREE_RATE/TRADING_DAYS_PER_YEAR - 
                          metrics['beta'] * (self.benchmark_returns.mean() - RISK_FREE_RATE/TRADING_DAYS_PER_YEAR))
        metrics['alpha'] = (1 + metrics['alpha']) ** TRADING_DAYS_PER_YEAR - 1  # Annualize
        
        # R-squared
        metrics['r_squared'] = np.corrcoef(self.portfolio_returns, self.benchmark_returns)[0,1] ** 2
        
        return metrics
    
    def calculate_turnover_analysis(self, weights: pd.DataFrame) -> Dict:
        """
        Calculate portfolio turnover metrics.
        
        Args:
            weights (pd.DataFrame): Historical portfolio weights
        """
        metrics = {}
        
        # Calculate weight changes
        weight_changes = weights.diff().abs()
        
        # Portfolio turnover
        metrics['avg_turnover'] = weight_changes.sum(axis=1).mean()
        metrics['max_turnover'] = weight_changes.sum(axis=1).max()
        
        # Asset-specific turnover
        metrics['asset_turnover'] = weight_changes.mean()
        
        return metrics
    
    def calculate_risk_decomposition(self, weights: pd.Series) -> Dict:
        """
        Decompose portfolio risk into individual asset contributions.
        
        Args:
            weights (pd.Series): Portfolio weights
        """
        # Portfolio variance
        portfolio_variance = weights.dot(self.returns.cov()).dot(weights)
        
        # Marginal contribution to risk
        mctr = self.returns.cov().dot(weights) / np.sqrt(portfolio_variance)
        
        # Component contribution to risk
        cctr = weights * mctr
        
        return {
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': np.sqrt(portfolio_variance),
            'marginal_contribution_to_risk': mctr,
            'component_contribution_to_risk': cctr
        }
    
    def get_all_metrics(self, weights: pd.Series) -> Dict:
        """
        Calculate all available metrics for the portfolio.
        
        Args:
            weights (pd.Series): Portfolio weights
        """
        all_metrics = {}
        
        # Basic metrics
        all_metrics.update(self.calculate_basic_metrics())
        
        # Advanced metrics
        all_metrics.update(self.calculate_advanced_metrics())
        
        # Factor analysis
        all_metrics.update(self.calculate_factor_analysis())
        
        # Risk decomposition
        all_metrics.update(self.calculate_risk_decomposition(weights))
        
        return all_metrics 