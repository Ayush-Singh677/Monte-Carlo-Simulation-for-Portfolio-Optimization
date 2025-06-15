from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from config import (
    TOTAL_PORTFOLIO_VALUE_FOR_ALLOCATION, 
    RISK_FREE_RATE,
    MAX_WEIGHT,
    MIN_WEIGHT,
    ALLOW_SHORTING,
    SOLVER,
    SOLVER_OPTIONS,
    TRADING_DAYS_PER_YEAR
)
from logger import logger
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, data, config, risk_free_rate):
        """
        Initialize the portfolio optimizer.
        
        Args:
            data (pd.DataFrame): Historical returns data
            config (Config): Configuration object
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        self.data = data
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.mean_returns = data.mean()
        self.cov_matrix = self._regularize_covariance(data.cov())
        
    def _regularize_covariance(self, cov_matrix):
        """
        Regularize the covariance matrix to ensure positive definiteness.
        
        Args:
            cov_matrix (pd.DataFrame): Original covariance matrix
            
        Returns:
            pd.DataFrame: Regularized covariance matrix
        """
        # Add small diagonal elements to ensure positive definiteness
        min_eigenvalue = np.min(np.linalg.eigvals(cov_matrix))
        if min_eigenvalue < 0:
            regularization = abs(min_eigenvalue) + 1e-6
            cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * regularization
            logger.info(f"Applied regularization of {regularization:.6f} to covariance matrix")
        return cov_matrix
    
    def _portfolio_metrics(self, weights):
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            tuple: (expected_return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def _objective_function(self, weights):
        """
        Objective function for optimization (negative Sharpe ratio).
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            float: Negative Sharpe ratio
        """
        portfolio_return, portfolio_vol, sharpe_ratio = self._portfolio_metrics(weights)
        return -sharpe_ratio
    
    def _constraints(self):
        """
        Define optimization constraints.
        
        Returns:
            list: List of constraint dictionaries
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        # Add position limits if specified
        if hasattr(self.config, 'MAX_POSITION_SIZE'):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.config.MAX_POSITION_SIZE - x
            })
        
        return constraints
    
    def _bounds(self):
        """
        Define optimization bounds.
        
        Returns:
            list: List of (min, max) tuples for each weight
        """
        return [(0, 1) for _ in range(len(self.mean_returns))]
    
    def optimize(self):
        """
        Optimize portfolio weights for maximum Sharpe ratio.
        
        Returns:
            np.array: Optimal portfolio weights
        """
        n_assets = len(self.mean_returns)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            self._objective_function,
            initial_weights,
            method='SLSQP',
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        return result.x
    
    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            dict: Dictionary containing portfolio metrics
        """
        expected_return, volatility, sharpe_ratio = self._portfolio_metrics(weights)
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def optimize_portfolio(self):
        """Optimize portfolio using Modern Portfolio Theory."""
        try:
            # Calculate returns and clean data
            returns = self.data.pct_change().dropna()
            returns = returns[~returns.isin([float('inf'), float('-inf')]).any(axis=1)]
            returns = returns.dropna()
            
            # Additional cleaning
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Verify data is clean
            if returns.isna().any().any() or np.isinf(returns).any().any():
                logger.error("Returns contain NaN or inf values after cleaning")
                raise ValueError("Invalid returns data")
            
            # Calculate annualized expected returns and covariance
            expected_returns = returns.mean() * TRADING_DAYS_PER_YEAR
            cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR
            
            # Log diagnostics
            logger.info(f"Expected returns: {expected_returns.to_dict()}")
            logger.info(f"Covariance matrix shape: {cov_matrix.shape}")
            logger.info(f"Covariance matrix condition number: {np.linalg.cond(cov_matrix)}")
            
            # Create efficient frontier
            ef = EfficientFrontier(expected_returns, cov_matrix)
            
            # Set weight bounds based on configuration
            if ALLOW_SHORTING:
                weight_bounds = (-1, 1)  # Allow short positions
            else:
                weight_bounds = (MIN_WEIGHT, MAX_WEIGHT)
            
            ef.add_constraint(lambda w: cp.sum(w) == 1)  # weights sum to 1
            ef.add_constraint(lambda w: cp.sum(cp.abs(w)) <= 1.5)  # limit leverage
            
            # Try min_volatility first
            try:
                logger.info("Attempting minimum volatility optimization...")
                weights = ef.min_volatility()
                logger.info("Minimum volatility optimization successful")
            except Exception as e:
                logger.warning(f"Minimum volatility optimization failed: {str(e)}")
                logger.info("Falling back to maximum Sharpe ratio optimization...")
                weights = ef.max_sharpe()
                logger.info("Maximum Sharpe ratio optimization successful")
            
            # Clean weights
            weights = ef.clean_weights()
            logger.info(f"Final weights: {weights}")
            
            return weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise

    def get_portfolio_performance(self):
        """Get portfolio performance metrics."""
        if self.ef is None:
            self.optimize_portfolio()
        
        expected_return, volatility, sharpe_ratio = self.ef.portfolio_performance(
            verbose=False, risk_free_rate=RISK_FREE_RATE
        )
        performance = {
            "Expected annual return": expected_return,
            "Annual volatility": volatility,
            "Sharpe Ratio": sharpe_ratio
        }
        return performance

    def get_discrete_allocation(self, portfolio_value):
        """Calculate discrete allocation for the optimized portfolio."""
        # Use the cleaned weights from optimize_portfolio
        weights = self.optimize_portfolio()
        allocation = DiscreteAllocation(
            weights,
            self.data.iloc[-1],
            total_portfolio_value=portfolio_value
        )
        return allocation.lp_portfolio() 