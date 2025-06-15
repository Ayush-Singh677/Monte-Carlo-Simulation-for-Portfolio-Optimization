import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class PortfolioPlotter:
    def __init__(self, price_data, backtest_results):
        """
        Initializes the portfolio plotter.
        
        Args:
            price_data (pd.DataFrame): The original historical price data.
            backtest_results (dict): The results from the backtester.
        """
        self.price_data = price_data
        self.results = backtest_results
        self.portfolio_values = backtest_results['portfolio_values']
        self.weights_over_time = backtest_results['weights_over_time']
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def plot_portfolio_performance(self):
        """Plot portfolio value (equity curve) over time."""
        plt.figure(figsize=(12, 6))
        self.portfolio_values.plot()
        plt.title('Portfolio Value Over Time (Equity Curve)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'portfolio_performance.png'))
        plt.close()
    
    def plot_drawdown(self):
        """Plot portfolio drawdown over time."""
        peak = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - peak) / peak
        
        plt.figure(figsize=(12, 6))
        drawdown.plot(kind='area', alpha=0.5)
        plt.title('Portfolio Drawdown Over Time')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'drawdown.png'))
        plt.close()
    
    def plot_rolling_volatility(self):
        """Plot rolling 30-day volatility."""
        returns = self.portfolio_values.pct_change()
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        
        plt.figure(figsize=(12, 6))
        rolling_vol.plot()
        plt.title('30-Day Rolling Annualized Volatility')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rolling_volatility.png'))
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of assets based on the full history."""
        returns = self.price_data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
        plt.title('Asset Correlation Matrix (Full History)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'correlation_heatmap.png'))
        plt.close()
    
    def plot_weight_evolution(self):
        """Plot the evolution of portfolio weights over time as a stacked area chart."""
        plt.figure(figsize=(12, 6))
        self.weights_over_time.plot(kind='area', stacked=True, figsize=(12, 6))
        plt.title('Portfolio Weight Evolution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'weight_evolution.png'))
        plt.close() 