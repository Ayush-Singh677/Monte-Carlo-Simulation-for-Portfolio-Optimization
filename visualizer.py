import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_STYLE, FIGURE_SIZE, DPI, PLOTS_PATH, EFFICIENT_FRONTIER_SAMPLES
import os
from pypfopt import plotting
from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
from logger import logger

class PortfolioVisualizer:
    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.plots_path = PLOTS_PATH

    def _save_plot(self, fig, filename):
        """Helper function to save a plot."""
        path = os.path.join(self.plots_path, filename)
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        logger.info(f"Saved plot to {path}")
        plt.close(fig)

    def plot_monte_carlo_simulation(self, portfolio_sims):
        """Plot Monte Carlo simulation results."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.plot(portfolio_sims)
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Days')
        ax.set_title('Monte Carlo Simulation of Portfolio Value')
        ax.grid(True)
        self._save_plot(fig, 'monte_carlo_simulation.png')

    def plot_portfolio_weights(self, weights):
        """Plot portfolio weights as a pie chart."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('Optimal Portfolio Weights Distribution')
        self._save_plot(fig, 'portfolio_weights.png')

    def plot_efficient_frontier(self, optimizer):
        """Plot the Efficient Frontier."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Create a new instance of EfficientFrontier for plotting
        ef_plot = EfficientFrontier(optimizer.exp_returns, optimizer.cov_matrix)
        
        # Plot the efficient frontier
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
        
        # Find the tangency portfolio
        ret_tangent, std_tangent, _ = optimizer.get_portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=200, c="r", label="Max Sharpe")

        ax.set_title("Efficient Frontier with Asset Tickers")
        ax.legend()
        self._save_plot(fig, 'efficient_frontier.png')

    def plot_backtest_performance(self, backtest_results):
        """Plot the portfolio value vs benchmark from the backtest."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        backtest_results.plot(ax=ax, colormap='viridis')
        ax.set_title('Backtest: Portfolio Performance vs. Benchmark')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.grid(True)
        self._save_plot(fig, 'backtest_performance.png')

    def plot_correlation_matrix(self, returns):
        """Plot correlation matrix heatmap."""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        corr_matrix = returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, center=0)
        ax.set_title('Asset Correlation Matrix')
        self._save_plot(fig, 'correlation_matrix.png')

    def plot_returns_distribution(self, returns):
        """Plot returns distribution."""
        plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        sns.histplot(returns, kde=True)
        plt.title('Portfolio Returns Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.show() 