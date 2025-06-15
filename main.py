import os
import logging
import pandas as pd
import config
from backtester import Backtester
from portfolio_plotter import PortfolioPlotter

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging for the application."""
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'portfolio_analysis.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_data(file_path):
    """Load data from a pickle file."""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_pickle(file_path)

def main():
    """Main function to run the portfolio analysis pipeline."""
    try:
        # Initialize logging and load data
        setup_logging()
        logger.info("Starting portfolio analysis pipeline...")
        portfolio_data = load_data(config.CACHE_FILE)
        
        # Initialize and run the backtester
        backtester = Backtester(portfolio_data)
        results = backtester.run_backtest()
        
        # Print final results
        print("\n--- Backtest Results ---")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Annualized Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        
        # Generate plots
        logger.info("Generating visualizations...")
        plotter = PortfolioPlotter(portfolio_data, results)
        plotter.plot_portfolio_performance()
        plotter.plot_drawdown()
        plotter.plot_rolling_volatility()
        plotter.plot_weight_evolution()
        plotter.plot_correlation_heatmap()
        
        print("\nAnalysis complete! Check the 'plots' directory for visualizations.")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 