from datetime import datetime
import os

# --- General Configuration ---
# Create a directory for generated files if it doesn't exist
if not os.path.exists('generated_output'):
    os.makedirs('generated_output')

# --- Portfolio Configuration ---
PORTFOLIO = ['AMZN', 'GOOG', 'MSFT', 'NFLX', 'AAPL', 'NVDA']
BENCHMARK_TICKER = 'SPY'  # Benchmark for backtesting (e.g., S&P 500)

# --- Date Configuration ---
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime.now()
# Date to split data for training (optimization) and testing (backtesting)
# If None, no backtesting will be performed.
BACKTEST_SPLIT_DATE = datetime(2022, 1, 1)

# --- Data Caching Configuration ---
DATA_CACHE_PATH = os.path.join('generated_output', 'portfolio_data.pkl')
CACHE_EXPIRATION_DAYS = 1  # How many days before refreshing the cache

# --- Simulation Configuration ---
SIMULATION_NUMBER = 20000
TIME_PERIOD = 252  # One year of trading days
INITIAL_PORTFOLIO_VALUE = 25000

# --- Optimization Configuration ---
TOTAL_PORTFOLIO_VALUE_FOR_ALLOCATION = 10000
RISK_FREE_RATE = 0.02

# --- Visualization Configuration ---
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (15, 8)
DPI = 300
EFFICIENT_FRONTIER_SAMPLES = 100 # Number of portfolios to plot on the frontier
# Path to save visualizations
PLOTS_PATH = os.path.join('generated_output', 'plots')
if not os.path.exists(PLOTS_PATH):
    os.makedirs(PLOTS_PATH)

# --- Reporting Configuration ---
REPORT_PATH = os.path.join('generated_output', 'summary_report.txt')

# --- Logging Configuration ---
LOG_FILE = os.path.join('generated_output', 'portfolio_analysis.log')
LOG_LEVEL = "INFO"  # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL

# Data fetching configuration
TICKERS = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'NFLX', 'NVDA', 'SPY']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'
CACHE_FILE = 'generated_output/portfolio_data.pkl'

# Portfolio optimization configuration
MAX_WEIGHT = 0.5
MIN_WEIGHT = 0.0
ALLOW_SHORTING = True
SOLVER = 'SCS'
SOLVER_OPTIONS = {
    'max_iters': 10000,
    'eps': 1e-8,
    'verbose': True
}

# Monte Carlo simulation configuration
NUM_SIMULATIONS = 1000
TIME_HORIZON = 252  # 1 year of trading days
CONFIDENCE_LEVEL = 0.95

# Backtesting configuration
INITIAL_CAPITAL = 100000.0  # Initial capital for the backtest
BACKTEST_START_DATE = '2013-01-01'  # Start date for the backtesting period
ROLLING_WINDOW_YEARS = 3  # Number of years of historical data for each optimization
REBALANCE_FREQUENCY = 'Q'  # 'Q' for quarterly, 'M' for monthly

# Statistical analysis configuration
ROLLING_WINDOW = 252  # One year rolling window for metrics
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]  # Value at Risk confidence levels
CVAR_CONFIDENCE_LEVELS = [0.95, 0.99]  # Conditional Value at Risk confidence levels

TRADING_DAYS_PER_YEAR = 252  # Typical number of trading days in a year
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

BACKTEST_START_DATE = '2022-01-01'  # Start date for backtesting 