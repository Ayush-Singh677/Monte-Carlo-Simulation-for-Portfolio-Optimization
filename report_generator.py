import pandas as pd
from datetime import datetime
from config import REPORT_PATH
from logger import logger
from stats_analyzer import PortfolioStatsAnalyzer

def format_dict_for_report(d, title):
    """Formats a dictionary into a string for the report."""
    report_str = f"--- {title} ---\n"
    for key, value in d.items():
        if isinstance(value, float):
            if abs(value) > 100:
                report_str += f"{key:<30}: ${value:,.2f}\n"
            elif (
                "return" in key.lower() or
                "ratio" in key.lower() or
                "volatility" in key.lower()
            ):
                 report_str += f"{key:<30}: {value:.2%}\n" if "return" in key.lower() or "volatility" in key.lower() else f"{key:<30}: {value:.2f}\n"
            else:
                report_str += f"{key:<30}: {value:.4f}\n"
        else:
            report_str += f"{key:<30}: {value}\n"
    return report_str + "\n"


class ReportGenerator:
    def __init__(self, portfolio_data, optimizer, backtest_results, stats_analyzer):
        self.portfolio_data = portfolio_data
        self.optimizer = optimizer
        self.backtest_results = backtest_results
        self.stats_analyzer = stats_analyzer
        self.report_path = REPORT_PATH

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info(f"Generating summary report to {self.report_path}...")
        
        with open(self.report_path, 'w') as f:
            f.write("PORTFOLIO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic Portfolio Information
            f.write("1. PORTFOLIO COMPOSITION\n")
            f.write("-" * 30 + "\n")
            weights = self.optimizer.get_weights()
            for asset, weight in weights.items():
                f.write(f"{asset}: {weight:.2%}\n")
            f.write("\n")
            
            # Performance Metrics
            f.write("2. PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            ret, vol, sharpe = self.optimizer.get_portfolio_performance()
            f.write(f"Expected Annual Return: {ret:.2%}\n")
            f.write(f"Annual Volatility: {vol:.2%}\n")
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write("\n")
            
            # Advanced Statistical Analysis
            f.write("3. ADVANCED STATISTICAL ANALYSIS\n")
            f.write("-" * 30 + "\n")
            metrics = self.stats_analyzer.get_all_metrics(pd.Series(weights))
            
            # Basic Metrics
            f.write("Basic Metrics:\n")
            f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
            f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n")
            f.write("\n")
            
            # Risk Metrics
            f.write("Risk Metrics:\n")
            f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n")
            f.write(f"Value at Risk (95%): {metrics['var_95']:.2%}\n")
            f.write(f"Value at Risk (99%): {metrics['var_99']:.2%}\n")
            f.write(f"Expected Shortfall (95%): {metrics['cvar_95']:.2%}\n")
            f.write(f"Expected Shortfall (99%): {metrics['cvar_99']:.2%}\n")
            f.write("\n")
            
            # Distribution Metrics
            f.write("Return Distribution Metrics:\n")
            f.write(f"Skewness: {metrics['skewness']:.2f}\n")
            f.write(f"Kurtosis: {metrics['kurtosis']:.2f}\n")
            f.write("\n")
            
            # Factor Analysis
            if 'beta' in metrics:
                f.write("Factor Analysis:\n")
                f.write(f"Beta: {metrics['beta']:.2f}\n")
                f.write(f"Alpha: {metrics['alpha']:.2%}\n")
                f.write(f"R-squared: {metrics['r_squared']:.2f}\n")
                f.write("\n")
            
            # Risk Decomposition
            f.write("Risk Decomposition:\n")
            f.write(f"Portfolio Variance: {metrics['portfolio_variance']:.4f}\n")
            f.write(f"Portfolio Volatility: {metrics['portfolio_volatility']:.2%}\n")
            f.write("\nMarginal Contribution to Risk:\n")
            for asset, mctr in metrics['marginal_contribution_to_risk'].items():
                f.write(f"{asset}: {mctr:.2%}\n")
            f.write("\nComponent Contribution to Risk:\n")
            for asset, cctr in metrics['component_contribution_to_risk'].items():
                f.write(f"{asset}: {cctr:.2%}\n")
            f.write("\n")
            
            # Backtest Results
            f.write("4. BACKTEST RESULTS\n")
            f.write("-" * 30 + "\n")
            portfolio_value = self.backtest_results['Portfolio'].iloc[-1]
            benchmark_value = self.backtest_results['Benchmark'].iloc[-1]
            f.write(f"Final Portfolio Value: ${portfolio_value:,.2f}\n")
            f.write(f"Final Benchmark Value: ${benchmark_value:,.2f}\n")
            f.write(f"Outperformance: ${(portfolio_value - benchmark_value):,.2f}\n")
            f.write(f"Outperformance %: {((portfolio_value/benchmark_value - 1) * 100):.2f}%\n")
            f.write("\n")
            
            # Turnover Analysis
            if 'avg_turnover' in metrics:
                f.write("5. TURNOVER ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Portfolio Turnover: {metrics['avg_turnover']:.2%}\n")
                f.write(f"Maximum Portfolio Turnover: {metrics['max_turnover']:.2%}\n")
                f.write("\nAsset-Specific Turnover:\n")
                for asset, turnover in metrics['asset_turnover'].items():
                    f.write(f"{asset}: {turnover:.2%}\n")
                f.write("\n")
            
            # Report Generation Info
            f.write("\nReport generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        logger.info("Summary report generated successfully.") 