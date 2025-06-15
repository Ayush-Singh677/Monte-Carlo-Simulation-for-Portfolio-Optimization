import numpy as np
from config import SIMULATION_NUMBER, TIME_PERIOD, INITIAL_PORTFOLIO_VALUE

class MonteCarloSimulator:
    def __init__(self, mean_returns, cov_matrix, weights):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.weights = weights
        self.portfolio_sims = None

    def run_simulation(self):
        """Run Monte Carlo simulation for portfolio value projection."""
        mean_matrix = np.full(shape=(TIME_PERIOD, len(self.weights)), 
                            fill_value=self.mean_returns)
        mean_matrix = mean_matrix.T
        
        self.portfolio_sims = np.full(shape=(TIME_PERIOD, SIMULATION_NUMBER), 
                                    fill_value=0.0)

        for m in range(SIMULATION_NUMBER):
            Z = np.random.normal(size=(TIME_PERIOD, len(self.weights)))
            L = np.linalg.cholesky(self.cov_matrix)
            daily_returns = mean_matrix + np.inner(L, Z)
            self.portfolio_sims[:, m] = np.cumprod(
                np.inner(self.weights, daily_returns.T) + 1
            ) * INITIAL_PORTFOLIO_VALUE

        return self.portfolio_sims

    def get_statistics(self):
        """Calculate key statistics from the simulation results."""
        if self.portfolio_sims is None:
            self.run_simulation()

        final_values = self.portfolio_sims[-1, :]
        return {
            'mean': np.mean(final_values),
            'std': np.std(final_values),
            'min': np.min(final_values),
            'max': np.max(final_values),
            'median': np.median(final_values),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1)
        } 