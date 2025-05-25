import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Optimize asset allocation using Sharpe Ratio maximization.
    Args:
        expected_returns (np.ndarray): Expected returns for each asset.
        cov_matrix (np.ndarray): Covariance matrix of returns.
        risk_free_rate (float): Risk-free rate of return.
    Returns:
        dict: Optimization results including weights, return, volatility, Sharpe ratio.
    """
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets

    # Objective: maximize Sharpe Ratio (minimize -Sharpe)
    def negative_sharpe(weights):
        port_return = np.sum(weights * expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -((port_return - risk_free_rate) / port_volatility)

    bounds = [(0, 1)] * num_assets  # Long-only portfolio
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]  # Sum of weights = 1

    result = minimize(negative_sharpe,
                      initial_weights,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    if not result.success:
        print("⚠️ Optimization failed:", result.message)
        weights = initial_weights
    else:
        weights = result.x

    port_return = np.sum(weights * expected_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility

    return {
        "Portfolio Weights": weights,
        "Portfolio Return": port_return,
        "Portfolio Volatility": port_volatility,
        "Sharpe Ratio": sharpe_ratio
    }
