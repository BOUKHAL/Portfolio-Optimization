# Créer un nouveau fichier: scripts/ai/adaptive_optimization.py
import numpy as np
from scipy.optimize import minimize

def optimize_with_predictions(expected_returns, cov_matrix, ai_predictions, confidence, risk_free_rate):
    """
    Optimise le portefeuille en utilisant à la fois les rendements historiques et les prédictions AI
    
    Args:
        expected_returns: Rendements moyens historiques
        cov_matrix: Matrice de covariance
        ai_predictions: Prédictions IA des rendements futurs
        confidence: Niveau de confiance dans les prédictions (0-1)
        risk_free_rate: Taux sans risque
        
    Returns:
        dict: Portfolio optimisé
    """
    # Combiner rendements historiques et prédictions
    combined_returns = expected_returns * (1 - confidence) + ai_predictions * confidence
    
    # Nombre d'actifs
    n_assets = len(combined_returns)
    
    # Fonction objectif: maximiser le ratio de Sharpe
    def negative_sharpe_ratio(weights):
        portfolio_return = np.sum(weights * combined_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Contraintes: la somme des poids = 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bornes: chaque poids entre 0 et 1
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Poids initiaux égaux
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimisation
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
    
    # Extraire les résultats
    weights = result['x']
    portfolio_return = np.sum(weights * combined_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'Portfolio Weights': weights,
        'Portfolio Return': portfolio_return,
        'Portfolio Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratio
    }