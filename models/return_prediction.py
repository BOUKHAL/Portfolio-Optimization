import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_features(data, window_sizes=[5, 10, 20, 50]):
    """Extraire des caractéristiques techniques pour la prédiction"""
    features = pd.DataFrame(index=data.index)
    
    # Identifier les colonnes de prix (non-Returns)
    price_columns = [col for col in data.columns if col in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]]
    
    # Identifier les colonnes de rendements (contenant "Returns")
    returns_columns = [col for col in data.columns if "Returns" in col]
    
    # Afficher les colonnes pour le débogage
    print("Colonnes de prix:", price_columns)
    print("Colonnes de rendements:", returns_columns)
    
    # Pour chaque colonne de prix
    for price_col in price_columns:
        # Créer des caractéristiques basées sur les prix
        for window in window_sizes:
            features[f'ma_{window}_{price_col}'] = data[price_col].rolling(window).mean()
    
    # Pour chaque colonne de rendements
    for returns_col in returns_columns:
        # Extraire le symbole du nom de la colonne (par exemple AAPL de "Returns_Log Returns (AAPL)")
        for symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]:
            if symbol in returns_col:
                # Créer des caractéristiques basées sur les rendements
                for window in window_sizes:
                    features[f'vol_{window}_{symbol}'] = data[returns_col].rolling(window).std()
                break
    
    return features.dropna()

def train_prediction_models(data, prediction_horizon=30):
    """Entraîner des modèles de prédiction pour chaque actif"""
    # Afficher les colonnes pour le débogage
    print("Colonnes disponibles:")
    print(data.columns.tolist())
    
    # Extraire les caractéristiques pour l'entraînement
    features = extract_features(data)
    
    if features.empty:
        print("Aucune caractéristique extraite - vérifiez les noms de colonnes")
        return {}, None
    
    # Mapper les symboles aux colonnes de rendements
    returns_map = {}
    for col in data.columns:
        if "Returns" in col:
            for symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]:
                if symbol in col:
                    returns_map[symbol] = col
                    break
    
    print("Mapping symboles -> colonnes de rendements:", returns_map)
    
    models = {}
    
    for symbol, returns_col in returns_map.items():
        print(f"Traitement du symbole {symbol} avec colonne de rendements {returns_col}")
        
        # Créer la cible: rendement futur
        target = data[returns_col].rolling(prediction_horizon).mean().shift(-prediction_horizon)
        
        # Supprimer les lignes avec NaN
        valid_idx = ~(target.isna())
        
        # Vérifier si features a suffisamment de données
        if len(features) < len(data):
            valid_features_idx = features.index.intersection(data.index[valid_idx])
            X = features.loc[valid_features_idx]
            y = target.loc[valid_features_idx]
        else:
            X = features[valid_idx]
            y = target[valid_idx]
        
        if len(y) < 20:  # Vérifier qu'il y a assez de données
            print(f"Pas assez de données pour {symbol}, on l'ignore")
            continue
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement du modèle
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Évaluation
        score = model.score(X_test_scaled, y_test)
        print(f"Score R² pour {symbol}: {score:.4f}")
        
        models[symbol] = {
            'model': model,
            'scaler': scaler,
            'score': score
        }
        # ✅ Sauvegarder les noms des colonnes pour la prédiction
        models[symbol]['feature_names'] = X.columns.tolist()
    
    return models, features