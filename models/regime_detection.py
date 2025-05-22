from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def detect_market_regimes(data, n_regimes=3):
    """Détecter différents régimes de marché avec le clustering K-means"""
    
    # Créer des caractéristiques pour la détection de régime
    features = pd.DataFrame(index=data.index)
    
    # Identifier les colonnes de rendements
    returns_columns = [col for col in data.columns if "Returns" in col]
    print("Colonnes de rendements pour détection de régimes:", returns_columns)
    
    if not returns_columns:
        print("Aucune colonne de rendements trouvée pour la détection de régimes")
        return pd.DataFrame(), None, {}
    
    # Calculer statistiques glissantes pour tous les actifs
    for col in returns_columns:
        # Identifier le symbole dans le nom de colonne
        symbol = None
        for s in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
            if s in col:
                symbol = s
                break
                
        if symbol:
            features[f'vol_{symbol}'] = data[col].rolling(21).std()  # Volatilité sur 1 mois
            features[f'trend_{symbol}'] = data[col].rolling(63).mean()  # Tendance sur 3 mois
    
    # Supprimer les lignes avec NaN
    features = features.dropna()
    
    # S'assurer qu'il y a des données
    if features.empty or features.shape[0] < 10:
        print("Pas assez de données pour la détection de régimes")
        return pd.DataFrame(), None, {}
    
    # Normalisation pour éviter que des échelles différentes n'affectent le clustering
    features_scaled = (features - features.mean()) / features.std()
    
    # Appliquer le clustering K-means
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    regimes = kmeans.fit_predict(features_scaled)
    
    # Créer un DataFrame avec dates et régimes
    regime_df = pd.DataFrame({
        'Date': features.index,
        'Regime': regimes
    })
    
    # Analyser les caractéristiques de chaque régime
    regime_analysis = {}
    for i in range(n_regimes):
        mask = (regimes == i)
        regime_features = features_scaled[mask]
        
        regime_analysis[i] = {
            'count': mask.sum(),
            'avg_features': regime_features.mean().to_dict()
        }
    
    return regime_df, kmeans, regime_analysis