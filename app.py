# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("📊 Portfolio Optimization - Visualisation des Données")

# Charger les données
data_path = "data/portfolio_data.csv"
try:
    df = pd.read_csv(data_path)
    st.success("Données chargées avec succès !")
    st.dataframe(df)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {data_path}")

st.subheader("📅 Évolution des prix des actions")

selected_stocks = st.multiselect(
    "Choisissez les actions à afficher :", 
    options=df.columns[1:], 
    default=list(df.columns[1:3])  # Par défaut AAPL et MSFT
)

if selected_stocks:
    fig, ax = plt.subplots()
    for stock in selected_stocks:
        ax.plot(df['Date'], df[stock], label=stock)

    ax.set_xlabel("Date")
    ax.set_ylabel("Prix ($)")
    ax.set_title("Historique des prix")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Veuillez sélectionner au moins une action.")

st.subheader("📈 Rendements journaliers")

# Calcul des rendements
df_returns = df.copy()
df_returns.set_index('Date', inplace=True)
daily_returns = df_returns.pct_change().dropna()

# Choix des actions à afficher
selected_returns = st.multiselect(
    "Choisissez les actions pour afficher les rendements :",
    options=daily_returns.columns,
    default=daily_returns.columns[:2]
)

# Affichage du graphique
if selected_returns:
    fig2, ax2 = plt.subplots()
    for stock in selected_returns:
        ax2.plot(daily_returns.index, daily_returns[stock], label=stock)
    ax2.set_title("Rendements journaliers")
    ax2.set_ylabel("Rendement (%)")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("Veuillez sélectionner au moins une action.")

st.subheader("📉 Matrices de corrélation et de covariance")

# Matrice de corrélation
st.markdown("#### 🔗 Corrélation")
st.dataframe(daily_returns.corr().round(2))

# Matrice de covariance
st.markdown("#### 📐 Covariance")
st.dataframe(daily_returns.cov().round(4))

from scipy.optimize import minimize
import numpy as np

st.subheader("⚖️ Optimisation du portefeuille (minimisation du risque)")

# Rendements espérés annuels
mean_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252
num_assets = len(mean_returns)

# Fonction objectif : variance du portefeuille
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Contraintes et bornes
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = num_assets * [1. / num_assets]

# Optimisation
result = minimize(portfolio_variance, initial_weights, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Affichage
optimal_df = pd.DataFrame({
    'Actif': mean_returns.index,
    'Poids optimal (%)': np.round(optimal_weights * 100, 2)
})

st.markdown("### 📌 Répartition optimale (minimisation du risque)")
st.dataframe(optimal_df.set_index('Actif'))

st.subheader("💰 Optimisation du portefeuille (maximisation du Sharpe Ratio)")

risk_free_rate = st.number_input("Taux sans risque (par défaut 0%)", min_value=0.0, max_value=0.1, value=0.0, step=0.005)

# Fonction pour le Sharpe ratio négatif
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - rf) / portfolio_volatility
    return -sharpe_ratio

# Optimisation
sharpe_result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

sharpe_weights = sharpe_result.x
sharpe_df = pd.DataFrame({
    'Actif': mean_returns.index,
    'Poids optimal (%)': np.round(sharpe_weights * 100, 2)
})

st.markdown("### 📌 Répartition optimale (maximisation du Sharpe Ratio)")
st.dataframe(sharpe_df.set_index('Actif'))

st.subheader("📉 Frontière efficiente")

num_portfolios = st.slider("Nombre de portefeuilles simulés", 100, 5000, 1000, step=100)

results = {'Rendement': [], 'Volatilité': [], 'Sharpe': [], 'Poids': []}
np.random.seed(42)

for _ in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol

    results['Rendement'].append(port_return)
    results['Volatilité'].append(port_vol)
    results['Sharpe'].append(sharpe)
    results['Poids'].append(weights)

# Convertir en DataFrame
portfolios_df = pd.DataFrame(results)

# Tracer la frontière efficiente
fig3, ax3 = plt.subplots()
scatter = ax3.scatter(
    portfolios_df['Volatilité'], 
    portfolios_df['Rendement'], 
    c=portfolios_df['Sharpe'], 
    cmap='viridis', alpha=0.7
)
plt.colorbar(scatter, label='Sharpe Ratio')
ax3.set_xlabel("Volatilité")
ax3.set_ylabel("Rendement")
ax3.set_title("Frontière efficiente")
st.pyplot(fig3)




st.subheader("🎛️ Simulation personnalisée du portefeuille")

# Initialiser les sliders
custom_weights = []
st.markdown("#### 🔧 Ajustez les poids (en %) :")

for stock in mean_returns.index:
    w = st.slider(f"{stock}", 0.0, 100.0, 0.0, step=1.0)
    custom_weights.append(w)

# Normaliser les poids
weight_array = np.array(custom_weights) / 100
weight_array /= np.sum(weight_array) if np.sum(weight_array) != 0 else 1

# Calculs du portefeuille
custom_return = np.dot(weight_array, mean_returns)
custom_volatility = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))
custom_sharpe = (custom_return - risk_free_rate) / custom_volatility

# Affichage
st.markdown("### 📊 Résultats du portefeuille personnalisé")
st.write(f"**Rendement attendu :** {custom_return:.2%}")
st.write(f"**Volatilité attendue :** {custom_volatility:.2%}")
st.write(f"**Sharpe Ratio :** {custom_sharpe:.2f}")

