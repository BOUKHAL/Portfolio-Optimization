# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.optimization import optimize_portfolio

st.set_page_config(page_title="📈 Optimisation de Portefeuille", layout="wide")
st.title("📈 Optimisation de Portefeuille avec Intelligence Artificielle")

# --- Sidebar pour charger les données ---
st.sidebar.header("1️⃣ Données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.subheader("📊 Données Importées")
    st.dataframe(df.head())

    # --- Calcul des rendements log ---
    log_returns = np.log(df / df.shift(1)).dropna()

    st.subheader("📉 Rendements Logarithmiques")
    st.line_chart(log_returns)

    # --- Affichage des statistiques ---
    st.subheader("📈 Statistiques")
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values

    stats_df = pd.DataFrame({
        "Rendement Moyen": log_returns.mean(),
        "Écart-type": log_returns.std()
    })
    st.dataframe(stats_df.style.format("{:.4f}"))

    # --- Paramètres d'optimisation ---
    st.sidebar.header("2️⃣ Paramètres d'Optimisation")
    risk_free_rate = st.sidebar.slider("Taux sans risque", min_value=0.0, max_value=0.05, value=0.01, step=0.001)

    st.subheader("⚙️ Résultat de l'Optimisation")
    result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)

    weights = result["Portfolio Weights"]
    weights_df = pd.DataFrame(weights, index=df.columns, columns=["Poids"])
    st.bar_chart(weights_df)

    st.markdown(f"✅ **Rendement Attendu :** {result['Portfolio Return']:.4f}")
    st.markdown(f"📉 **Volatilité :** {result['Portfolio Volatility']:.4f}")
    st.markdown(f"📊 **Ratio de Sharpe :** {result['Sharpe Ratio']:.4f}")

    # --- Pie Chart ---
    fig, ax = plt.subplots()
    ax.pie(weights, labels=df.columns, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

else:
    st.info("Veuillez importer un fichier CSV contenant les prix d’actions.")
