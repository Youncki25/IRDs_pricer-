import streamlit as st
import pandas as pd
from datetime import date, timedelta
import numpy as np
import pandas_datareader.data as web
import plotly.graph_objects as go

import Curves as curves


def render():
    tab = st.selectbox("Choisir", options=["Tableau d'amortissement", "Données macro", "Courbe des taux"], index=0)

    # --- Tableau d’amortissement ---
    if tab == "Tableau d'amortissement":
        if "df_amort" in st.session_state and isinstance(st.session_state["df_amort"], pd.DataFrame):
            st.dataframe(st.session_state["df_amort"], use_container_width=True)
            st.download_button(
                "📥 Télécharger le TA (CSV)",
                data=st.session_state["df_amort"].to_csv(index=False).encode("utf-8"),
                file_name="tableau_amortissement.csv",
                mime="text/csv"
            )
        else:
            st.warning("Aucun TA en mémoire. Va dans 'Pricer' → génère le TA puis reviens ici.")

    # --- Données macro US (exemple) ---
    elif tab == "Données macro":
        Pays = st.selectbox("Quel pays souhaitez-vous ?", options=["US", "France", "Germany", "England", "Europe"], index=0)

        if Pays == "US":
            type1 = st.multiselect("Quel sujet ?", options=["Croissance", "Inflation", "Marché du travail", "Autres"])

            if "Autres" in type1:
                st.subheader("Les données US 🇺🇸 : Courbe des taux (Trésor)")
                try:
                    today = date.today()
                    start = today - timedelta(days=10)
                    maturities = ['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS10','DGS20','DGS30']

                    # 1) Téléchargement FRED
                    df_yield = web.DataReader(maturities, 'fred', start, today)

                    # 2) Dernière date valide et vecteurs triés
                    last_valid = df_yield.dropna().index[-1]
                    curve = df_yield.loc[last_valid]
                    mat_labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y"]

                    # Tenors numériques (années)
                    tenors_years = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30], dtype=float)
                    yields_pct   = np.array([curve[k] for k in maturities], dtype=float)

                    # 3) Grille annuelle 1 → 30 ans
                    grid_years = np.arange(1, 31, dtype=float)

                    # 4) Choix des méthodes à afficher
                    methods = st.multiselect(
                        "Méthodes d'interpolation à tracer",
                        options=["Linéaire (taux %)", "Linéaire (zéro-continu)", "Log-linéaire (DF)", "PCHIP (log-DF)"],
                        default=["Log-linéaire (DF)", "Linéaire (zéro-continu)"]
                    )

                    # 5) Calcul des courbes interpolées
                    curves_dict = {}
                    if "Linéaire (taux %)" in methods:
                        curves_dict["Linéaire (taux %)"] = [curves.linear_interpolation(tenors_years, yields_pct, T) for T in grid_years]
                    if "Linéaire (zéro-continu)" in methods:
                        curves_dict["Linéaire (zéro-continu)"] = [curves.interp_zero_cont_linear(tenors_years, yields_pct, T) for T in grid_years]
                    if "Log-linéaire (DF)" in methods:
                        curves_dict["Log-linéaire (DF)"] = [curves.interp_yield_logDF(tenors_years, yields_pct, T) for T in grid_years]
                    if "PCHIP (log-DF)" in methods:
                        try:
                            curves_dict["PCHIP (log-DF)"] = [curves.interp_yield_pchip(tenors_years, yields_pct, T, on="logDF") for T in grid_years]
                        except Exception:
                            st.info("PCHIP indisponible (SciPy non installé).")

                    # 6) Graphe : points FRED + courbes
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(x=mat_labels, y=yields_pct, mode='markers',
                                                   name=f'FRED points — {last_valid.date()}', marker=dict(size=9)))
                    for name, yvals in curves_dict.items():
                        fig_curve.add_trace(go.Scatter(x=grid_years, y=yvals, mode='lines', name=name))
                    fig_curve.update_layout(
                        title="Courbe des taux US — points FRED + interpolations (1Y → 30Y)",
                        xaxis_title="Maturité (années)", yaxis_title="Taux (%)",
                        template="plotly_white", legend_title="Méthodes"
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur chargement FRED / interpolation : {e}")
