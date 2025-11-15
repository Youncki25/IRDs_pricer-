import streamlit as st
import pandas_datareader.data as web
from datetime import date, timedelta

CSS = """
<style>
.stApp {
  background-image: linear-gradient(rgba(0,0,0,.5), rgba(0,0,0,.5)),
                    url('https://media.istockphoto.com/id/1487894858/fr/photo/graphique-en-chandelier-et-données-du-marché-financier.jpg?s=612x612&w=0&k=20&c=tJoRghcmr2l10qJflJUkmY1kGjUqjccYGxiBSxRiQFc=');
  background-size: cover;
  background-attachment: fixed;
  color: #fff;
}
/* … le reste de votre CSS existant … */
</style>
"""

def sidebar() -> str:
    return st.sidebar.selectbox(
        "Choisissez une rubrique",
        ("Accueil", "Pricer", "Graphique", "Risk Management", "Strategie de trading")
    )


def swap_inputs():
    col1, col2 = st.columns(2)
    with col1:
        d0 = st.date_input("🗓 Date de début", min_value=date.today(), value=date(2026, 1, 1))
    with col2:
        d1 = st.date_input("🗓 Date de fin", min_value=d0 + timedelta(days=1), value=date(2030, 1, 1))
    col3, col4 = st.columns(2)
    with col3:
        p_var = st.selectbox("Périodicité variable (mois)", [1, 3, 6, 12])
        p_fix = st.selectbox("Périodicité fixe (mois)", [1, 3, 6, 12])
    with col4:
        amort = st.selectbox("Amortissement", ["Linéaire", "In Fine"])
        fixing = st.selectbox("Fixing", [f"J{n}" if n < 0 else "J" if n == 0 else f"J+{n}" for n in range(-1, 11)])
    notionnel = st.number_input("Notionnel €", 0, 1_000_000_000, 1_000_000)
    indice = st.selectbox("Indice variable", ["Euribor", "Ester", "Libor", "SOFR", "SONIA"])
    return d0, d1, p_var, p_fix, amort, notionnel, indice, fixing

def show_sofr_banner(series_id: str = "SOFR"):
    try:
        df = web.DataReader(series_id, "fred")
        last_val = float(df.iloc[-1, 0])
        last_dt = df.index[-1].date()
        line = f"Dernier {series_id} publié — {last_dt}: {last_val:.2f}%"
    except Exception as e:
        st.warning(f"{series_id} non disponible : {e}")
        line = f"Dernier {series_id} publié — N/A"
    st.markdown(f'<p class="sofr-text">{line}</p>', unsafe_allow_html=True)
