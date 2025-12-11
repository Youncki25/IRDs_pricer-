import streamlit as st
import fx_trading
import ui as U

from config import APP_TITLE, APP_LAYOUT

import accueil
import pricer
import graphique
import obligations
import financement
import dashboard_macro   


#streamlit run c:/Users/youne/https-github.com-Youncki25-credit-rating-app/IRDs_pricer-/main.py
# --------------------- App config & sidebar ---------------------
st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)
st.markdown(U.CSS, unsafe_allow_html=True)

# Menu latéral (déjà défini dans ui.py)
page = U.sidebar()

# Bandeau SOFR en haut (comme avant)
U.show_sofr_banner("SOFR")

if page == "Accueil":
    accueil.render()
elif page == "FX Trading":
    fx_trading.render()   
elif page == "Dashboard Macroéconomique":
    dashboard_macro.render()   
elif page == "Pricer":
    pricer.render()
elif page == "Graphique":
    graphique.render()
elif page == "Obligations":
    obligations.render()
elif page == "Tableau amortissement pour financement":
    financement.render()
