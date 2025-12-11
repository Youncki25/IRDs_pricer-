import streamlit as st
import pandas as pd
from change import get_eur_cross

# Mapping devise -> emoji du drapeau (Unicode)
FLAGS = {
    "USD": "🇺🇸",
    "GBP": "🇬🇧",
    "JPY": "🇯🇵",
    "CHF": "🇨🇭",
    "AUD": "🇦🇺",
    "CAD": "🇨🇦",
    "SEK": "🇸🇪",
    "NOK": "🇳🇴",
    "DKK": "🇩🇰",
    "PLN": "🇵🇱",
    "CZK": "🇨🇿",
    "HUF": "🇭🇺",
    "CNY": "🇨🇳",
}


def render():
    st.title("💱 FX Trading – Data ECB")

    currencies = list(FLAGS.keys())
    rows = []
    last_update = None

    for ccy in currencies:
        try:
            date, eur_ccy = get_eur_cross(ccy)
            ccy_eur = 1 / eur_ccy
            last_update = date

            rows.append({
                "Cross": f"EUR/{ccy}",
                "EUR→CCY": eur_ccy,
                "CCY→EUR": ccy_eur,
            })

        except Exception:
            rows.append({
                "Cross": f"EUR/{ccy}",
                "EUR→CCY": "Erreur",
                "CCY→EUR": "Erreur",
            })

    df = pd.DataFrame(rows)

    # --- Date de calcul (value date) ---
    if last_update:
        st.markdown(f"📅 **Date de calcul des données : `{last_update}`**")

    st.subheader("📊 Taux spot ECB – EUR/CCY")
    st.dataframe(df, use_container_width=True)
