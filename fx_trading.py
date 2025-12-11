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
    st.write("Données spot FX issues directement de l’API ECB (1 EUR = X CCY).")

    currencies = list(FLAGS.keys())
    rows = []

    for ccy in currencies:
        try:
            date, eur_ccy = get_eur_cross(ccy)
            ccy_eur = 1 / eur_ccy

            rows.append({
                "Devise": f"{FLAGS[ccy]} {ccy}",
                "EUR/CCY": eur_ccy,
                "CCY/EUR": ccy_eur,
                "Date": date
            })
        except Exception as e:
            rows.append({
                "Devise": f"{FLAGS[ccy]} {ccy}",
                "EUR/CCY": "Erreur",
                "CCY/EUR": "Erreur",
                "Date": "-"
            })

    df = pd.DataFrame(rows)

    st.subheader("📊 Cross FX avec drapeaux")
    st.dataframe(df, use_container_width=True)
