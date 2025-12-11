import streamlit as st
from change import get_eur_cross

def render():
    st.title("💱 FX Trading – Data ECB")

    st.write("Données spot FX issues directement de l'API ECB (1 EUR = X CCY).")

    currencies = [
        "USD", "GBP", "JPY", "CHF",
        "AUD", "CAD",
        "SEK", "NOK", "DKK",
        "PLN", "CZK", "HUF",
        "CNY",
    ]

    results = {}

    for ccy in currencies:
        try:
            date, eur_ccy = get_eur_cross(ccy)
            ccy_eur = 1 / eur_ccy

            results[ccy] = {
                "EUR/CCY": eur_ccy,
                "CCY/EUR": ccy_eur,
                "date": date
            }

        except Exception as e:
            results[ccy] = {"error": str(e)}

    st.subheader("📊 Cross FX")
    st.write(results)

    st.success("FX data loaded successfully from ECB API.")
