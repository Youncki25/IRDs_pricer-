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
    st.write("Données FX spot issues de l’API ECB (fixing quotidien 16h CET).")

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

    # --- Date de valeur / value date ---
    if last_update:
        st.markdown(
            f"📅 **Date de valeur (ECB Spot FX) : `{last_update}`**  \n"
            f"ℹ️ Données FX fixées à **16h CET**, publiées avec un **décalage d’environ 1 jour**."
        )

    st.subheader("📊 Taux spot ECB – Cross EUR/CCY")
    st.dataframe(df, use_container_width=True)
    st.markdown(
        """
        ---
        *Données fournies par la [Banque Centrale Européenne (ECB)](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html).*
        """
    )
    
