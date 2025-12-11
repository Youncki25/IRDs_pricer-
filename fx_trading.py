import streamlit as st
from change import get_eur_cross

import streamlit as st
from change import get_eur_cross
from emoji_country_flag import flag
import pandas as pd

FLAGS = {
    "USD": flag("US"),
    "GBP": flag("GB"),
    "JPY": flag("JP"),
    "CHF": flag("CH"),
    "AUD": flag("AU"),
    "CAD": flag("CA"),
    "SEK": flag("SE"),
    "NOK": flag("NO"),
    "DKK": flag("DK"),
    "PLN": flag("PL"),
    "CZK": flag("CZ"),
    "HUF": flag("HU"),
    "CNY": flag("CN"),
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
    st.markdown(
        """
        *Données récupérées via l’API publique de la Banque Centrale Européenne (ECB)*
        """
    )
