import streamlit as st
import plotly.express as px

from ecb_api import ecb_get_series


def render():
    st.header("📊 Dashboard Macro — Données BCE")

    # Choix simple de série pour commencer
    serie = st.selectbox(
        "Choisis une série BCE :",
        options=[
            "Taux directeur principal (MRO)",
            "Inflation HICP YoY (Zone euro)",
        ],
    )

    if serie == "Taux directeur principal (MRO)":
        # Dataset : FM, série MRO
        flow = "FM"
        key = "D.U2.EUR.4F.KR.MRR_FR.LEV"
        start = "2015-01-01"
        label_y = "Taux (%)"
    else:
        # Dataset : ICP, inflation YoY HICP
        flow = "ICP"
        key = "M.U2.N.000000.4.ANR"
        start = "2015-01"
        label_y = "Inflation YoY (%)"

    # Récupération des données
    df = ecb_get_series(flow, key, start=start)

    st.subheader("Aperçu des données")
    st.dataframe(df.tail())

    st.subheader("Graphique")
    fig = px.line(
        df,
        x="TIME_PERIOD",
        y="OBS_VALUE",
        labels={"TIME_PERIOD": "Date", "OBS_VALUE": label_y},
        title=f"Série BCE — {serie}",
    )
    st.plotly_chart(fig, use_container_width=True)
