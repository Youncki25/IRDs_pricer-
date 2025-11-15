import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import plotly.graph_objects as go
import QuantLib as ql
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

import analytic
from analytic import calcul_forward_xY, amort_table, mid_swap, add_months
import ui

from fredapi import Fred
import pandas_datareader.data as web  # <--- manquait

# ⚠️ Idéalement, mets ces clés dans st.secrets
FRED_API_KEY = "3c8e78b7fbab629ebde0669dc2f41f28"
BF_API_KEY = "fa0f05769356de38c744b31dde3a1736134911a86522dc5713b10a3c"

# Client FRED via fredapi
fred = Fred(api_key=FRED_API_KEY)

# Exemple : série 10y UST
data_dgs10 = fred.get_series("DGS10")


# Données macro
@st.cache_data(show_spinner=False)
def load_us_labor_market() -> pd.DataFrame:
    """
    Charge UNRATE, JTSJOL, PAYEMS sur 20 ans via FRED.
    """
    end = date.today()
    start = end - timedelta(days=365 * 20)

    series = ["UNRATE", "JTSJOL", "PAYEMS"]
    df = web.DataReader(series, "fred", start, end)

    df = (
        df.dropna()
        .reset_index()
        .rename(
            columns={
                "DATE": "Date",
                "UNRATE": "Taux de chômage (%)",
                "JTSJOL": "Job openings (milliers)",
                "PAYEMS": "Non-farm payrolls (milliers)",
            }
        )
    )
    return df


@st.cache_data(show_spinner=False)
def load_us_inflation(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Charge plusieurs indicateurs d’inflation US + CPI YoY.
    """
    end = date.today()

    ids = {
        "CPI - Headline": "CPIAUCSL",
        "CPI core": "CPILFESL",
        "PCE Headline": "PCEPI",
        "PCE Core": "PCEPILFE",
        "Déflateur PIB": "GDPDEF",
        "Median CPI": "MEDCPIM158SFRBCLE",
        "Trimmed-Mean CPI 16%": "TRMMEANCPIM158SFRBCLE",
        "Breakeven 10y": "T10YIE",
    }

    # Niveaux
    df_lvl = web.DataReader(list(ids.values()), "fred", start, end)
    df_lvl.columns = list(ids.keys())
    df_lvl = df_lvl.dropna().reset_index().rename(columns={"DATE": "Date"})

    # CPI YoY
    cpi = web.DataReader("CPIAUCSL", "fred", start, end).dropna()
    cpi = cpi.reset_index().rename(columns={"DATE": "Date"})
    cpi["CPI YoY (%)"] = cpi["CPIAUCSL"].pct_change(12) * 100

    yoy = cpi[["Date", "CPI YoY (%)"]]

    # Merge
    return df_lvl.merge(yoy, on="Date", how="left")


def fetch_fred_series(series_id: str, api_key: str = FRED_API_KEY) -> pd.DataFrame:
    """
