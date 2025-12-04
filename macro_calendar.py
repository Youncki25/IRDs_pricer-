# fichier calendrier économiques : 
import requests
import pandas as pd
from datetime import datetime, timedelta

TRADING_ECONOMICS_API_KEY = "TON_API_KEY"  # à mettre dans un secret streamlit si possible

def get_this_week_calendar(country: str | None = None) -> pd.DataFrame:
    """
    Récupère les événements macro de la semaine (lundi-dimanche).
    Optionnellement filtré par pays.
    """
    today = datetime.utcnow().date()
    # début = lundi de cette semaine
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    base_url = "https://api.tradingeconomics.com/calendar"

    params = {
        "c": TRADING_ECONOMICS_API_KEY,
        "d1": start_str,
        "d2": end_str,
        "format": "json"
    }

    if country:
        params["country"] = country

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Normalisation des colonnes utiles
    # (les noms exacts peuvent varier, à adapter avec la doc)
    keep_cols = [
        "Date",            # date/heure de la publi
        "Country",
        "Category",        # type de stat (GDP, CPI...)
        "Event",           # nom de la stat
        "Importance",      # impact (Low/Medium/High)
        "Actual",
        "Previous",
        "Forecast"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Convertir en datetime, trier
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    return df
