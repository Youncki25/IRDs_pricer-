import requests

BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR"


def get_eur_cross(ccy: str, freq: str = "D"):
    """
    Retourne :
      - la date de calcul (value date)
      - le taux spot EUR/CCY

    Exemple : get_eur_cross("USD") -> ("2025-12-10", 1.1714)
    """

    ccy = ccy.upper()

    # Clé SDMX : FREQ.CCY.EUR.SP00.A
    key = f"{freq}.{ccy}.EUR.SP00.A"
    url = f"{BASE_URL}/{key}"

    params = {
        "detail": "dataonly",
        "lastNObservations": 1,
        "format": "jsondata",
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()

    # ---- Parsing SDMX JSON ----
    series = data["dataSets"][0]["series"]["0:0:0:0:0"]
    observations = series["observations"]

    # On récupère l’index de la dernière observation (ex: "0")
    idx_str = sorted(observations.keys())[-1]
    idx = int(idx_str)

    # La vraie date est dans la dimension d'observation
    date_values = data["structure"]["dimensions"]["observation"][0]["values"]
    date_str = date_values[idx]["id"]     # ex: "2025-12-10"

    # Valeur du spot EUR/CCY
    eur_ccy = observations[idx_str][0]

    return date_str, eur_ccy
