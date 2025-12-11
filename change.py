import requests

BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR"

def get_eur_cross(quote_ccy: str, freq: str = "D"):
    """
    Récupère 1 EUR = ? quote_ccy
    (ex: quote_ccy='USD' -> 1 EUR = x USD)

    freq: 'D' = daily, 'M' = monthly, etc.
    """
    quote_ccy = quote_ccy.upper()

    # Série ECB : FREQ.CCY.EUR.SP00.A
    key = f"{freq}.{quote_ccy}.EUR.SP00.A"
    url = f"{BASE_URL}/{key}"

    params = {
        "detail": "dataonly",
        "lastNObservations": 1,
        "format": "jsondata",
    }

    resp = requests.get(url, params=params)
    print(f"[{quote_ccy}] Status code:", resp.status_code)
    print(f"[{quote_ccy}] URL appelée:", resp.url)
    resp.raise_for_status()

    data = resp.json()

    # --- Parsing SDMX-JSON (même logique qu’avant) ---
    series = data["dataSets"][0]["series"]["0:0:0:0:0"]
    observations = series["observations"]

    last_date = sorted(observations.keys())[-1]
    eur_quote = observations[last_date][0]  # 1 EUR = eur_quote (CCY)

    return last_date, eur_quote


if __name__ == "__main__":

    # Liste des devises qu’on veut
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
                "date": date,
                "EUR/CCY": eur_ccy,
                "CCY/EUR": ccy_eur,
            }

            print("-" * 40)
            print(f"Date       : {date}")
            print(f"EUR/{ccy}  : {eur_ccy}")
            print(f"{ccy}/EUR  : {ccy_eur}")

        except Exception as e:
            print(f"⚠️ Erreur pour {ccy} : {e}")

    print("\n=== Récap (dictionnaire utilisable dans ton pricer) ===")
    print(results)
