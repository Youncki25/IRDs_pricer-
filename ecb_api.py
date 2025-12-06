import requests
import pandas as pd

BASE_URL = "https://sdw-wsrest.ecb.europa.eu/service/data"
HEADERS = {"Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"}


def ecb_get_series(flow_ref: str, key: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    params: dict = {"detail": "dataonly"}
    if start:
        params["startPeriod"] = start
    if end:
        params["endPeriod"] = end

    url = f"{BASE_URL}/{flow_ref}/{key}"
    r = requests.get(url, params=params, headers=HEADERS)
    r.raise_for_status()
    data = r.json()

    series = data["data"]["dataSets"][0]["series"]
    series_key = list(series.keys())[0]
    observations = series[series_key]["observations"]

    time_dim = data["data"]["structure"]["dimensions"]["observation"][0]
    time_values = [v["id"] for v in time_dim["values"]]

    rows = []
    for obs_key, obs_val in observations.items():
        time_index = int(obs_key)
        time_period = time_values[time_index]
        value = obs_val[0]
        rows.append((time_period, float(value)))

    df = pd.DataFrame(rows, columns=["TIME_PERIOD", "OBS_VALUE"])
    return df
