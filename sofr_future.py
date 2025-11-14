import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta

MONTH_CODES = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}

def third_wednesday(y, m):
    d = date(y, m, 15)
    while d.weekday() != 2:
        d += timedelta(days=1)
    return d

def sr3_symbols(start_year, end_year):
    syms, meta = [], []
    months = [3, 6, 9, 12]
    for y in range(start_year, end_year + 1):
        yy = str(y)[-2:]
        for code, m in MONTH_CODES.items():
            sym = f"SR3{code}{yy}.CME"
            start = third_wednesday(y, m)
            idx = months.index(m)
            m_end = months[(idx + 1) % 4]
            y_end = y + (1 if m == 12 else 0)
            end = third_wednesday(y_end, m_end)
            syms.append(sym)
            meta.append({"symbol": sym, "start_date": start, "end_date": end})
    return syms, pd.DataFrame(meta)

def download_last_close(symbols, period="6mo", interval="1d", auto_adjust=False):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=auto_adjust, progress=False, group_by="ticker", threads=True)
    closes = {}
    for s in symbols:
        try:
            ser = raw[s]["Close"].dropna()
            if not ser.empty:
                closes[s] = ser
        except Exception:
            continue
    latest_rows = []
    for s, ser in closes.items():
        latest_rows.append({
            "symbol": s,
            "last_trade_date": ser.index[-1].date(),
            "last_close": float(ser.iloc[-1])
        })
    snapshot = pd.DataFrame(latest_rows)
    panel = pd.DataFrame(closes)
    return snapshot, panel

def add_implied_rates(df, price_col="last_close"):
    out = df.copy()
    out["fut_rate"] = (100.0 - out[price_col]) / 100.0
    return out

def apply_convexity_correction(df, sigma=0.0):
    if sigma == 0.0:
        df = df.copy()
        df["fwd_rate"] = df["fut_rate"]
        return df
    out = df.copy()
    def yf_yearfrac(d1, d2):
        days = (pd.to_datetime(d2) - pd.to_datetime(d1)).days
        return days / 360.0
    t0 = pd.Timestamp(date.today())
    out["T_start"] = out["start_date"].apply(lambda d: yf_yearfrac(t0, d))
    out["T_end"]   = out["end_date"].apply(lambda d: yf_yearfrac(t0, d))
    out["convexity"] = 0.5 * (sigma**2) * out["T_start"] * out["T_end"]
    out["fwd_rate"] = out["fut_rate"] - out["convexity"]
    return out

def add_tau(df):
    out = df.copy()
    days = (pd.to_datetime(out["end_date"]) - pd.to_datetime(out["start_date"])).dt.days
    out["tau_SE"] = days / 360.0
    return out

def bootstrap_dfs_from_futures(df, r0_overnight=0.052):
    out = df.copy().sort_values(["start_date", "end_date"]).reset_index(drop=True)
    t0 = pd.Timestamp(date.today())
    out = out[pd.to_datetime(out["end_date"]) > t0].reset_index(drop=True)
    if out.empty:
        return out.assign(DF_start=np.nan, DF_end=np.nan, T_0E=np.nan,
                          zero_simple=np.nan, zero_cont=np.nan)

    def yf_yearfrac(d1, d2):
        return (pd.to_datetime(d2) - pd.to_datetime(d1)).days / 360.0

    n = len(out)
    DF_S = [np.nan] * n
    DF_E = [np.nan] * n

    S1 = pd.to_datetime(out.loc[0, "start_date"])
    E1 = pd.to_datetime(out.loc[0, "end_date"])
    r1 = out.loc[0, "fwd_rate"]

    if t0 < S1:
        tau_0S1 = yf_yearfrac(t0, S1)
        DF_S[0] = 1.0 / (1.0 + r0_overnight * tau_0S1)
        DF_E[0] = DF_S[0] / (1.0 + r1 * out.loc[0, "tau_SE"])
    else:
        tau_0E1 = yf_yearfrac(t0, E1)
        DF_S[0] = 1.0
        DF_E[0] = 1.0 / (1.0 + r1 * tau_0E1)

    for i in range(1, n):
        DF_S[i] = DF_E[i - 1]
        DF_E[i] = DF_S[i] / (1.0 + out.loc[i, "fwd_rate"] * out.loc[i, "tau_SE"])

    out["DF_start"] = DF_S
    out["DF_end"] = DF_E

    out["T_0E"] = out["end_date"].apply(lambda d: yf_yearfrac(t0, d))
    out.loc[out["T_0E"] <= 0, "T_0E"] = np.nan

    out["zero_simple"] = (1.0 / out["DF_end"] - 1.0) / out["T_0E"]
    out["zero_cont"]   = -np.log(out["DF_end"]) / out["T_0E"]
    return out

def build_sofr_curve_from_sr3(df_raw, r0_overnight=0.052, convexity_sigma=0.0):
    df1 = add_implied_rates(df_raw, price_col="last_close")
    df2 = apply_convexity_correction(df1, sigma=convexity_sigma)
    df3 = add_tau(df2)
    curve = bootstrap_dfs_from_futures(df3, r0_overnight=r0_overnight)
    keep = ["symbol","start_date","end_date","last_close","fut_rate","fwd_rate",
            "tau_SE","DF_start","DF_end","T_0E","zero_simple","zero_cont"]
    return curve[keep].copy()
