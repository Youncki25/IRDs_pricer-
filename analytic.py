from datetime import datetime, date
import pandas as pd

def calcul_discount_factor_RFR(OIS, base, calendrier):
    df = pd.DataFrame({"Maturité": calendrier, "Taux": OIS})
    df["Discount Factor"] = (1 / (1 + df["Taux"] / base)).cumprod()
    return df

def calcul_forward_xY(df, x):
    maturities = df["Maturité"].tolist()
    dfs = df["Discount Factor"].tolist()
    fwd = []
    try:
        idx_start = maturities.index(float(x))
    except ValueError:
        raise ValueError(f"La maturité {x} ans n'existe pas dans la courbe.")
    DF_t1 = dfs[idx_start]
    t1 = maturities[idx_start]
    for i in range(len(df)):
        if i < idx_start or i + x >= len(df):
            fwd.append("Pas de fwd")
        else:
            t2 = maturities[i + x]
            DF_t2 = dfs[i + x]
            taux = (DF_t1 / DF_t2) ** (1 / (t2 - t1)) - 1
            fwd.append(round(taux * 100, 4))
    df[f"Forward {x}Y"] = fwd
    return df

def add_months(start: datetime, months: int) -> datetime:
    m = start.month - 1 + months
    y, m = start.year + m // 12, m % 12 + 1
    d = start.day
    while True:
        try:
            return datetime(y, m, d)
        except ValueError:
            d -= 1

def amort_table(d0: date | datetime, d1: date | datetime,
                p_var: int, p_fix: int, amort: str, notionnel: float,
                indice: str, fixing: str = "J-1") -> pd.DataFrame:
    if isinstance(d0, date):
        d0 = datetime.combine(d0, datetime.min.time())
    if isinstance(d1, date):
        d1 = datetime.combine(d1, datetime.min.time())

    p_months = min(p_var, p_fix)
    dates_d, dates_f, caps, idx, fix = [], [], [], [], []
    n_periode, temp = 0, d0

    while temp < d1:
        temp = add_months(temp, p_months)
        if temp > d1:
            temp = d1
        n_periode += 1

    remb = notionnel / n_periode if amort.lower() == "linéaire" else 0
    cap_rest = notionnel
    cur = d0
    while cur < d1:
        nxt = add_months(cur, p_months)
        if nxt > d1:
            nxt = d1
        dates_d.append(cur); dates_f.append(nxt)
        caps.append(cap_rest); idx.append(f"{indice}-{p_var}M"); fix.append(fixing)
        if amort.lower() == "linéaire":
            cap_rest -= remb
        cur = nxt

    return pd.DataFrame({
        "Date début": [d.strftime("%Y-%m-%d") for d in dates_d],
        "Date fin":   [d.strftime("%Y-%m-%d") for d in dates_f],
        "Capital restant (€)": caps,
        "Indice": idx,
        "Fixing": fix
    })

def mid_swap(maturities, dfs, alpha):
    res = [None]
    for n in range(1, len(maturities)):
        dfs_sel = dfs[:n + 1]
        rate = (1 - dfs_sel[-1]) / sum(alpha * d for d in dfs_sel)
        res.append(rate * 100)
    return res

def npv(rate: float, cashflows: list[float]) -> float:
    total = 0.0
    for i, cf in enumerate(cashflows):
        total += cf / (1 + rate) ** i
    return total


# Fonction calcul 
# Interpolation des courbes de taux
import numpy as np

#convertisseur tx discret à conti
def convertiseur_dis_to_con(r_discret):
    r_continue=np.log(1+r_discret)
    return r_continue
def convertiseur_con_to_dis(r_continue):
    r_discret=np.expm1(r_discret)
# conversion YC à DF CV : 
def yc_to_df(tenor_years,yield_pct):
    df=1/(1+yield_pct)**tenor_years
    return df
def df_to_yc(tenor_years,df):
    yc=(1/df)**(1/tenor_years)-1
    return yc

# Linear Interpolation sur des taux simple 
def linear_interpolation(tenor_years,yields_pct,target_y):
    ten=np.asarray(tenor_years,dtype=float)
    y=np.asarray(yields_pct,dtype=float)
    return float(np.interp(target_y,ten,y))
#Linear sur tx continue
def interp_zero_cont_linear(tenors_years, yields_pct, target_years):
    ten=np.asarray(tenors_years,dtype=float)
    r_s=np.asarray(yields_pct,dtype=float)
    r_c=convertiseur_dis_to_con(r_s)
    r_c_interp=np.interp(target_years,ten,r_c)
    r_s_interp=convertiseur_con_to_dis(r_c_interp)
    return r_s_interp
# Log-linear sur tx simple
def interp_yield_logDF(tenors_years, yields_pct, target_years):
    ten = np.asarray(tenors_years, dtype=float)
    DF  = yc_to_df(ten, yields_pct)
    lnDF = np.log(DF)
    lnDF_t = np.interp(target_years, ten, lnDF)
    DF_t = np.exp(lnDF_t)
    r_s_t = DF_t ** (-1.0 / target_years) - 1.0
    return 100.0 * r_s_t

try:
    from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def interp_yield_pchip(tenors_years, yields_pct, target_years, on="logDF"):
    """
    Interpolation monotone (PCHIP).
    on: "yield" (taux simples %), "cont" (zéros continus), "logDF" (reco).
    """
    if not _HAS_SCIPY:
        # fallback raisonnable
        return interp_yield_logDF(tenors_years, yields_pct, target_years)

    x = np.asarray(tenors_years, dtype=float)
    y_pct = np.asarray(yields_pct, dtype=float)

    if on == "yield":
        y = y_pct
        f = PchipInterpolator(x, y)
        return float(f(target_years))

    elif on == "cont":
        r_c = convertiseur_dis_to_con(y_pct / 100.0)
        f = PchipInterpolator(x, r_c)
        r_c_t = float(f(target_years))
        return 100.0 * convertiseur_con_to_dis(r_c_t)
    else:  # "logDF"
        DF = yc_to_df(x, y_pct)
        lnDF = np.log(DF)
        f = PchipInterpolator(x, lnDF)
        lnDF_t = float(f(target_years))
        DF_t = np.exp(lnDF_t)
        r_s_t = DF_t ** (-1.0 / target_years) - 1.0
        return 100.0 * r_s_t
def interp_yield_akima(tenors_years, yields_pct, target_years, on="logDF"):
    if not _HAS_SCIPY:
        return interp_yield_logDF(tenors_years, yields_pct, target_years)

    x = np.asarray(tenors_years, dtype=float)
    y_pct = np.asarray(yields_pct, dtype=float)

    if on == "yield":
        y = y_pct
        f = Akima1DInterpolator(x, y)
        return float(f(target_years))

    elif on == "cont":
        r_c = convertiseur_dis_to_con(y_pct / 100.0)
        f = Akima1DInterpolator(x, r_c)
        r_c_t = float(f(target_years))
        return 100.0 * convertiseur_con_to_dis(r_c_t)

    else:  # "logDF"
        DF = yc_to_df(x, y_pct)
        lnDF = np.log(DF)
        f = Akima1DInterpolator(x, lnDF)
        lnDF_t = float(f(target_years))
        DF_t = np.exp(lnDF_t)
        r_s_t = DF_t ** (-1.0 / target_years) - 1.0
        return 100.0 * r_s_t

# --- Fallback utilisé par cubic_spline quand SciPy indispo ---
def interp_yield_zero_safefallback(tenors_years, yields_pct, target_years):
    """Fallback: log-DF linéaire (robuste)."""
    return interp_yield_logDF(tenors_years, yields_pct, target_years)




