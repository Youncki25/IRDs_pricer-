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




