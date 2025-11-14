import QuantLib as ql
import pandas as pd
import numpy as np
from typing import Optional, List, Literal

# ------------------------- Mapping utilitaires -------------------------
def get_calendar(name: str) -> ql.Calendar:
    name = name.lower()
    if name in ("target", "eur", "euro"):
        return ql.TARGET()
    if name in ("france", "fr", "paris"):
        return ql.France()
    if name in ("uk", "london", "gb", "unitedkingdom"):
        return ql.UnitedKingdom()
    if name in ("us", "ny", "usa", "unitedstates"):
        return ql.UnitedStates()
    return ql.NullCalendar()

def get_bdc_convention(name: str):
    name = name.lower()
    if name in ("following", "follow", "f"):
        return ql.Following
    if name in ("modifiedfollowing", "mf"):
        return ql.ModifiedFollowing
    if name in ("preceding", "pre", "p"):
        return ql.Preceding
    if name in ("modifiedpreceding", "mp"):
        return ql.ModifiedPreceding
    if name in ("unadjusted", "ua"):
        return ql.Unadjusted
    return ql.ModifiedFollowing

def get_dayaccount(name: str) -> ql.DayCounter:
    name = name.lower()
    if name in ("act/360", "actual/360", "a/360"):
        return ql.Actual360()
    if name in ("act/365", "actual/365", "a/365", "act/365f"):
        return ql.Actual365Fixed()
    if name in ("30/360", "30e/360", "thirty360"):
        return ql.Thirty360()
    return ql.Actual360()

def get_rule(name: str):
    name = name.lower()
    return {
        "forward": ql.DateGeneration.Forward,
        "backward": ql.DateGeneration.Backward,
        "twentieth": ql.DateGeneration.Twentieth,
        "twentiethimm": ql.DateGeneration.TwentiethIMM,
        "oldcds": ql.DateGeneration.OldCDS,
        "cds": ql.DateGeneration.CDS,
    }.get(name, ql.DateGeneration.Forward)

# ------------------------- Schedule -------------------------
def build_schedule(
    start: ql.Date,
    end: ql.Date,
    tenor_months: int,
    calendar_name: str = "TARGET",
    bdc_name: str = "ModifiedFollowing",
    end_of_month: bool = False,
    rule_name: str = "Forward",
    first_date: Optional[ql.Date] = None,         # front stub (optionnel)
    next_to_last_date: Optional[ql.Date] = None,  # back stub  (optionnel)
) -> ql.Schedule:
    cal = get_calendar(calendar_name)
    bdc = get_bdc_convention(bdc_name)
    tenor = ql.Period(tenor_months, ql.Months)
    rule = get_rule(rule_name)

    sched = ql.Schedule(
        start, end, tenor, cal,
        bdc, bdc,
        rule, end_of_month,
        first_date if first_date else ql.Date(),
        next_to_last_date if next_to_last_date else ql.Date()
    )
    return sched

# ------------------------- TA / Amortissement -------------------------
def amort_table(
    start: ql.Date,
    end: ql.Date,
    tenor_months: int,
    daycount_name: str = "ACT/360",
    calendar_name: str = "TARGET",
    bdc_name: str = "ModifiedFollowing",
    end_of_month: bool = False,
    rule_name: str = "Forward",
    first_date: Optional[ql.Date] = None,
    next_to_last_date: Optional[ql.Date] = None,
    notional: float = 1_000_000.0,
    amortization: Literal["in_fine", "lineaire", "custom"] = "lineaire",
    custom_notionals: Optional[List[float]] = None,
    payment_lag_days: int = 1,
    payment_bdc_name: Optional[str] = None,
    payment_calendar_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retourne un DataFrame avec :
    - AccrualStart, AccrualEnd, AccrualDCF
    - PaymentDate
    - NotionalStart, Amort, NotionalEnd
    """
    # 1) Schedule + conventions
    sched = build_schedule(
        start, end, tenor_months,
        calendar_name, bdc_name, end_of_month, rule_name,
        first_date, next_to_last_date
    )
    cal = get_calendar(calendar_name)
    bdc = get_bdc_convention(bdc_name)
    dc = get_dayaccount(daycount_name)

    # 2) Construction des périodes (accual start/end)
    accrual_starts: List[ql.Date] = []
    accrual_ends: List[ql.Date] = []
    for i in range(len(sched) - 1):
        accrual_starts.append(sched[i])
        accrual_ends.append(sched[i + 1])

    # 3) Conventions de paiement
    pay_cal = get_calendar(payment_calendar_name) if payment_calendar_name else cal
    pay_bdc = get_bdc_convention(payment_bdc_name) if payment_bdc_name else bdc

    # 4) Profil de notionnel
    n_periods = len(accrual_starts)
    notionals_start: List[float] = []
    amort_list: List[float] = []
    notionals_end: List[float] = []

    if amortization == "in_fine":
        cur_notional = float(notional)
        for _ in range(n_periods):
            notionals_start.append(cur_notional)
            amort_list.append(0.0)
            notionals_end.append(cur_notional)
        # remboursement total en dernière période
        amort_list[-1] = cur_notional
        notionals_end[-1] = 0.0

    elif amortization == "lineaire":
        cur_notional = float(notional)
        equal_amort = cur_notional / n_periods if n_periods > 0 else 0.0
        for _ in range(n_periods):
            notionals_start.append(cur_notional)
            amort_list.append(equal_amort)
            cur_notional = max(cur_notional - equal_amort, 0.0)
            notionals_end.append(cur_notional)

    elif amortization == "custom":
        if not custom_notionals or len(custom_notionals) != (n_periods + 1):
            raise ValueError("Pour 'custom', fournir custom_notionals de longueur n_periods+1 (notionnel restant à chaque date).")
        for i in range(n_periods):
            ns = float(custom_notionals[i])
            ne = float(custom_notionals[i + 1])
            notionals_start.append(ns)
            amort_list.append(ns - ne)
            notionals_end.append(ne)
    else:
        raise ValueError("amortization doit être 'in_fine', 'lineaire', ou 'custom'.")

    # 5) Accrual DCF + Payment dates
    accrual_dcf: List[float] = []
    payment_dates: List[ql.Date] = []

    def to_ql_date(d) -> ql.Date:
    # si c'est déjà un QuantLib.Date: le réutiliser tel quel (pas de reconstruction)
        if isinstance(d, ql.Date):
            return d
    # pandas.Timestamp ou datetime.date/datetime.datetime
    # -> QuantLib attend (Day, Month, Year) avec des int
        day = int(getattr(d, "day"))
        month = int(getattr(d, "month"))
        year = int(getattr(d, "year"))
        return ql.Date(day, month, year)


    for d0, d1 in zip(accrual_starts, accrual_ends):
        accrual_dcf.append(dc.yearFraction(d0, d1))
        pay = to_ql_date(d1)         # <-- plus de ql.Date(d1) !
        if payment_lag_days:
            pay = pay + int(payment_lag_days)
        pay = pay_cal.adjust(pay, pay_bdc)
        payment_dates.append(pay)


    # 6) Conversion QL Date -> pandas.Timestamp
    def ql_to_tt(d: ql.Date) -> pd.Timestamp:
        return pd.Timestamp(d.year(), d.month(), d.dayOfMonth())

    df = pd.DataFrame({
        "AccrualStart": [ql_to_tt(d) for d in accrual_starts],
        "AccrualEnd":   [ql_to_tt(d) for d in accrual_ends],
        "AccrualDCF":   accrual_dcf,
        "PaymentDate":  [ql_to_tt(d) for d in payment_dates],
        "NotionalStart": notionals_start,
        "Amort":         amort_list,
        "NotionalEnd":   notionals_end,
    })
    return df


def show(df: pd.DataFrame, title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(df.head(10))            # aperçu
    print("\nRésumé:")
    print(df[["AccrualDCF","Amort"]].sum(numeric_only=True))
    # vérifs simples
    total_amort = float(df["Amort"].sum())
    first_notional = float(df.loc[0, "NotionalStart"])
    last_notional  = float(df.iloc[-1]["NotionalEnd"])
    print(f"Check: first_notional={first_notional:,.2f}  last_notional={last_notional:,.2f}  total_amort={total_amort:,.2f}")
    # invariants
    assert abs(first_notional - (total_amort + last_notional)) < 1e-6, "Somme des amortissements incohérente."