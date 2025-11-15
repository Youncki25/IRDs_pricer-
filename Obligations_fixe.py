from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import date

import numpy as np
import QuantLib as ql


# ===========================
# Utilitaires QuantLib
# ===========================

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


def get_bdc(name: str):
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


def get_daycount(name: str) -> ql.DayCounter:
    name = name.lower()
    if name in ("act/360", "actual/360", "a/360"):
        return ql.Actual360()
    if name in ("act/365", "actual/365", "a/365", "act/365f"):
        return ql.Actual365Fixed()
    if name in ("30/360", "30e/360", "thirty360"):
        return ql.Thirty360()
    return ql.ActualActual()


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


def pydate_to_ql(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


# ===========================
# Échéancier d'obligation
# ===========================

@dataclass
class PaymentPeriod:
    accrual_start: ql.Date
    accrual_end: ql.Date
    accrual_dcf: float
    payment_date: ql.Date
    # --- infos supplémentaires ---
    is_stub: bool = False              # True si période de stub
    stub_type: str = "regular"         # "front", "back" ou "regular"


@dataclass
class BondSchedule:
    periods: List[PaymentPeriod]

    # méta sur les conventions
    calendar_name: str = "TARGET"
    payment_calendar_name: str = "TARGET"
    bdc_name: str = "ModifiedFollowing"
    payment_bdc_name: str = "ModifiedFollowing"
    daycount_name: str = "ACT/365"
    tenor_months: int = 6
    payment_lag_days: int = 0

    @classmethod
    def build(
        cls,
        issue_date: date,
        maturity_date: date,
        tenor_months: int,
        calendar_name: str = "TARGET",
        bdc_name: str = "ModifiedFollowing",
        daycount_name: str = "ACT/365",
        rule_name: str = "Forward",
        end_of_month: bool = False,
        # stubs optionnels :
        first_date: Optional[date] = None,         # front stub date (première accrual end)
        next_to_last_date: Optional[date] = None,  # back stub date (avant-dernière accrual end)
        payment_lag_days: int = 0,
        payment_calendar_name: Optional[str] = None,
        payment_bdc_name: Optional[str] = None,
    ) -> "BondSchedule":
        """
        Construit un échéancier complet avec calendrier, day-count, BDC, lag
        et tags de stub (front/back).
        """
        cal = get_calendar(calendar_name)
        bdc = get_bdc(bdc_name)
        dc = get_daycount(daycount_name)
        rule = get_rule(rule_name)

        ql_start = pydate_to_ql(issue_date)
        ql_end = pydate_to_ql(maturity_date)
        tenor = ql.Period(tenor_months, ql.Months)

        ql_first = pydate_to_ql(first_date) if first_date else ql.Date()
        ql_next_to_last = pydate_to_ql(next_to_last_date) if next_to_last_date else ql.Date()

        sched = ql.Schedule(
            ql_start,
            ql_end,
            tenor,
            cal,
            bdc,
            bdc,
            rule,
            end_of_month,
            ql_first,
            ql_next_to_last,
        )

        pay_cal = get_calendar(payment_calendar_name) if payment_calendar_name else cal
        pay_bdc = get_bdc(payment_bdc_name) if payment_bdc_name else bdc

        periods: List[PaymentPeriod] = []

        has_front_stub = first_date is not None
        has_back_stub = next_to_last_date is not None
        n_qldates = len(sched)

        for i in range(n_qldates - 1):
            d0 = sched[i]
            d1 = sched[i + 1]
            dcf = dc.yearFraction(d0, d1)

            # date de paiement = end date + lag, ajustée calendrier de paiement
            pay = d1
            if payment_lag_days:
                pay = pay + int(payment_lag_days)
            pay = pay_cal.adjust(pay, pay_bdc)

            # tag stub
            is_stub = False
            stub_type = "regular"
            if has_front_stub and i == 0:
                is_stub = True
                stub_type = "front"
            if has_back_stub and i == n_qldates - 2:
                is_stub = True
                stub_type = "back"

            periods.append(
                PaymentPeriod(
                    accrual_start=d0,
                    accrual_end=d1,
                    accrual_dcf=float(dcf),
                    payment_date=pay,
                    is_stub=is_stub,
                    stub_type=stub_type,
                )
            )

        return cls(
            periods=periods,
            calendar_name=calendar_name,
            payment_calendar_name=payment_calendar_name or calendar_name,
            bdc_name=bdc_name,
            payment_bdc_name=payment_bdc_name or bdc_name,
            daycount_name=daycount_name,
            tenor_months=tenor_months,
            payment_lag_days=payment_lag_days,
        )

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def total_years(self) -> float:
        """Approximation de la maturité en années via la somme des DCF."""
        return float(sum(p.accrual_dcf for p in self.periods))

    @property
    def has_front_stub(self) -> bool:
        return any(p.stub_type == "front" for p in self.periods)

    @property
    def has_back_stub(self) -> bool:
        return any(p.stub_type == "back" for p in self.periods)


# ===========================
# Obligation à taux fixe
# ===========================

@dataclass
class FixedRateBond:
    """
    Obligation à taux fixe avec échéancier QuantLib.

    coupon_rate : taux annuel en décimal (5% -> 0.05)
    redemption  : 1.0 = 100% du nominal
    """
    nominal: float
    coupon_rate: float
    schedule: BondSchedule
    freq: int = 2          # fréquence "théorique" des coupons (pour le taux actuariel)
    redemption: float = 1.0

    # ------- Cash-flows générés par l'échéancier -------
    def cashflows(self):
        """
        Retourne:
          - periods: 1..N
          - times: temps en années (somme cumulée des DCF)
          - cfs: flux de trésorerie (coupons + remboursement final)
        """
        n = self.schedule.n_periods
        periods = np.arange(1, n + 1, dtype=float)
        dcfs = np.array([p.accrual_dcf for p in self.schedule.periods], dtype=float)
        times = np.cumsum(dcfs)

        # coupon proportionnel au DCF : Nominal * coupon_rate * DCF
        coupons = self.nominal * self.coupon_rate * dcfs
        cfs = coupons.copy()
        # remboursement final
        cfs[-1] += self.nominal * self.redemption

        return periods, times, cfs

    @property
    def freq_label(self) -> str:
        mapping = {
            1: "Annuel (1)",
            2: "Semestriel (2)",
            4: "Trimestriel (4)",
            12: "Mensuel (12)",
        }
        return mapping.get(self.freq, f"{self.freq} fois par an")

    # ------- Pricing & risques -------
    def price(self, yield_rate: float) -> float:
        """
        Prix de l'obligation pour un taux actuariel 'yield_rate'
        (en décimal, ex: 0.04 pour 4%).

        On utilise un schéma discret :
          v = 1 / (1 + y/freq)^t
        où t = nº de période (1,2,...).
        Les périodes peuvent être stub, mais on garde la même fréquence
        de capitalisation pour le taux actuariel.

        Retourne un prix **par nominal** (ex: 1.08 = 108% du nominal si nominal=1).
        """
        periods, _, cfs = self.cashflows()
        r_per = yield_rate / self.freq       # taux par période
        disc = 1.0 / (1.0 + r_per) ** periods
        pv = cfs * disc
        return float(pv.sum() / self.nominal)

    def macaulay_duration(self, yield_rate: float) -> float:
        """
        Duration de Macaulay (en années).
        """
        periods, times, cfs = self.cashflows()
        r_per = yield_rate / self.freq
        disc = 1.0 / (1.0 + r_per) ** periods
        pv = cfs * disc
        prix = float(pv.sum())
        if prix == 0:
            return float("nan")

        macaulay = float((times * pv).sum() / prix)
        return macaulay

    def modified_duration(self, yield_rate: float) -> float:
        """
        Duration modifiée (en années).
        """
        macaulay = self.macaulay_duration(yield_rate)
        r_per = yield_rate / self.freq
        return macaulay / (1.0 + r_per)

    def convexity(self, yield_rate: float) -> float:
        """
        Convexité annuelle (approx discrète).
        """
        periods, times, cfs = self.cashflows()
        r_per = yield_rate / self.freq
        disc = 1.0 / (1.0 + r_per) ** periods
        pv = cfs * disc
        prix = float(pv.sum())
        if prix == 0:
            return float("nan")

        num = (pv * times * (times + 1.0)).sum()
        conv = float(num / ((1 + r_per) ** 2 * prix))
        return conv

    def dv01(self, yield_rate: float) -> float:
        """
        DV01 ≈ variation de prix (par nominal) pour +1 bp sur le taux.
        """
        mod_dur = self.modified_duration(yield_rate)
        price_0 = self.price(yield_rate)
        dy = 1e-4  # 1 bp
        dP = -mod_dur * price_0 * dy
        return -dP

    @classmethod
    def from_dates(
        cls,
        nominal: float,
        coupon_rate: float,
        issue_date: date,
        maturity_date: date,
        freq: int,
        calendar_name: str = "TARGET",
        bdc_name: str = "ModifiedFollowing",
        daycount_name: str = "ACT/365",
        rule_name: str = "Forward",
        end_of_month: bool = False,
        first_date: Optional[date] = None,
        next_to_last_date: Optional[date] = None,
        payment_lag_days: int = 0,
        payment_calendar_name: Optional[str] = None,
        payment_bdc_name: Optional[str] = None,
        redemption: float = 1.0,
    ) -> "FixedRateBond":
        """
        Fabrique directement une obligation en construisant son échéancier QuantLib,
        avec support de stub si first_date / next_to_last_date sont renseignés.
        """
        tenor_months = int(12 / freq)   # ex: freq=2 -> 6M; freq=1 -> 12M, etc.

        schedule = BondSchedule.build(
            issue_date=issue_date,
            maturity_date=maturity_date,
            tenor_months=tenor_months,
            calendar_name=calendar_name,
            bdc_name=bdc_name,
            daycount_name=daycount_name,
            rule_name=rule_name,
            end_of_month=end_of_month,
            first_date=first_date,
            next_to_last_date=next_to_last_date,
            payment_lag_days=payment_lag_days,
            payment_calendar_name=payment_calendar_name,
            payment_bdc_name=payment_bdc_name,
        )

        return cls(
            nominal=nominal,
            coupon_rate=coupon_rate,
            schedule=schedule,
            freq=freq,
            redemption=redemption,
        )
