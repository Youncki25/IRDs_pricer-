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