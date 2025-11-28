import streamlit as st
import pandas as pd
from datetime import date

from Obligations_fixe import FixedRateBond


def render():
    st.header("📘 Pricing Obligations à taux fixe")
    st.write("Vous pouvez pricer vos obligations, calculer les sensibilités et générer un échéancier complet.")

    # ===================== 1. Paramètres généraux =====================
    st.subheader("Paramètres de l'obligation")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            nominal = st.number_input(
                "Nominal (€)",
                min_value=0.0,
                value=100_000.0,
                step=1_000.0,
                format="%.2f",
            )
            coupon_pct = st.number_input(
                "Coupon annuel (%)",
                min_value=0.0,
                value=5.00,
                step=0.10,
                format="%.2f",
                help="Saisir 5 pour 5% (pas 0.05).",
            )
            y_actu_pct = st.number_input(
                "Taux actuariel / Yield (%)",
                min_value=0.0,
                value=4.00,
                step=0.10,
                format="%.2f",
                help="Saisir 4 pour 4%.",
            )

        with col2:
            redemption_pct = st.number_input(
                "Remboursement final (% du nominal)",
                min_value=0.0,
                value=100.0,
                step=0.10,
                format="%.2f",
            )
            freq = st.selectbox(
                "Fréquence des coupons",
                options=[1, 2, 4, 12],
                index=1,
                format_func=lambda x: {
                    1: "Annuel (1)",
                    2: "Semestriel (2)",
                    4: "Trimestriel (4)",
                    12: "Mensuel (12)",
                }[x],
            )

    # ===================== 2. Dates, calendrier, stubs =====================
    st.subheader("Échéancier : dates, calendrier, stubs")

    c3, c4, c5 = st.columns(3)
    with c3:
        issue_date = st.date_input(
            "Date d'émission / début",
            value=date(2025, 1, 15),
        )
    with c4:
        maturity_date = st.date_input(
            "Date de maturité",
            value=date(2035, 1, 15),
            min_value=issue_date,
        )
    with c5:
        calendar_name = st.selectbox(
            "Calendrier (accrual/paiement par défaut)",
            options=["TARGET", "France", "UnitedStates", "UnitedKingdom"],
            index=0,
        )

    c6, c7, c8 = st.columns(3)
    with c6:
        daycount_name = st.selectbox(
            "Day count",
            options=["ACT/365", "ACT/360", "30/360"],
            index=0,
        )
    with c7:
        bdc_name = st.selectbox(
            "Business Day Convention",
            options=[
                "ModifiedFollowing",
                "Following",
                "Preceding",
                "ModifiedPreceding",
                "Unadjusted",
            ],
            index=0,
        )
    with c8:
        rule_name = st.selectbox(
            "Règle de génération des dates",
            options=["Forward", "Backward", "Twentieth", "TwentiethIMM", "OldCDS", "CDS"],
            index=0,
        )

    # Stubs
    stubs_col1, stubs_col2, stubs_col3 = st.columns(3)
    first_date = None
    next_to_last_date = None

    with stubs_col1:
        use_front_stub = st.checkbox("Front stub ?", value=False)
    with stubs_col2:
        use_back_stub = st.checkbox("Back stub ?", value=False)
    with stubs_col3:
        payment_lag_days = st.number_input(
            "Payment lag (jours, J+n)",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )

    if use_front_stub:
        first_date = st.date_input(
            "Date du premier coupon (front stub)",
            value=issue_date,
            help="Si différente d'une période régulière, crée un front stub.",
        )
    if use_back_stub:
        next_to_last_date = st.date_input(
            "Avant-dernière date (back stub)",
            value=maturity_date,
            help="Si différente, crée un back stub.",
        )

    # Calendrier de paiement (optionnel)
    pay_c1, pay_c2 = st.columns(2)
    with pay_c1:
        payment_calendar_name = st.selectbox(
            "Calendrier de paiement",
            options=["Idem accrual", "TARGET", "France", "UnitedStates", "UnitedKingdom"],
            index=0,
        )
    with pay_c2:
        payment_bdc_name = st.selectbox(
            "BDC paiement",
            options=[
                "Idem accrual",
                "Following",
                "ModifiedFollowing",
                "Preceding",
                "ModifiedPreceding",
                "Unadjusted",
            ],
            index=0,
        )

    if payment_calendar_name == "Idem accrual":
        payment_calendar_name = calendar_name
    if payment_bdc_name == "Idem accrual":
        payment_bdc_name = bdc_name

    # ===================== 3. Ce que l'utilisateur veut voir =====================
    st.subheader("Sorties à afficher")

    metrics_to_show = st.multiselect(
        "Que veux-tu afficher ?",
        options=[
            "Prix",
            "Duration de Macaulay",
            "Duration modifiée",
            "Convexité",
            "DV01",
            "Flows",
        ],
        default=["Prix", "Duration de Macaulay", "Duration modifiée", "DV01"],
    )

    # ===================== 4. Calcul =====================
    if st.button("📌 Calculer l'obligation", type="primary"):
        try:
            coupon_rate = coupon_pct / 100.0
            y_actu = y_actu_pct / 100.0
            redemption = redemption_pct / 100.0

            # Construction via Objet
            bond = FixedRateBond.from_dates(
                nominal=nominal,
                coupon_rate=coupon_rate,
                issue_date=issue_date,
                maturity_date=maturity_date,
                freq=int(freq),
                calendar_name=calendar_name,
                bdc_name=bdc_name,
                daycount_name=daycount_name,
                rule_name=rule_name,
                end_of_month=False,
                first_date=first_date,
                next_to_last_date=next_to_last_date,
                payment_lag_days=int(payment_lag_days),
                payment_calendar_name=payment_calendar_name,
                payment_bdc_name=payment_bdc_name,
                redemption=redemption,
            )

            prix = bond.price(y_actu)
            macaulay = bond.macaulay_duration(y_actu)
            modified = bond.modified_duration(y_actu)
            conv = bond.convexity(y_actu)
            dv01 = bond.dv01(y_actu)

            # ----- Résultats principaux -----
            st.subheader("Résultats")

            cols = st.columns(4)
            if "Prix" in metrics_to_show:
                cols[0].metric("💶 Prix (VA)", f"{prix:,.2f}")
            if "Duration de Macaulay" in metrics_to_show:
                cols[1].metric("Duration de Macaulay (ans)", f"{macaulay:.4f}")
            if "Duration modifiée" in metrics_to_show:
                cols[2].metric("Duration modifiée (ans)", f"{modified:.4f}")
            if "Convexité" in metrics_to_show:
                cols[3].metric("Convexité (annuelle)", f"{conv:.4f}")

            if "DV01" in metrics_to_show:
                st.caption(f"DV01 ≈ {dv01:,.4f} € par 1 bp (0.01%) de mouvement de taux.")

            # ----- Échéancier détaillé -----
            if "Flows" in metrics_to_show:
                st.subheader("Échéancier détaillé (cash-flows)")

                periods, times_years, cfs = bond.cashflows()
                accrual_start = [p.accrual_start for p in bond.schedule.periods]
                accrual_end = [p.accrual_end for p in bond.schedule.periods]
                payment_dates = [p.payment_date for p in bond.schedule.periods]
                dcfs = [p.accrual_dcf for p in bond.schedule.periods]
                is_stub = [p.is_stub for p in bond.schedule.periods]
                stub_type = [p.stub_type for p in bond.schedule.periods]

                r_per = y_actu / freq
                disc = 1.0 / (1.0 + r_per) ** periods
                pv = cfs * disc

                df_sched = pd.DataFrame({
                    "Période": periods.astype(int),
                    "Accrual start": [d.to_date() for d in accrual_start],
                    "Accrual end": [d.to_date() for d in accrual_end],
                    "Payment date": [d.to_date() for d in payment_dates],
                    "DCF": dcfs,
                    "Temps (ans cumulés)": times_years,
                    "Cash-flow": cfs,
                    "Facteur d'actualisation": disc,
                    "Valeur actuelle": pv,
                    # --- meta ajoutées ---
                    "Stub?": is_stub,
                    "Type de stub": stub_type,                        # front / back / regular
                    "Calendrier accrual": bond.schedule.calendar_name,
                    "Calendrier paiement": bond.schedule.payment_calendar_name,
                    "BDC accrual": bond.schedule.bdc_name,
                    "BDC paiement": bond.schedule.payment_bdc_name,
                    "Day-count": bond.schedule.daycount_name,
                    "Tenor (mois)": bond.schedule.tenor_months,
                    "Payment lag (jours)": bond.schedule.payment_lag_days,
                    "Fréquence théorique": bond.freq_label,
                })

                st.dataframe(
                    df_sched.style.format({
                        "DCF": "{:.6f}",
                        "Temps (ans cumulés)": "{:.4f}",
                        "Cash-flow": "{:,.2f}",
                        "Facteur d'actualisation": "{:.6f}",
                        "Valeur actuelle": "{:,.2f}",
                    }),
                    use_container_width=True,
                )

                st.download_button(
                    "📥 Télécharger l'échéancier (CSV)",
                    data=df_sched.to_csv(index=False).encode("utf-8"),
                    file_name="bond_schedule_cashflows.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Erreur lors du pricing de l'obligation : {e}")
