import streamlit as st
import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

import QuantLib as ql

# imports locaux
import sofr_future as sf
from ta import amort_table
from Obligations_fixe import FixedRateBond


# --------------------- SOFR / SR3 helper ---------------------
def fetch_sofr_sr3(start_year=2025, end_year=2032, r0_overnight=0.052, convexity_sigma=0.0):
    """
    Construit la courbe zéro SOFR via SR3 (fonctions dans sofr_future.py) :
      - sr3_symbols, download_last_close, build_sofr_curve_from_sr3
    Retourne: df_all (meta+prix), curve (table finale), fig (matplotlib)
    """
    # 1) Génère les symboles + méta (start/end date)
    symbols, meta_df = sf.sr3_symbols(start_year, end_year)
    meta_df = meta_df.sort_values(["start_date", "symbol"]).reset_index(drop=True)

    # 2) Télécharge les derniers cours de clôture
    snapshot, _ = sf.download_last_close(symbols, period="6mo", interval="1d", auto_adjust=False)

    # 3) Merge et nettoyage
    df_all = meta_df.merge(snapshot, on="symbol", how="left")
    df_clean = df_all[df_all["last_close"].notna()].copy()

    # 4) Construit la courbe (fwd → DF → zero)
    curve = sf.build_sofr_curve_from_sr3(
        df_clean,
        r0_overnight=r0_overnight,
        convexity_sigma=convexity_sigma
    )

    # 5) Figure matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["T_0E"].values, curve["zero_cont"].values, marker="o")
    ax.set_xlabel("Maturité (années, ACT/360)")
    ax.set_ylabel("Taux zéro (continu)")
    ax.set_title("Courbe zéro SOFR (strip SR3)")
    ax.grid(True)

    return df_all, curve, fig


@st.cache_data(show_spinner=False)
def _load_sofr(start=2025, end=2032, r0=0.052, sigma=0.0):
    df_all, curve, fig = fetch_sofr_sr3(
        start_year=start,
        end_year=end,
        r0_overnight=r0,
        convexity_sigma=sigma
    )
    return df_all, curve, fig


def render():
    st.header("Tableau d’amortissement & pricer de produits de taux")

    # ===================== Tableau d'amortissement — Config complète =====================
    with st.expander("🧱 Construire un Tableau d’amortissement (TA) – paramètres complets", expanded=True):
        st.caption("Choisis toutes les conventions; on calcule ensuite le TA via QuantLib.")

        # --- Dates & tenor ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ta_start = st.date_input("Date de début (effective)", value=date(2026, 1, 1))
        with c2:
            ta_end = st.date_input("Date de fin (termination)", value=date(2029, 1, 1), min_value=ta_start)
        with c3:
            tenor_months = st.number_input("Tenor (mois)", min_value=1, max_value=1000, value=6, step=1)
        with c4:
            end_of_month = st.toggle("End-of-Month (EOM)", value=False)

        # --- Calendriers & conventions ---
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            calendar_name = st.selectbox(
                "Calendrier (accrual)",
                ["TARGET", "France", "UnitedKingdom", "UnitedStates"],
                index=0
            )
        with c6:
            bdc_name = st.selectbox(
                "BDC (accrual)",
                ["ModifiedFollowing", "Following", "Preceding", "ModifiedPreceding", "Unadjusted"],
                index=0
            )
        with c7:
            daycount_name = st.selectbox(
                "Day Count",
                ["ACT/360", "ACT/365", "30/360"],
                index=0
            )
        with c8:
            rule_name = st.selectbox(
                "Règle de génération",
                ["Forward", "Backward", "Twentieth", "TwentiethIMM", "OldCDS", "CDS"],
                index=0
            )

        # --- Stubs (optionnels) ---
        stubs = st.toggle("Activer des stubs (front/back) ?", value=False)
        first_date = None
        next_to_last_date = None
        if stubs:
            s1, s2 = st.columns(2)
            with s1:
                fd = st.date_input("Front stub (firstDate)", value=None, key="ta_fd", format="YYYY-MM-DD")
                first_date = None if fd is None else ql.Date(fd.day, fd.month, fd.year)
            with s2:
                nd = st.date_input("Back stub (nextToLastDate)", value=None, key="ta_nd", format="YYYY-MM-DD")
                next_to_last_date = None if nd is None else ql.Date(nd.day, nd.month, nd.year)

        # --- Paiement (lag & conventions) ---
        c9, c10, c11 = st.columns(3)
        with c9:
            payment_lag_days = st.number_input("Payment lag (jours, J+n)", min_value=0, max_value=30, value=2, step=1)
        with c10:
            payment_calendar_name = st.selectbox(
                "Calendrier (paiement)",
                ["TARGET", "France", "UnitedKingdom", "UnitedStates"],
                index=0
            )
        with c11:
            payment_bdc_name = st.selectbox(
                "BDC (paiement)",
                ["Following", "ModifiedFollowing", "Preceding", "ModifiedPreceding", "Unadjusted"],
                index=0
            )

        # --- Notionnel & profil d'amortissement ---
        c12, c13, c14 = st.columns(3)
        with c12:
            notional = st.number_input("Notionnel", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%.2f")
        with c13:
            amortization = st.selectbox("Amortissement", ["lineaire", "in_fine", "custom"], index=0)
        with c14:
            custom_str = st.text_input(
                "Custom notionnels (niveaux restants, séparés par des virgules) — si 'custom'",
                placeholder="ex: 10_000_000, 8_000_000, 6_000_000, 4_000_000, 2_000_000, 0"
            )

        # --- Bouton de génération TA ---
        if st.button("⚙️ Générer le TA maintenant", type="primary"):
            try:
                # conversions dates → QuantLib
                ql_start = ql.Date(ta_start.day, ta_start.month, ta_start.year)
                ql_end = ql.Date(ta_end.day, ta_end.month, ta_end.year)

                # parser custom notionals si demandé
                custom_notionals = None
                if amortization == "custom":
                    if not custom_str.strip():
                        st.error("Veuillez renseigner 'Custom notionnels' ou choisissez un autre type d’amortissement.")
                        st.stop()
                    try:
                        cleaned = custom_str.replace("€", "").replace("%", "").replace("_", "").replace(" ", "")
                        custom_notionals = [float(x) for x in cleaned.split(",") if x != ""]
                    except Exception:
                        st.error("Format des 'Custom notionnels' invalide. Exemple: 10000000,8000000,6000000,4000000,2000000,0")
                        st.stop()

                # appelle TA
                df_amort = amort_table(
                    start=ql_start,
                    end=ql_end,
                    tenor_months=int(tenor_months),
                    daycount_name=daycount_name,
                    calendar_name=calendar_name,
                    bdc_name=bdc_name,
                    end_of_month=bool(end_of_month),
                    rule_name=rule_name,
                    first_date=first_date,
                    next_to_last_date=next_to_last_date,
                    notional=float(notional),
                    amortization=amortization,
                    custom_notionals=custom_notionals,
                    payment_lag_days=int(payment_lag_days),
                    payment_bdc_name=payment_bdc_name,
                    payment_calendar_name=payment_calendar_name,
                )

                st.success("TA généré ✅")
                st.dataframe(df_amort, use_container_width=True)

                # garde-le pour l’onglet Graphique
                st.session_state["df_amort"] = df_amort

                # export CSV
                st.download_button(
                    "📥 Télécharger le TA (CSV)",
                    data=df_amort.to_csv(index=False).encode("utf-8"),
                    file_name="tableau_amortissement.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Erreur lors de la génération du TA : {e}")

    # ===================== Sélecteur de produit (SOFR, Bonds, etc.) =====================
    st.header("Pricer de produits de taux 💸")
    produit = st.selectbox(
        "Quel produit ?",
        options=[
            "Swap Plain Vanilla", "Forward", "Basis Swap",
            "STIR on RFR", "STIR on XIBOR", "CCS", "Option",
            "Vanilla Bonds"
        ],
        index=0
    )

    # === STIR on RFR / SOFR ===
    if produit == "STIR on RFR":
        indice = st.selectbox("Indice RFR", ["SOFR", "ESTR", "SONIA"], index=0)
        if indice == "SOFR":
            st.subheader("SOFR via strip SR3 (CME)")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                start_year = st.number_input("Start year", 2025, 2100, 2025, step=1)
            with col2:
                end_year = st.number_input("End year", 2025, 2100, 2032, step=1)
            with col3:
                r0 = st.number_input("ON/OIS court terme (r0)", 0.0, 1.0, 0.052, step=0.001, format="%.3f")
            with col4:
                sigma = st.number_input("Sigma convexité (optionnel)", 0.0, 1.0, 0.0, step=0.001, format="%.3f")

            with st.spinner("Téléchargement des prix SR3 et construction de la courbe…"):
                try:
                    df_all, curve, fig = _load_sofr(start_year, end_year, r0, sigma)
                except Exception as e:
                    st.error(f"Erreur during SR3/SOFR: {e}")
                    st.stop()

            # Figure matplotlib
            st.pyplot(fig)

            # Tableau d’aperçu
            st.markdown("**Tableau (extrait)** — `symbol`, `T_0E`, `zero_cont`, `zero_simple`")
            st.dataframe(curve[["symbol", "T_0E", "zero_cont", "zero_simple"]], use_container_width=True)

            # Export
            st.download_button(
                "📥 Télécharger la courbe (CSV)",
                data=curve.to_csv(index=False).encode("utf-8"),
                file_name="SOFR_curve_from_SR3.csv",
                mime="text/csv"
            )

    # === VANILLA BONDS ===
    if produit == "Vanilla Bonds":
        st.header("Obligation vanille — prix & sensibilités")

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                nominal = st.number_input("Nominal :", min_value=0.0, value=100_000.0, step=1_000.0, format="%.2f")
                coupon = st.number_input(
                    "Coupon (en % annuel) :",
                    min_value=0.0, value=5.00, step=0.10, format="%.2f",
                    help="Saisir 5 pour 5% (pas 0.05)."
                )
                duree_ans = st.number_input("Durée (en années) :", min_value=0.25, value=10.0, step=0.25, format="%.2f")
                redemption_pct = st.number_input(
                    "Remboursement final (% du nominal) :",
                    min_value=0.0, value=100.0, step=0.10, format="%.2f"
                )

            with col2:
                # Fréquence = périodes par an
                freq = st.selectbox(
                    "Fréquence des coupons",
                    options=[1, 2, 4, 12],
                    index=1,
                    format_func=lambda x: {1: "Annuel (1)", 2: "Semestriel (2)", 4: "Trimestriel (4)", 12: "Mensuel (12)"}[x]
                )
                y_actu = st.number_input(
                    "Taux d'actualisation (en % annuel) :",
                    min_value=0.0, value=4.00, step=0.10, format="%.2f",
                    help="Saisir 4 pour 4%."
                )
                show_table = st.checkbox("Afficher le détail des flux actualisés", value=True)

        # ---- Calculs ----
        f = int(freq)  # périodes/an (1,2,4,12)
        n_float = duree_ans * f  # Nb périodes
        n = int(round(n_float))
        if abs(n - n_float) > 1e-8:
            st.warning(f"Durée × fréquence = {n_float:.2f} n'est pas entier → arrondi à {n} périodes.")

        c = coupon / 100.0  # coupon annuel (décimal)
        y = y_actu / 100.0  # yield annuel (décimal)
        redemption = (redemption_pct / 100.0) * nominal

        coupon_periodique = nominal * c / f
        r_per = y / f
        ts = np.arange(1, n + 1, dtype=float)

        cashflows = np.full(n, coupon_periodique, dtype=float)
        cashflows[-1] += redemption

        disc = 1.0 / (1.0 + r_per) ** ts
        pv = cashflows * disc
        prix = float(np.sum(pv))

        # Durations & convexité
        if prix > 0:
            t_years = ts / f
            macaulay = float(np.sum(t_years * pv) / prix)
            modified = macaulay / (1.0 + y / f)
            conv = float(np.sum(pv * ts * (ts + 1)) / ((1 + r_per) ** 2 * prix * f ** 2))
            dv01 = modified * prix / 10000.0
        else:
            macaulay = modified = conv = dv01 = float("nan")

        st.subheader("Résultats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💶 Prix (VA)", f"{prix:,.2f}")
        c2.metric("Duration de Macaulay (ans)", f"{macaulay:.4f}")
        c3.metric("Duration modifiée (ans)", f"{modified:.4f}")
        c4.metric("Convexité (annuelle)", f"{conv:.4f}")

        st.caption(f"DV01 ≈ {dv01:,.4f} par 1 bp — ΔPrix ≈ −Duration_mod × Prix × Δtaux.")

        # Sensibilité instantanée ±100 bps
        colA, colB = st.columns(2)
        with colA:
            dy_bp = st.slider("Choc de taux (bps)", min_value=-200, max_value=200, value=+100, step=5)
            dy = dy_bp / 10000.0
            dP_lin = -modified * prix * dy
            dP_conv = 0.5 * conv * prix * (dy ** 2)
            approx_price = prix + dP_lin + dP_conv
            st.info(f"Prix approx. pour Δtaux = {dy_bp:+} bps → **{approx_price:,.2f}** (linéaire + convexité)")

        with colB:
            y_new = max(y + dy, 0.0)
            r_per_new = y_new / f
            disc_new = 1.0 / (1.0 + r_per_new) ** ts
            prix_exact = float(np.sum(cashflows * disc_new))
            st.success(f"Prix recalculé (exact) → **{prix_exact:,.2f}**")

        # Tableau
        if show_table:
            df_bond = pd.DataFrame({
                "Période (t)": ts.astype(int),
                "Temps (ans)": ts / f,
                "Flux": cashflows,
                "Facteur d'actualisation": disc,
                "Valeur actuelle": pv
            })
            st.dataframe(
                df_bond.style.format({
                    "Temps (ans)": "{:.4f}",
                    "Flux": "{:,.2f}",
                    "Facteur d'actualisation": "{:.6f}",
                    "Valeur actuelle": "{:,.2f}",
                }),
                use_container_width=True
            )

            st.download_button(
                "📥 Télécharger les flux (CSV)",
                data=df_bond.to_csv(index=False).encode("utf-8"),
                file_name="vanilla_bond_cashflows.csv",
                mime="text/csv"
            )

        with st.expander("Hypothèses & rappels"):
            st.markdown(
                "- **Actualisation discrète** à la **même fréquence** que les coupons.\n"
                "- **Pas de dates** → pas d’accrued/clean price; on affiche le **prix total**.\n"
                "- Formules: Prix = Σ CF_t / (1+y/f)^t ; Duration modifiée = Macaulay / (1+y/f)."
            )
