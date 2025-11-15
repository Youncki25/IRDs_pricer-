# chemin 1 pour code avec enzo et chemin 2 code pour mon desktop
#streamlit run "/Users/beldjenna/Desktop/GeoKapital-DashBoard/main.py"


# streamlit run "/Users/beldjenna/Desktop/GeoKapital-DashBoard/main.py"

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import pandas_datareader.data as web
import plotly.graph_objects as go
import numpy as np
import QuantLib as ql

# imports locaux
import Curves as curves
import DataUS as data_us
from ta import amort_table
import sofr_future as sf
import ui as U
from config import FRED_API_KEY, BF_API_KEY, APP_TITLE, APP_LAYOUT
import Obligations_fixe


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
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["T_0E"].values, curve["zero_cont"].values, marker="o")
    ax.set_xlabel("Maturité (années, ACT/360)")
    ax.set_ylabel("Taux zéro (continu)")
    ax.set_title("Courbe zéro SOFR (strip SR3)")
    ax.grid(True)

    return df_all, curve, fig


# --------------------- App config & sidebar ---------------------
st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)
st.markdown(U.CSS, unsafe_allow_html=True)
page = U.sidebar()
U.show_sofr_banner("SOFR")

# =============================== PAGES ===============================

# --------------------- Accueil ---------------------
if page == "Accueil":
    st.header("Bienvenue !")
    st.write("Outils de trading et de gestion des risques.")

# --------------------- Pricer ---------------------
elif page == "Pricer":
    st.header("Vous avez le tableau amortissement que vous pouvez lier au produit et aux cash flows.")

    # ===================== Tableau d'amortissement — Config complète =====================
    with st.expander("🧱 Construire un Tableau d’amortissement (TA) – paramètres complets", expanded=True):
        st.caption("Choisis toutes les conventions; on calcule ensuite le TA via QuantLib.")

        # --- Dates & tenor ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ta_start = st.date_input("Date de début (effective)", value=date(2026, 1, 1))
        with c2:
            ta_end   = st.date_input("Date de fin (termination)", value=date(2029, 1, 1), min_value=ta_start)
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
                ql_end   = ql.Date(ta_end.day, ta_end.month, ta_end.year)

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

            @st.cache_data(show_spinner=False)
            def _load_sofr(start=2025, end=2032, r0=0.052, sigma=0.0):
                df_all, curve, fig = fetch_sofr_sr3(
                    start_year=start,
                    end_year=end,
                    r0_overnight=r0,
                    convexity_sigma=sigma
                )
                return df_all, curve, fig

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                start_year = st.number_input("Start year", 2025, 2100, 2025, step=1)
            with col2:
                end_year   = st.number_input("End year",   2025, 2100, 2032, step=1)
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
        st.header("Vous devez remplir les caractéristiques de l'obligation Vanille")

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
        n_float = duree_ans * f # Nb périodes
        n = int(round(n_float))
        if abs(n - n_float) > 1e-8:
            st.warning(f"Durée × fréquence = {n_float:.2f} n'est pas entier → arrondi à {n} périodes.")

        c = coupon / 100.0                           # coupon annuel (décimal)
        y = y_actu / 100.0                           # yield annuel (décimal)
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

# --------------------- Graphique ---------------------
elif page == "Graphique":
    tab = st.selectbox("Choisir", options=["Tableau d'amortissement", "Données macro", "Courbe des taux"], index=0)

    # --- Tableau d’amortissement ---
    if tab == "Tableau d'amortissement":
        if "df_amort" in st.session_state and isinstance(st.session_state["df_amort"], pd.DataFrame):
            st.dataframe(st.session_state["df_amort"], use_container_width=True)
            st.download_button(
                "📥 Télécharger le TA (CSV)",
                data=st.session_state["df_amort"].to_csv(index=False).encode("utf-8"),
                file_name="tableau_amortissement.csv",
                mime="text/csv"
            )
        else:
            st.warning("Aucun TA en mémoire. Va dans 'Pricer' → génère le TA puis reviens ici.")

    # --- Données macro US (exemple) ---
    elif tab == "Données macro":
        Pays = st.selectbox("Quel pays souhaitez-vous ?", options=["US", "France", "Germany", "England", "Europe"], index=0)

        if Pays == "US":
            type1 = st.multiselect("Quel sujet ?", options=["Croissance", "Inflation", "Marché du travail", "Autres"])

            if "Autres" in type1:
                st.subheader("Les données US 🇺🇸 : Courbe des taux (Trésor)")
                try:
                    today = date.today()
                    start = today - timedelta(days=10)
                    maturities = ['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS10','DGS20','DGS30']

                    # 1) Téléchargement FRED
                    df_yield = web.DataReader(maturities, 'fred', start, today)

                    # 2) Dernière date valide et vecteurs triés
                    last_valid = df_yield.dropna().index[-1]
                    curve = df_yield.loc[last_valid]
                    mat_labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "30Y"]

                    # Tenors numériques (années)
                    tenors_years = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30], dtype=float)
                    yields_pct   = np.array([curve[k] for k in maturities], dtype=float)

                    # 3) Grille annuelle 1 → 30 ans
                    grid_years = np.arange(1, 31, dtype=float)

                    # 4) Choix des méthodes à afficher
                    methods = st.multiselect(
                        "Méthodes d'interpolation à tracer",
                        options=["Linéaire (taux %)", "Linéaire (zéro-continu)", "Log-linéaire (DF)", "PCHIP (log-DF)"],
                        default=["Log-linéaire (DF)", "Linéaire (zéro-continu)"]
                    )

                    # 5) Calcul des courbes interpolées
                    curves_dict = {}
                    if "Linéaire (taux %)" in methods:
                        curves_dict["Linéaire (taux %)"] = [curves.linear_interpolation(tenors_years, yields_pct, T) for T in grid_years]
                    if "Linéaire (zéro-continu)" in methods:
                        curves_dict["Linéaire (zéro-continu)"] = [curves.interp_zero_cont_linear(tenors_years, yields_pct, T) for T in grid_years]
                    if "Log-linéaire (DF)" in methods:
                        curves_dict["Log-linéaire (DF)"] = [curves.interp_yield_logDF(tenors_years, yields_pct, T) for T in grid_years]
                    if "PCHIP (log-DF)" in methods:
                        try:
                            from scipy.interpolate import PchipInterpolator  # check dispo
                            curves_dict["PCHIP (log-DF)"] = [curves.interp_yield_pchip(tenors_years, yields_pct, T, on="logDF") for T in grid_years]
                        except Exception:
                            st.info("PCHIP indisponible (SciPy non installé).")

                    # 6) Graphe : points FRED + courbes
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(x=mat_labels, y=yields_pct, mode='markers',
                                                   name=f'FRED points — {last_valid.date()}', marker=dict(size=9)))
                    for name, yvals in curves_dict.items():
                        fig_curve.add_trace(go.Scatter(x=grid_years, y=yvals, mode='lines', name=name))
                    fig_curve.update_layout(
                        title="Courbe des taux US — points FRED + interpolations (1Y → 30Y)",
                        xaxis_title="Maturité (années)", yaxis_title="Taux (%)",
                        template="plotly_white", legend_title="Méthodes"
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur chargement FRED / interpolation : {e}")

    # --- Courbe des taux (placeholder si besoin séparé) ---
    elif tab == "Courbe des taux":
        st.info("Sélectionne 'Données macro' → 'Autres' pour la courbe UST avec FRED.")

# Obligation Pricer 
elif page == "Obligations":
    st.header("📘 Pricing Obligations à taux fixe")
    st.write("Vous pouvez pricer vos obligations, calculer les sensibilités et générer l’échéancier complet.")

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
            "Échéancier détaillé",
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
            if "Échéancier détaillé" in metrics_to_show:
                st.subheader("Échéancier détaillé (cash-flows)")

                periods, times_years, cfs = bond.cashflows()
                accrual_start = [p.accrual_start for p in bond.schedule.periods]
                accrual_end = [p.accrual_end for p in bond.schedule.periods]
                payment_dates = [p.payment_date for p in bond.schedule.periods]
                dcfs = [p.accrual_dcf for p in bond.schedule.periods]

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

