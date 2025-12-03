import streamlit as st
from datetime import date
import plotly.express as px
import plotly.graph_objects as go

from ta_emprunt import (
    tableau_amortissement_emprunt,
    tableau_amortissement_emprunt_variable,
)


# ---------- Helpers format FR ----------

def format_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def format_eur(n: float, decimals: int = 2) -> str:
    fmt = f"{{:,.{decimals}f}}".format(n)
    return fmt.replace(",", " ").replace(".", ",")


def format_pct_fr(x: float, decimals: int = 2) -> str:
    s = f"{x * 100:.{decimals}f}".replace(".", ",")
    return s + " %"


# ---------- IRR maison pour le TAEG ----------

def irr_periodique_bissection(cash_flows, tol=1e-7, max_iter=100):
    def npv(r):
        total = 0.0
        for t, cf in enumerate(cash_flows):
            total += cf / ((1 + r) ** t)
        return total

    r_low, r_high = 0.0, 1.0
    npv_low = npv(r_low)
    npv_high = npv(r_high)

    if npv_low * npv_high > 0:
        return None

    for _ in range(max_iter):
        r_mid = (r_low + r_high) / 2
        npv_mid = npv(r_mid)

        if abs(npv_mid) < tol:
            return r_mid

        if npv_low * npv_mid < 0:
            r_high = r_mid
            npv_high = npv_mid
        else:
            r_low = r_mid
            npv_low = npv_mid

    return (r_low + r_high) / 2


# ---------- Page Streamlit ----------

def render():
    st.header("📄 Tableau d'amortissement — Emprunt bancaire")

    # ---- Type de taux ----
    type_taux = st.radio(
        "Type de taux",
        ["Taux fixe", "Taux variable"],
        horizontal=True,
    )

    # ---- Inputs principaux ----
    capital = st.number_input(
        "Montant emprunté",
        value=1_000_000,
        step=50_000,
        format="%d",
        help="Montant du prêt (ex : 1 000 000 €).",
    )

    date_debut = st.date_input(
        "Date de début du prêt",
        value=date(2026, 1, 1),
    )

    duree = st.number_input(
        "Durée (années)",
        value=10,
        min_value=1,
        step=1,
    )

    freq = st.selectbox(
        "Fréquence des paiements",
        [12, 4, 1],
        index=0,
        format_func=lambda x: {
            12: "Mensuel (12)",
            4: "Trimestriel (4)",
            1: "Annuel (1)",
        }[x],
    )

    # ---- Choix du taux ----
    if type_taux == "Taux fixe":
        taux_pct = st.number_input(
            "Taux annuel fixe (%)",
            value=4.0,
            step=0.1,
            format="%.2f",
            help="Ex : 4 pour 4%.",
        )
        taux_variables = None
    else:
        taux_var_str = st.text_input(
            "Taux annuels par année (%)",
            value="3.0, 3.5, 4.0",
            help="Ex : 3, 3.5, 4 pour : année 1 → 3 %, année 2 → 3,5 %, année 3 → 4 %, etc.",
        )
        taux_variables = taux_var_str

    # ---- Inputs spécifiques TAEG ----
    col1, col2, col3 = st.columns(3)
    with col1:
        frais_dossier = st.number_input(
            "Frais de dossier (€)",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )
    with col2:
        frais_garantie = st.number_input(
            "Frais de garantie (€)",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )
    with col3:
        assurance_taux_pct = st.number_input(
            "Assurance emprunteur (% du capital / an)",
            min_value=0.0,
            value=0.0,
            step=0.05,
            format="%.2
