import streamlit as st
from ta_emprunt import tableau_amortissement_emprunt
from datetime import date

def format_int(n):
    """Affiche 1 000 000 au lieu de 1000000"""
    return f"{int(n):,}".replace(",", " ")

def render():
    st.header("📄 Tableau d'amortissement — Emprunt bancaire")

    capital = st.number_input(
        "Montant emprunté",
        value=1_000_000,
        step=50_000,
        format="%d",
        help="Montant du prêt (ex : 1 000 000 €)."
    )

    taux_pct = st.number_input(
        "Taux annuel (%)",
        value=4.0,
        step=0.1,
        format="%.2f",
        help="Ex : 4 pour 4%."
    )

    date_debut = st.date_input(
        "Date de début du prêt",
        value=date(2026, 1, 1)
    )

    duree = st.number_input(
        "Durée (années)",
        value=10,
        min_value=1,
        step=1
    )

    freq = st.selectbox(
        "Fréquence des paiements",
        [12, 4, 1],
        index=0,
        format_func=lambda x: {12: "Mensuel (12)", 4: "Trimestriel (4)", 1: "Annuel (1)"}[x]
    )

    if st.button("Générer le tableau", type="primary"):

        taux = taux_pct / 100

        df = tableau_amortissement_emprunt(
            capital_initial=capital,
            taux_annuel=taux,
            date_debut=date_debut,
            duree_annees=int(duree),
            paiements_par_an=int(freq)
        )

        st.success(f"Tableau d'amortissement généré pour un prêt de **{format_int(capital)} €** 💶")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "📥 Télécharger (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="tableau_amortissement_emprunt.csv",
            mime="text/csv"
        )
