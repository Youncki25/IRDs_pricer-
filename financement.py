import streamlit as st
from ta_emprunt import tableau_amortissement_emprunt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go


def format_int(n: float) -> str:
    """Affiche 1 000 000 au lieu de 1000000."""
    return f"{int(round(n)):,}".replace(",", " ")


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
        format_func=lambda x: {
            12: "Mensuel (12)",
            4: "Trimestriel (4)",
            1: "Annuel (1)"
        }[x]
    )

    if st.button("Générer le tableau", type="primary"):

        taux = taux_pct / 100.0

        df = tableau_amortissement_emprunt(
            capital_initial=capital,
            taux_annuel=taux,
            date_debut=date_debut,
            duree_annees=int(duree),
            paiements_par_an=int(freq)
        )

        # ===== Annuité constante (en blanc, en gras) =====
        annuite = df["Mensualité (€)"].iloc[0]
        st.markdown(
            f"""
            <div style="margin-top:0.5rem; margin-bottom:0.8rem;
                        font-size:1.1rem; font-weight:700; color:white;">
                Annuité constante : {format_int(annuite)} € 
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(
            f"Tableau d'amortissement généré pour un prêt de **{format_int(capital)} €** 💶"
        )

        # ===== Graphique 1 : Capital restant dû =====
        fig_crd = px.line(
            df,
            x="Période",
            y="Capital restant dû (€)",
            title="Évolution du capital restant dû",
        )
        fig_crd.update_layout(xaxis_title="Période", yaxis_title="Capital restant dû (€)")
        st.plotly_chart(fig_crd, use_container_width=True)

        # ===== Graphique 2 : Intérêts vs Amortissement =====
        fig_cf = go.Figure()
        fig_cf.add_trace(
            go.Bar(
                x=df["Période"],
                y=df["Intérêts (€)"],
                name="Intérêts",
            )
        )
        fig_cf.add_trace(
            go.Bar(
                x=df["Période"],
                y=df["Amortissement (€)"],
                name="Amortissement du capital",
            )
        )
        fig_cf.update_layout(
            barmode="stack",
            title="Décomposition de l'annuité : Intérêts vs Amortissement",
            xaxis_title="Période",
            yaxis_title="Montant par période (€)",
            legend_title="Composantes",
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        # ===== Tableau =====
        st.dataframe(df, use_container_width=True)

        # ===== Export CSV =====
        st.download_button(
            "📥 Télécharger (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="tableau_amortissement_emprunt.csv",
            mime="text/csv"
        )
