import streamlit as st
from datetime import date
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ta_emprunt import tableau_amortissement_emprunt


# ---------- Helpers format FR ----------

def format_int(n: float) -> str:
    """Affiche 1 000 000 au lieu de 1000000."""
    return f"{int(round(n)):,}".replace(",", " ")


def format_eur(n: float, decimals: int = 2) -> str:
    """
    Format EUR style FR : 1 234 567,89
    """
    fmt = f"{{:,.{decimals}f}}".format(n)
    return fmt.replace(",", " ").replace(".", ",")


# ---------- Page Streamlit ----------

def render():
    st.header("📄 Tableau d'amortissement — Emprunt bancaire")

    # ---- Inputs ----
    capital = st.number_input(
        "Montant emprunté",
        value=1_000_000,
        step=50_000,
        format="%d",
        help="Montant du prêt (ex : 1 000 000 €).",
    )

    taux_pct = st.number_input(
        "Taux annuel (%)",
        value=4.0,
        step=0.1,
        format="%.2f",
        help="Ex : 4 pour 4%.",
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
        format_func=lambda x: {12: "Mensuel (12)", 4: "Trimestriel (4)", 1: "Annuel (1)"}[x],
    )

    # 🔹 Nouveau : frais initiaux pour le TAEG
    frais_initiaux = st.number_input(
        "Frais initiaux (en €)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help="Frais payés au début du prêt (frais de dossier, assurance, etc.). "
             "Utilisés pour le calcul du TAEG.",
    )

    # ---- Calcul / affichage du cas principal ----
    if st.button("Générer le tableau", type="primary"):

        taux = taux_pct / 100.0

        df = tableau_amortissement_emprunt(
            capital_initial=capital,
            taux_annuel=taux,
            date_debut=date_debut,
            duree_annees=int(duree),
            paiements_par_an=int(freq),
        )

        # Annuité constante (1ère mensualité)
        annuite = df["Mensualité (€)"].iloc[0]
        st.markdown(
            f"""
            <div style="margin-top:0.5rem; margin-bottom:0.8rem;
                        font-size:1.1rem; font-weight:700; color:white;">
                Annuité constante : {format_eur(annuite)} €
            </div>
            """,
            unsafe_allow_html=True,
        )

        cout_interets_total = df["Intérêts (€)"].sum()
        st.success(
            f"Tableau d'amortissement généré pour un prêt de **{format_eur(capital, 0)} €** "
            f"— coût total des intérêts : **{format_eur(cout_interets_total)} €** 💶"
        )

        # 🔹 🔹 TAEG (approximation par IRR) 🔹 🔹
        cash_flows = [capital - frais_initiaux] + [-x for x in df["Mensualité (€)"]]
        try:
            irr_periodique = np.irr(cash_flows)
        except Exception:
            irr_periodique = None

        if irr_periodique is not None and not np.isnan(irr_periodique):
            taeg = (1 + irr_periodique) ** freq - 1
            taeg_str = f"{taeg * 100:.2f}".replace(".", ",")

            st.info(f"**TAEG (approx.) : {taeg_str} %** (incluant les frais initiaux saisis).")

            st.markdown(
                """
                ### ℹ️ TAEG : c’est quoi et à quoi ça sert ?

                **TAEG** = *Taux Annuel Effectif Global*.

                - C’est le **coût total et réel de votre crédit**, exprimé en **taux annuel**.
                - Il inclut :
                  - le **taux d’intérêt nominal**,
                  - les **frais de dossier**,
                  - les **frais d’assurance obligatoires**,
                  - les **frais de garantie** (hypothèque, caution…),
                  - et tous les frais **obligatoires** pour obtenir le prêt.

                👉 Le TAEG sert à :
                - **Comparer plusieurs offres de crédit entre elles** :  
                  une banque peut afficher un taux nominal bas mais un TAEG plus élevé à cause des frais.
                - Donner une **vision standardisée et transparente** du coût de votre financement :  
                  la publication du TAEG est **obligatoire** pour les établissements prêteurs.

                > En résumé : le TAEG vous indique **combien votre financement vous coûte vraiment**, par an,
                > une fois tous les frais intégrés.
                """
            )

        # ===== Graphique 1 : Capital restant dû =====
        fig_crd = px.line(
            df,
            x="Période",
            y="Capital restant dû (€)",
            title="Évolution du capital restant dû",
        )
        fig_crd.update_layout(
            xaxis_title="Période",
            yaxis_title="Capital restant dû (€)",
        )
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

        # ===== Tableau formaté (FR) =====
        df_formatted = df.copy()

        for col in [
            "Mensualité (€)",
            "Intérêts (€)",
            "Amortissement (€)",
            "Capital restant dû (€)",
        ]:
            df_formatted[col] = df_formatted[col].apply(lambda x: format_eur(x))

        st.dataframe(df_formatted, use_container_width=True)

        # ===== Export CSV (données brutes) =====
        st.download_button(
            "📥 Télécharger (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="tableau_amortissement_emprunt.csv",
            mime="text/csv",
        )

        # ===== COMPARAISON DE DIFFÉRENTS TAUX D'INTÉRÊT =====
        st.markdown("---")
        st.subheader("📊 Comparer plusieurs financements selon le taux d'intérêt")

        taux_saisie = st.text_input(
            "Taux à comparer (%) (séparés par des virgules)",
            value="2.0, 3.0, 4.0, 5.0",
            help="Exemple : 2, 3.5, 4, 5.25",
        )

        if st.button("Tracer la comparaison des taux"):
            # Parse la liste de taux
            try:
                liste_taux = [
                    float(t.strip().replace(",", "."))
                    for t in taux_saisie.split(",")
                    if t.strip() != ""
                ]
            except ValueError:
                st.error("Format des taux invalide. Exemple : 2, 3.5, 4, 5.25")
                return

            if not liste_taux:
                st.warning("Merci de saisir au moins un taux.")
                return

            fig_comp = go.Figure()
            resume = []

            for t_pct in liste_taux:
                t_decimal = t_pct / 100.0
                df_t = tableau_amortissement_emprunt(
                    capital_initial=capital,
                    taux_annuel=t_decimal,
                    date_debut=date_debut,
                    duree_annees=int(duree),
                    paiements_par_an=int(freq),
                )

                fig_comp.add_trace(
                    go.Scatter(
                        x=df_t["Période"],
                        y=df_t["Capital restant dû (€)"],
                        mode="lines",
                        name=f"{t_pct:.2f} %",
                    )
                )

                cout_int = df_t["Intérêts (€)"].sum()
                annuite_t = df_t["Mensualité (€)"].iloc[0]
                resume.append(
                    {
                        "Taux (%)": f"{t_pct:.2f}",
                        "Annuité (€)": format_eur(annuite_t),
                        "Coût total intérêts (€)": format_eur(cout_int),
                    }
                )

            fig_comp.update_layout(
                title="Comparaison des capitaux restants dus selon différents taux",
                xaxis_title="Période",
                yaxis_title="Capital restant dû (€)",
                legend_title="Taux",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("**Résumé des coûts par taux :**")
            st.table(resume)
