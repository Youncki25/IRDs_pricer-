import streamlit as st

def render():
    st.markdown(
        """
        <h1 style="margin-top: -30px;">Bienvenue sur Desk Taux</h1>
        <p>
        Plateforme de pricing IRS / IRD, d'analyse de courbes de taux, 
        de construction d'échéanciers et d'extraction macroéconomique.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🧾 À propos du projet")
    st.markdown(
        """
        Cette application explore :
        - le pricing de swaps (IRS), IRDs, obligations  
        - la construction de courbes (OIS, swaps, STIRs)  
        - l’analyse des marchés via plusieurs API (FRED, ECB, Alpha Vantage, Quandl…)  
        - la reproduction de courbes *trading-floor-like*  
        - outil basique de financement avec tableau d'amortissement
        - la visualisation de données macroéconomiques et financières
        - la création de graphiques financiers interactifs
        
        """
    )

    st.subheader("🧭 Rubriques")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Pricer**  
            • IRS, STIR, OIS  
            • Obligations  
            • Tableaux d’amortissement  
            """
        )

    with col2:
        st.markdown(
            """
            **Macro / Graphiques**  
            • Courbes FRED  
            • Interpolations (zéro, log-DF, PCHIP)  
            • Données macro US / Europe  
            """
        )
