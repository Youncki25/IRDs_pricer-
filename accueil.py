import streamlit as st
from author_card import show_author_card


def render():
    # HERO
    st.markdown(
        """
        <div class="home-hero">
            <div class="home-hero-card">
                <div class="home-hero-badge">Desk Taux • Outils quantitatifs</div>
                <h1>Bienvenue !</h1>
                <p class="home-hero-subtitle">
                    Plateforme de trading et de gestion des risques pour les produits de taux :
                    swaps, obligations, courbes de taux et tableaux d’amortissement, avec une
                    approche la plus proche possible des pratiques des <em>trading floors</em>.
                </p>
                <div class="home-hero-tags">
                    <span>📊 Pricing de swaps & STIR</span>
                    <span>🏦 Obligations à taux fixe</span>
                    <span>📅 Tableaux d’amortissement</span>
                    <span>🌎 Macro & courbes globales</span>
                </div>
            </div>
        </div>

        <style>
        .home-hero {
            width: 100%;
            min-height: 60vh;
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }
        .home-hero-card {
            max-width: 720px;
            margin-left: 3rem;
            padding: 2.4rem 2.8rem;
            border-radius: 24px;
            background: rgba(5, 5, 15, 0.78);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 18px 40px rgba(0,0,0,0.45);
            color: #f5f5f5;
        }
        .home-hero-badge {
            display: inline-block;
            padding: 0.20rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            letter-spacing: .08em;
            text-transform: uppercase;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            margin-bottom: 0.75rem;
        }
        .home-hero-card h1 {
            font-size: 2.4rem;
            margin: 0 0 0.4rem 0;
        }
        .home-hero-subtitle {
            font-size: 0.98rem;
            line-height: 1.6;
            margin-bottom: 1.4rem;
            color: #e3e3e3;
        }
        .home-hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .home-hero-tags span {
            font-size: 0.82rem;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.16);
        }
        @media (max-width: 900px) {
            .home-hero-card {
                margin: 1.5rem 0.5rem 0 0.5rem;
                padding: 1.8rem 1.4rem;
            }
            .home-hero-card h1 {
                font-size: 1.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # === A PROPOS DU PROJET ===
    st.markdown("### 🧾 À propos du projet")

    st.markdown(
        """
        Cette application a été développée pour **explorer la tarification et la gestion des risques
        sur les produits de taux** (IRS, IRDs, obligations à taux fixe) en s’appuyant sur Python,
        QuantLib et des **données de marché gratuites ou quasi temps réel**.

        L’idée centrale est :

        - de **faire des prix sur des IRS et IRDs** (swaps de taux, STIR, produits dérivés de taux) ;
        - de **construire des courbes de taux** (OIS, swap, souveraines) aussi proches que possible
          de celles utilisées sur les *trading floors* :
          stripping, intégration des STIRs (SR3, SOFR), interpolation avancée, gestion des calendriers, stubs, etc.

        Dans la partie **macroéconomie**, l’application s’appuie sur plusieurs API afin de proposer
        une **overview de marché dans le monde** :
        **FRED**, **ECB Statistical Data Warehouse**, éventuellement **Banque de France**, **Alpha Vantage**,
        **Quandl / Nasdaq Data Link**, et potentiellement **Refinitiv Eikon** lorsqu’un accès est disponible.
        """
    )

    # === STRUCTURE / RUBRIQUES ===
    st.markdown("### 🧭 Structure de l’application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Accueil**  
            Présentation générale de l’outil, bandeau SOFR, contexte du projet et rappel des objectifs.

            **Pricer**  
            - Construction de **tableaux d’amortissement** complets (calendriers, stubs, payment lag…).  
            - Pricing de produits de taux :  
              • swaps plain vanilla  
              • STIR sur RFR (SOFR, €STR, SONIA) via SR3 & consorts  
              • forwards, basis swaps  
              • obligations vanille (pricing analytique).
            """
        )

    with col2:
        st.markdown(
            """
            **Graphique**  
            - Visualisation des **tableaux d’amortissement** générés.  
            - Données macro / courbes de taux (par ex. **courbe US Trésor via FRED**)
              avec différentes méthodes d’interpolation (zéro-continu, log-DF, PCHIP…).  

            **Obligations**  
            - Pricing complet d’**obligations à taux fixe** via un objet `FixedRateBond`.  
            - Calcul des sensibilités (Duration de Macaulay, durée modifiée, convexité, DV01).  
            - Génération d’échéanciers détaillés exportables en CSV.
            """
        )

    st.markdown("---")

    # Crédit / mémoire + carte auteur
    st.markdown(
        """
        <div style="font-size:0.85rem; opacity:0.8;">
        Application réalisée par <strong>Younes Beldjenna</strong> dans le cadre de son mémoire de Master
        en Banque &amp; Finance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_author_card()
