import streamlit as st
import pandas as pd
from datetime import date, datetime
from math import isnan

# ------------------ Helpers format FR ------------------ #

def format_int(n: float) -> str:
    """Affiche 1 000 000 au lieu de 1000000."""
    return f"{int(round(n)):,}".replace(",", " ")

def format_eur(n: float, decimals: int = 2) -> str:
    """
    Format EUR style FR : 1 234 567,89 €
    """
    if n is None:
        return ""
    fmt = f"{{:,.{decimals}f}}"
    s = fmt.format(float(n)).replace(",", " ").replace(".", ",")
    return s + " €"

def format_pct(n: float, decimals: int = 2) -> str:
    if n is None:
        return ""
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(float(n)) + " %"


# ------------------ Calcul TAEG ------------------ #

def calcul_taeg(cashflows):
    """
    cashflows = liste de tuples (date, montant)
    montant < 0 : sortie de cash (décaissement pour l'emprunteur)
    montant > 0 : encaissement (versements du prêt, etc.)

    On calcule un taux r tel que :
        somme_i CF_i / (1 + r)^(t_i) = 0
    avec t_i en années (ACT/365) par rapport au premier flux.
    """

    if not cashflows or len(cashflows) < 2:
        return None

    d0 = cashflows[0][0]

    def npv(r):
        total = 0.0
        for d, cf in cashflows:
            t = (d - d0).days / 365.0
            total += cf / ((1 + r) ** t)
        return total

    def d_npv(r):
        total = 0.0
        for d, cf in cashflows:
            t = (d - d0).days / 365.0
            if t == 0:
                continue
            total += -t * cf / ((1 + r) ** (t + 1))
        return total

    # Newton simple
    r = 0.03  # 3 % comme point de départ
    for _ in range(50):
        f = npv(r)
        df = d_npv(r)
        if df == 0:
            break
        r_new = r - f / df
        if abs(r_new - r) < 1e-8:
            r = r_new
            break
        r = r_new

    if r < -0.9999 or isnan(r):
        return None

    return r  # en décimal (ex: 0.025 = 2,5 %)


# ------------------ Amortissement ------------------ #

def genere_echeancier_fixe_annuite(
    capital: float,
    taux_annuel: float,
    date_debut: date,
    date_fin: date,
    periodicite: str
) -> pd.DataFrame:
    """
    Emprunt classique à annuités constantes avec taux fixe.

    periodicite: 'Mensuelle', 'Trimestrielle', 'Annuelle'
    """
    if periodicite == "Mensuelle":
        freq = "MS"
        n_par_an = 12
    elif periodicite == "Trimestrielle":
        freq = "QS"
        n_par_an = 4
    else:
        freq = "YS"
        n_par_an = 1

    # Génération des dates d'échéance (hors date_debut)
    dates = pd.date_range(start=date_debut, end=date_fin, freq=freq)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)
    r = taux_annuel / 100 / n_par_an  # taux par période

    # Annuité constante
    annuite = capital * r / (1 - (1 + r) ** (-n))

    crd = capital
    rows = []

    for dt_ech in dates:
        interets = crd * r
        amort = annuite - interets
        crd = crd - amort

        rows.append(
            {
                "Date échéance": dt_ech.date(),
                "Annuité": annuite,
                "Intérêts": interets,
                "Amortissement": amort,
                "Capital restant dû": max(crd, 0),
            }
        )

    df = pd.DataFrame(rows)
    return df


def genere_echeancier_variable_const_amort(
    capital: float,
    taux_annuel: float,
    date_debut: date,
    date_fin: date,
    periodicite: str
) -> pd.DataFrame:
    """
    Exemple simple de taux "variable" :
    - amortissement de capital constant
    - taux peut être ajusté plus tard pour chaque période
      (ici on prend un taux constant en entrée pour rester simple).

    Tu pourras ensuite remplacer la logique du taux par une vraie courbe.
    """
    if periodicite == "Mensuelle":
        freq = "MS"
        n_par_an = 12
    elif periodicite == "Trimestrielle":
        freq = "QS"
        n_par_an = 4
    else:
        freq = "YS"
        n_par_an = 1

    dates = pd.date_range(start=date_debut, end=date_fin, freq=freq)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)
    r = taux_annuel / 100 / n_par_an  # taux par période

    amortissement_const = capital / n
    crd = capital
    rows = []

    for dt_ech in dates:
        interets = crd * r
        annuite = interets + amortissement_const
        crd = crd - amortissement_const
        rows.append(
            {
                "Date échéance": dt_ech.date(),
                "Annuité": annuite,
                "Intérêts": interets,
                "Amortissement": amortissement_const,
                "Capital restant dû": max(crd, 0),
            }
        )

    df = pd.DataFrame(rows)
    return df


# ------------------ Page Streamlit ------------------ #

def render():
    st.header("📄 Tableau d'amortissement — Emprunt bancaire")

    st.markdown(
        """
        Cette page permet de simuler un **financement** (taux fixe ou variable),
        d'afficher le **tableau d'amortissement** et de calculer le **TAEG**.
        """
    )

    col_g, col_d = st.columns(2)

    with col_g:
        capital = st.number_input(
            "Montant emprunté (€)",
            value=1_000_000,
            step=50_000,
            format="%d",
            help="Montant du prêt (ex : 1 000 000).",
        )

        date_debut = st.date_input(
            "Date de début du prêt",
            value=date.today(),
        )

        date_fin = st.date_input(
            "Date de fin du prêt",
            value=date(date.today().year + 20, date.today().month, date.today().day),
            help="Date de la dernière échéance (fin du prêt).",
        )

        periodicite = st.selectbox(
            "Périodicité des échéances",
            ["Mensuelle", "Trimestrielle", "Annuelle"],
            index=0,
        )

    with col_d:
        type_taux = st.radio(
            "Type de taux",
            ["Taux fixe", "Taux variable"],
            horizontal=True,
        )

        if type_taux == "Taux fixe":
            taux_annuel = st.number_input(
                "Taux nominal annuel (%)",
                value=4.00,
                step=0.10,
                format="%.2f",
            )
        else:
            taux_annuel = st.number_input(
                "Taux annuel de départ (%)",
                value=4.00,
                step=0.10,
                format="%.2f",
                help="Pour l'instant un seul taux est utilisé pour tout l'échéancier (exemple simple).",
            )

        frais_dossier = st.number_input(
            "Frais de dossier (€)",
            value=0.0,
            step=100.0,
            format="%.2f",
        )

        frais_garantie = st.number_input(
            "Frais de garantie (€)",
            value=0.0,
            step=100.0,
            format="%.2f",
        )

        assurance = st.number_input(
            "Assurance (% du capital emprunté par an)",
            value=0.0,
            step=0.10,
            format="%.2f",
            help="Assurance obligatoire incluse dans le TAEG si applicable.",
        )

    st.markdown("---")

    if st.button("📊 Calculer le tableau d'amortissement"):
        if date_fin <= date_debut:
            st.error("La date de fin doit être **postérieure** à la date de début.")
            return

        # --- Génération de l'échéancier ---

        if type_taux == "Taux fixe":
            df = genere_echeancier_fixe_annuite(
                capital=capital,
                taux_annuel=taux_annuel,
                date_debut=date_debut,
                date_fin=date_fin,
                periodicite=periodicite,
            )
        else:
            df = genere_echeancier_variable_const_amort(
                capital=capital,
                taux_annuel=taux_annuel,
                date_debut=date_debut,
                date_fin=date_fin,
                periodicite=periodicite,
            )

        if df.empty:
            st.warning("Aucune échéance générée : vérifie la période et la périodicité.")
            return

        # Ajout de l'assurance (simple : % du capital emprunté / nb échéances)
        if assurance > 0:
            n_ech = len(df)
            cout_assurance_total = capital * (assurance / 100) * (
                (date_fin - date_debut).days / 365.0
            )
            cout_assurance_par_ech = cout_assurance_total / n_ech
            df["Assurance"] = cout_assurance_par_ech
        else:
            df["Assurance"] = 0.0

        df["Flux total (sortie)"] = df["Annuité"] + df["Assurance"]

        # --- Calcul de l'annuité moyenne (pour affichage) ---
        annuite_moy = df["Annuité"].mean()

        # --- Affichage KPIs --- #
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Montant emprunté",
                value=format_eur(capital, decimals=0),
            )

        with col2:
            st.metric(
                "Annuité (moyenne)",
                value=f"**{format_eur(annuite_moy, decimals=2)}**",
            )

        with col3:
            duree_annees = (date_fin - date_debut).days / 365.0
            st.metric(
                "Durée du prêt",
                value=f"{duree_annees:.1f} ans",
            )

        st.markdown("### 🧾 Tableau d'amortissement")

        # Copie formatée pour affichage
        df_aff = df.copy()
        df_aff["Annuité"] = df_aff["Annuité"].apply(lambda x: format_eur(x, 2))
        df_aff["Intérêts"] = df_aff["Intérêts"].apply(lambda x: format_eur(x, 2))
        df_aff["Amortissement"] = df_aff["Amortissement"].apply(lambda x: format_eur(x, 2))
        df_aff["Capital restant dû"] = df_aff["Capital restant dû"].apply(
            lambda x: format_eur(x, 0)
        )
        df_aff["Assurance"] = df_aff["Assurance"].apply(lambda x: format_eur(x, 2))
        df_aff["Flux total (sortie)"] = df_aff["Flux total (sortie)"].apply(
            lambda x: format_eur(x, 2)
        )

        st.dataframe(df_aff, use_container_width=True)

        # ------------------ Calcul du TAEG ------------------ #

        # Flux init : décaissement net pour l'emprunteur = +capital - frais
        # (attention signe : on se place du point de vue de l'emprunteur)
        cashflows = []

        d0 = date_debut
        montant_net_recu = capital - frais_dossier - frais_garantie
        # L'emprunteur reçoit l'argent => flux positif
        cashflows.append((d0, montant_net_recu))

        # Puis il rembourse des annuités + assurance => flux négatifs
        for _, row in df.iterrows():
            d_ech = row["Date échéance"]
            # On force type date
            if isinstance(d_ech, datetime):
                d_ech = d_ech.date()
            cf = -float(row["Flux total (sortie)"])
            cashflows.append((d_ech, cf))

        taeg_decimal = calcul_taeg(cashflows)
        if taeg_decimal is not None:
            taeg_pct = taeg_decimal * 100
            st.markdown("### 📌 TAEG (Taux Annuel Effectif Global)")
            st.success(
                f"**TAEG : {format_pct(taeg_pct, 2)}**\n\n"
                "Le **TAEG** intègre :\n"
                "- les **intérêts** du prêt,\n"
                "- les **frais de dossier** et de **garantie**, \n"
                "- l'**assurance obligatoire** (si renseignée).\n\n"
                "C'est le taux **global** à utiliser pour comparer deux financements "
                "de durées, structures et frais différents."
            )
        else:
            st.warning("Impossible de calculer un TAEG (pas de solution numérique trouvée).")

        # ------------------ Graphiques ------------------ #

        st.markdown("### 📈 Graphiques")

        tab1, tab2 = st.tabs(["Capital restant dû", "Intérêts vs Amortissement"])

        with tab1:
            st.line_chart(
                df.set_index("Date échéance")["Capital restant dû"],
                use_container_width=True,
            )

        with tab2:
            chart_df = df[["Date échéance", "Intérêts", "Amortissement"]].set_index(
                "Date échéance"
            )
            st.area_chart(chart_df, use_container_width=True)

        st.markdown(
            """
            **Interprétation rapide :**
            - La courbe de *capital restant dû* doit décroître jusqu'à 0 à la fin du prêt.
            - Le graphique *Intérêts vs Amortissement* montre comment la composition de l'annuité évolue dans le temps.
            """
        )
