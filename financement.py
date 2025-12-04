import streamlit as st
import pandas as pd
from datetime import date, datetime
from math import isnan
import io
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


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


# ------------------ Amortissement (3 méthodes) ------------------ #

def _freq_from_periodicite(periodicite: str):
    if periodicite == "Mensuelle":
        return "MS", 12
    elif periodicite == "Trimestrielle":
        return "QS", 4
    else:
        return "YS", 1


def genere_echeancier_annuite_constante(
    capital: float,
    taux_annuel: float,
    date_debut: date,
    date_fin: date,
    periodicite: str
) -> pd.DataFrame:
    """
    Emprunt à annuités constantes.
    """
    freq, n_par_an = _freq_from_periodicite(periodicite)

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

    return pd.DataFrame(rows)


def genere_echeancier_amort_const(
    capital: float,
    taux_annuel: float,
    date_debut: date,
    date_fin: date,
    periodicite: str
) -> pd.DataFrame:
    """
    Emprunt à amortissement de capital constant :
    - même montant de capital remboursé à chaque échéance,
    - annuité qui diminue dans le temps.
    """
    freq, n_par_an = _freq_from_periodicite(periodicite)

    dates = pd.date_range(start=date_debut, end=date_fin, freq=freq)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)
    r = taux_annuel / 100 / n_par_an  # taux par période

    amort_const = capital / n
    crd = capital
    rows = []

    for dt_ech in dates:
        interets = crd * r
        annuite = interets + amort_const
        crd = crd - amort_const

        rows.append(
            {
                "Date échéance": dt_ech.date(),
                "Annuité": annuite,
                "Intérêts": interets,
                "Amortissement": amort_const,
                "Capital restant dû": max(crd, 0),
            }
        )

    return pd.DataFrame(rows)


def genere_echeancier_bullet(
    capital: float,
    taux_annuel: float,
    date_debut: date,
    date_fin: date,
    periodicite: str
) -> pd.DataFrame:
    """
    Prêt in fine (bullet) :
    - intérêts payés à chaque échéance,
    - capital remboursé en totalité à la dernière échéance.
    """
    freq, n_par_an = _freq_from_periodicite(periodicite)

    dates = pd.date_range(start=date_debut, end=date_fin, freq=freq)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)
    r = taux_annuel / 100 / n_par_an  # taux par période

    crd = capital
    rows = []

    for i, dt_ech in enumerate(dates, start=1):
        interets = crd * r
        if i == n:
            amort = capital
        else:
            amort = 0.0

        annuite = interets + amort
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

    return pd.DataFrame(rows)


# ------------------ Génération PDF ------------------ #

def generer_pdf(
    df: pd.DataFrame,
    capital: float,
    taux_annuel: float,
    type_amort: str,
    periodicite: str,
    date_debut: date,
    date_fin: date,
    frais_dossier: float,
    frais_garantie: float,
    assurance: float,
    taeg_pct: float | None,
) -> bytes:
    """
    Crée un PDF en mémoire avec :
    - en-tête (date, auteur, description)
    - caractéristiques du financement
    - tableau d'amortissement
    - graphiques (capital restant dû + intérêts/amortissement)
    - disclaimer
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    today_str = datetime.today().strftime("%d/%m/%Y")

    # --- En-tête --- #
    titre = f"Échéancier d'emprunt — génération du {today_str}"
    story.append(Paragraph(titre, styles["Title"]))
    story.append(Spacer(1, 12))

    intro = (
        "Document généré automatiquement par l’outil de simulation d’échéanciers "
        "développé par Younes Beldjenna. Cet outil gratuit permet de construire "
        "des tableaux d’amortissement et des graphiques pour différents types de "
        "financements. Il est conçu pour être flexible, pédagogique et simple d’utilisation."
    )
    story.append(Paragraph(intro, styles["BodyText"]))
    story.append(Spacer(1, 12))

    # --- Caractéristiques du financement --- #
    story.append(Paragraph("<b>Caractéristiques du financement</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    data_carac = [
        ["Montant emprunté", format_eur(capital, 0)],
        ["Taux nominal annuel", format_pct(taux_annuel, 2)],
        ["Méthode d'amortissement", type_amort],
        ["Périodicité des échéances", periodicite],
        ["Date de début", date_debut.strftime("%d/%m/%Y")],
        ["Date de fin", date_fin.strftime("%d/%m/%Y")],
        ["Frais de dossier", format_eur(frais_dossier, 2)],
        ["Frais de garantie", format_eur(frais_garantie, 2)],
        ["Assurance (taux annuel)", format_pct(assurance, 2)],
    ]
    if taeg_pct is not None:
        data_carac.append(["TAEG (indicatif)", format_pct(taeg_pct, 2)])

    table_carac = Table(data_carac, hAlign="LEFT", colWidths=[160, 260])
    table_carac.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(table_carac)
    story.append(Spacer(1, 12))

    # --- Tableau d'amortissement --- #
    story.append(Paragraph("<b>Tableau d’amortissement</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    df_pdf = df.copy()
    df_pdf["Date échéance"] = df_pdf["Date échéance"].apply(
        lambda d: d.strftime("%d/%m/%Y") if isinstance(d, (date, datetime)) else str(d)
    )
    df_pdf["Annuité"] = df_pdf["Annuité"].round(2)
    df_pdf["Intérêts"] = df_pdf["Intérêts"].round(2)
    df_pdf["Amortissement"] = df_pdf["Amortissement"].round(2)
    df_pdf["Capital restant dû"] = df_pdf["Capital restant dû"].round(2)
    if "Assurance" in df_pdf.columns:
        df_pdf["Assurance"] = df_pdf["Assurance"].round(2)
    if "Flux total (sortie)" in df_pdf.columns:
        df_pdf["Flux total (sortie)"] = df_pdf["Flux total (sortie)"].round(2)

    cols = [
        "Date échéance",
        "Annuité",
        "Intérêts",
        "Amortissement",
        "Capital restant dû",
    ]
    if "Assurance" in df_pdf.columns:
        cols.append("Assurance")
    if "Flux total (sortie)" in df_pdf.columns:
        cols.append("Flux total (sortie)")

    data_tab = [cols] + df_pdf[cols].values.tolist()

    table_ech = Table(data_tab, repeatRows=1)
    table_ech.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(table_ech)
    story.append(Spacer(1, 12))

    # --- Graphiques (via matplotlib) --- #
    story.append(Paragraph("<b>Graphiques</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    # Capital restant dû
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(df["Date échéance"], df["Capital restant dû"])
    ax1.set_title("Évolution du capital restant dû")
    ax1.set_xlabel("Date d’échéance")
    ax1.set_ylabel("Capital restant dû")
    ax1.grid(True)
    fig1.autofmt_xdate()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches="tight")
    plt.close(fig1)
    buf1.seek(0)
    story.append(RLImage(buf1, width=500, height=250))
    story.append(Spacer(1, 12))

    # Intérêts / amortissement
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.stackplot(
        df["Date échéance"],
        df["Intérêts"],
        df["Amortissement"],
        labels=["Intérêts", "Amortissement"],
    )
    ax2.set_title("Décomposition des échéances")
    ax2.set_xlabel("Date d’échéance")
    ax2.set_ylabel("Montant")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    fig2.autofmt_xdate()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    plt.close(fig2)
    buf2.seek(0)
    story.append(RLImage(buf2, width=500, height=250))
    story.append(Spacer(1, 12))

    # --- Disclaimer --- #
    disclaimer = (
        "Ce document est généré par un outil gratuit mis à disposition à titre purement "
        "informatif. Aucune garantie n’est donnée quant à l’exactitude des calculs ni à "
        "leur adéquation avec votre situation personnelle. L’auteur de l’outil ne peut "
        "en aucun cas être tenu responsable des décisions ou des conséquences résultant "
        "de l’utilisation de ce document. L’auteur n’est pas rémunéré pour ce service et "
        "ne fournit pas de conseil financier personnalisé."
    )
    story.append(Paragraph("<b>Disclaimer</b>", styles["Heading3"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(disclaimer, styles["BodyText"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ------------------ Page Streamlit ------------------ #

def render():
    st.header("📄 Tableau d'amortissement — Emprunt bancaire")

    st.markdown(
        """
        Cette page permet de simuler un **financement** et de calculer :
        - un **échéancier d’emprunt** pour la durée de ton choix (1 an, 5 ans, 30 ans, etc.),
        - le **TAEG** (taux annuel effectif global),
        - des **graphes** de capital restant dû et de décomposition intérêts / amortissement,
        - et un **PDF exportable** prêt à être partagé.
        """
    )

    col_g, col_d = st.columns(2)

    # --------- Paramètres à gauche --------- #
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
            help="Tu peux choisir n'importe quelle date (1 an, 10 ans, 25 ans...).",
        )

        periodicite = st.selectbox(
            "Périodicité des échéances",
            ["Mensuelle", "Trimestrielle", "Annuelle"],
            index=0,
        )

        type_amort = st.selectbox(
            "Méthode d'amortissement",
            ["Annuités constantes", "Amortissement constant", "In fine (bullet)"],
            index=0,
            help=(
                "• Annuités constantes : mensualité fixe\n"
                "• Amortissement constant : capital remboursé de façon constante\n"
                "• In fine : capital remboursé en une fois à la fin"
            ),
        )

    # --------- Paramètres à droite --------- #
    with col_d:
        type_taux = st.radio(
            "Type de taux",
            ["Taux fixe"],  # on garde simple pour l’instant
            horizontal=True,
        )

        taux_annuel = st.number_input(
            "Taux nominal annuel (%)",
            value=4.00,
            step=0.10,
            format="%.2f",
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
            help="Assurance éventuelle à intégrer dans le TAEG.",
        )

    st.markdown("---")

    # --------- Lancement du calcul --------- #
    if not st.button("📊 Calculer le tableau d'amortissement"):
        return

    if date_fin <= date_debut:
        st.error("La date de fin doit être **postérieure** à la date de début.")
        return

    # --- Génération de l'échéancier selon la méthode choisie --- #

    if type_amort == "Annuités constantes":
        df = genere_echeancier_annuite_constante(
            capital=capital,
            taux_annuel=taux_annuel,
            date_debut=date_debut,
            date_fin=date_fin,
            periodicite=periodicite,
        )
    elif type_amort == "Amortissement constant":
        df = genere_echeancier_amort_const(
            capital=capital,
            taux_annuel=taux_annuel,
            date_debut=date_debut,
            date_fin=date_fin,
            periodicite=periodicite,
        )
    else:  # In fine (bullet)
        df = genere_echeancier_bullet(
            capital=capital,
            taux_annuel=taux_annuel,
            date_debut=date_debut,
            date_fin=date_fin,
            periodicite=periodicite,
        )

    if df.empty:
        st.warning("Aucune échéance générée : vérifie la période et la périodicité.")
        return

    # --- Ajout de l'assurance --- #
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

    # ------------------ KPIs ------------------ #

    annuite_moy = df["Annuité"].mean()
    duree_annees = (date_fin - date_debut).days / 365.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Montant emprunté",
            value=format_eur(capital, decimals=0),
        )

    with col2:
        st.metric(
            "Annuité moyenne",
            value=format_eur(annuite_moy, decimals=2),
        )

    with col3:
        st.metric(
            "Durée du prêt",
            value=f"{duree_annees:.1f} ans",
        )

    st.caption(
        f"Période du prêt : {date_debut.strftime('%d/%m/%Y')} → {date_fin.strftime('%d/%m/%Y')}"
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

    st.markdown("### 📌 TAEG (Taux Annuel Effectif Global)")

    cashflows = []
    d0 = date_debut
    montant_net_recu = capital - frais_dossier - frais_garantie
    cashflows.append((d0, montant_net_recu))

    for _, row in df.iterrows():
        d_ech = row["Date échéance"]
        if isinstance(d_ech, datetime):
            d_ech = d_ech.date()
        cf = -float(row["Flux total (sortie)"])
        cashflows.append((d_ech, cf))

    taeg_decimal = calcul_taeg(cashflows)
    taeg_pct = None
    if taeg_decimal is not None:
        taeg_pct = taeg_decimal * 100
        st.success(
            f"**TAEG : {format_pct(taeg_pct, 2)}**\n\n"
            "Le TAEG permet de comparer plusieurs financements de manière homogène "
            "en intégrant intérêts, frais et assurance éventuelle."
        )
    else:
        st.warning("Impossible de calculer un TAEG (pas de solution numérique trouvée).")

    # ------------------ Graphiques ------------------ #

    st.markdown("### 📈 Graphiques")

    tab1, tab2 = st.tabs(["Capital restant dû", "Intérêts vs Amortissement"])

    with tab1:
        crd_chart = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Date échéance": [date_debut],
                        "Capital restant dû": [capital],
                    }
                ),
                df[["Date échéance", "Capital restant dû"]],
            ],
            ignore_index=True,
        ).sort_values("Date échéance")

        st.line_chart(
            crd_chart.set_index("Date échéance")["Capital restant dû"],
            use_container_width=True,
        )

    with tab2:
        chart_df = df[["Date échéance", "Intérêts", "Amortissement"]].set_index(
            "Date échéance"
        )
        st.area_chart(chart_df, use_container_width=True)

    st.markdown(
        """
        **Lecture rapide :**
        - La courbe de *capital restant dû* part du niveau initial et décroît jusqu'à 0
          (ou reste constante puis tombe à 0 pour un prêt in fine).
        - Le graphique *Intérêts vs Amortissement* montre la part d'intérêts et de capital
          dans chaque échéance.
        """
    )

    # ------------------ Export PDF ------------------ #

    st.markdown("### 📄 Export PDF")

    pdf_bytes = generer_pdf(
        df=df,
        capital=capital,
        taux_annuel=taux_annuel,
        type_amort=type_amort,
        periodicite=periodicite,
        date_debut=date_debut,
        date_fin=date_fin,
        frais_dossier=frais_dossier,
        frais_garantie=frais_garantie,
        assurance=assurance,
        taeg_pct=taeg_pct,
    )

    st.download_button(
        label="📥 Télécharger le PDF de l’échéancier",
        data=pdf_bytes,
        file_name="echeancier_emprunt_younes_beldjenna.pdf",
        mime="application/pdf",
    )
