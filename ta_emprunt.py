import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta


def tableau_amortissement_emprunt(
    capital_initial: float,
    taux_annuel: float,
    date_debut: date,
    duree_annees: int,
    paiements_par_an: int = 12,
):
    """
    Génère un tableau d'amortissement pour un emprunt bancaire à annuités constantes.

    capital_initial : montant emprunté (ex : 1_000_000)
    taux_annuel     : ex : 0.04 pour 4%
    date_debut      : date de début du prêt (date du 1er jour de la 1ère période)
    duree_annees    : durée totale du crédit, en années
    paiements_par_an: 12 pour mensualités, 4 pour trimestrielles, 1 pour annuelles
    """

    taux_periodique = taux_annuel / paiements_par_an
    nb_paiements = duree_annees * paiements_par_an

    # Annuité constante
    mensualite = capital_initial * taux_periodique / (
        1 - (1 + taux_periodique) ** (-nb_paiements)
    )

    data = []
    capital_restant = capital_initial
    date_debut_periode = date_debut

    for periode in range(1, nb_paiements + 1):
        # Fin de période = début + 1/paiements_par_an an
        date_fin_periode = date_debut_periode + relativedelta(
            months=12 // paiements_par_an
        )

        interets = capital_restant * taux_periodique
        amort = mensualite - interets
        capital_restant = max(capital_restant - amort, 0)

        data.append(
            {
                "Période": periode,
                "Début période": date_debut_periode,
                "Fin période": date_fin_periode,
                "Mensualité (€)": mensualite,
                "Intérêts (€)": interets,
                "Amortissement (€)": amort,
                "Capital restant dû (€)": capital_restant,
            }
        )

        date_debut_periode = date_fin_periode

    df = pd.DataFrame(data)
    return df
