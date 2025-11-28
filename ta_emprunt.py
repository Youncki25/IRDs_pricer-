import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

def tableau_amortissement_emprunt(
    capital_initial: float,
    taux_annuel: float,
    date_debut: date,
    duree_annees: int,
    paiements_par_an: int = 12
):
    """
    Génère un tableau d'amortissement bancaire classique :
    - mensualités constantes
    - dates de début / fin période
    - capital restant dû
    """

    taux_periodique = taux_annuel / paiements_par_an
    nb_paiements = duree_annees * paiements_par_an

    # Mensualité constante
    mensualite = capital_initial * taux_periodique / (1 - (1 + taux_periodique) ** (-nb_paiements))

    # Amortissement
    data = []
    capital_restant = capital_initial
    date_debut_periode = date_debut

    for periode in range(1, nb_paiements + 1):

        # Fin période = début période + 1/n an
        date_fin_periode = date_debut_periode + relativedelta(months=12 // paiements_par_an)

        interets = capital_restant * taux_periodique
        amort = mensualite - interets
        capital_restant = max(capital_restant - amort, 0)

        data.append({
            "Période": periode,
            "Début période": date_debut_periode,
            "Fin période": date_fin_periode,
            "Mensualité (€)": mensualite,
            "Intérêts (€)": interets,
            "Amortissement (€)": amort,
            "Capital restant dû (€)": capital_restant
        })

        date_debut_periode = date_fin_periode

    df = pd.DataFrame(data)
    return df
