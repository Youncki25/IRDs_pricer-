import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta


def tableau_amortissement_emprunt(
    capital_initial: float,
    taux_annuel: float,
    date_debut: date,
    duree_annees: int,
    paiements_par_an: int = 12,
    type_remboursement: str = "annuite",
):
    """
    Génère un tableau d'amortissement pour un emprunt bancaire.

    type_remboursement :
        - "annuite"                : annuités constantes
        - "amortissement_constant" : amortissement du capital constant
        - "bullet"                 : intérêts périodiques, capital remboursé à la fin
    """

    type_remboursement = type_remboursement.lower()
    if type_remboursement not in {"annuite", "amortissement_constant", "bullet"}:
        raise ValueError("type_remboursement doit être 'annuite', "
                         "'amortissement_constant' ou 'bullet'")

    taux_periodique = taux_annuel / paiements_par_an
    nb_paiements = duree_annees * paiements_par_an

    data = []
    capital_restant = capital_initial
    date_debut_periode = date_debut

    # --- Pré-calculs selon le type ---
    if type_remboursement == "annuite":
        # Annuité constante
        mensualite_constante = capital_initial * taux_periodique / (
            1 - (1 + taux_periodique) ** (-nb_paiements)
        )
    elif type_remboursement == "amortissement_constant":
        amortissement_const = capital_initial / nb_paiements
    else:
        # bullet : pas de pré-calcul particulier
        mensualite_constante = None
        amortissement_const = None

    for periode in range(1, nb_paiements + 1):
        # Fin de période = début + 1/paiements_par_an an
        date_fin_periode = date_debut_periode + relativedelta(
            months=12 // paiements_par_an
        )

        if type_remboursement == "annuite":
            mensualite = mensualite_constante
            interets = capital_restant * taux_periodique
            amort = mensualite - interets

        elif type_remboursement == "amortissement_constant":
            interets = capital_restant * taux_periodique
            amort = amortissement_const
            mensualite = interets + amort

        else:  # bullet
            interets = capital_restant * taux_periodique
            if periode < nb_paiements:
                amort = 0.0
                mensualite = interets
            else:
                amort = capital_restant
                mensualite = interets + amort

        capital_restant = max(capital_restant - amort, 0.0)

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
