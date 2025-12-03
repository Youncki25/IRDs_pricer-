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
    Emprunt à annuités constantes, taux FIXE.
    """
    taux_periodique = taux_annuel / paiements_par_an
    nb_paiements = duree_annees * paiements_par_an

    mensualite = capital_initial * taux_periodique / (
        1 - (1 + taux_periodique) ** (-nb_paiements)
    )

    data = []
    capital_restant = capital_initial
    date_debut_periode = date_debut

    for periode in range(1, nb_paiements + 1):
        date_fin_periode = date_debut_periode + relativedelta(
            months=12 // paiements_par_an
        )

        interets = capital_restant * taux_periodique
        amort = mensualite - interets
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

    return pd.DataFrame(data)


def tableau_amortissement_emprunt_variable(
    capital_initial: float,
    liste_taux_annuels: list[float],
    date_debut: date,
    duree_annees: int,
    paiements_par_an: int = 12,
):
    """
    Emprunt à annuités recalculées chaque année (taux VARIABLE).

    - liste_taux_annuels : liste de taux annuels (en décimal, ex 0.03 pour 3%)
      année 1 -> taux[0], année 2 -> taux[1], etc.
      Si la liste est plus courte que la durée, on répète le dernier taux.
    """
    nb_paiements_total = duree_annees * paiements_par_an

    # Ajuste la liste des taux à la durée du prêt
    if len(liste_taux_annuels) < duree_annees:
        last = liste_taux_annuels[-1]
        liste_taux_annuels = liste_taux_annuels + [last] * (
            duree_annees - len(liste_taux_annuels)
        )
    elif len(liste_taux_annuels) > duree_annees:
        liste_taux_annuels = liste_taux_annuels[:duree_annees]

    data = []
    capital_restant = capital_initial
    date_debut_periode = date_debut
    periode_global = 0

    for annee in range(duree_annees):
        taux_annuel = liste_taux_annuels[annee]
        taux_periodique = taux_annuel / paiements_par_an

        periodes_restantes = nb_paiements_total - periode_global
        if periodes_restantes <= 0:
            break

        # Annuité recalculée au début de chaque année
        mensualite_annee = capital_restant * taux_periodique / (
            1 - (1 + taux_periodique) ** (-periodes_restantes)
        )

        for _ in range(paiements_par_an):
            if periode_global >= nb_paiements_total:
                break

            periode_global += 1
            date_fin_periode = date_debut_periode + relativedelta(
                months=12 // paiements_par_an
            )

            interets = capital_restant * taux_periodique
            amort = mensualite_annee - interets
            capital_restant = max(capital_restant - amort, 0.0)

            data.append(
                {
                    "Période": periode_global,
                    "Début période": date_debut_periode,
                    "Fin période": date_fin_periode,
                    "Mensualité (€)": mensualite_annee,
                    "Intérêts (€)": interets,
                    "Amortissement (€)": amort,
                    "Capital restant dû (€)": capital_restant,
                }
            )

            date_debut_periode = date_fin_periode

    return pd.DataFrame(data)
