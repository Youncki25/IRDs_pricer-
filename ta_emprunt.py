import pandas as pd

def tableau_amortissement_emprunt(
    capital_initial: float,
    taux_annuel: float,
    duree_annees: int,
    paiements_par_an: int = 12
):
    """
    Génère un tableau d'amortissement pour un emprunt bancaire classique à mensualité constante.
    
    capital_initial : montant emprunté (ex : 1_000_000)
    taux_annuel : ex : 0.04 pour 4%
    duree_annees : durée totale du crédit
    paiements_par_an : 12 pour mensualités, 4 pour trimestrielles...
    """

    taux_periodique = taux_annuel / paiements_par_an
    nb_paiements = duree_annees * paiements_par_an

    # Mensualité constante
    mensualite = capital_initial * taux_periodique / (1 - (1 + taux_periodique) ** (-nb_paiements))

    # Tableau
    data = []
    capital_restant = capital_initial

    for periode in range(1, nb_paiements + 1):
        interets = capital_restant * taux_periodique
        amort = mensualite - interets
        capital_restant = max(capital_restant - amort, 0)

        data.append({
            "Période": periode,
            "Mensualité (€)": mensualite,
            "Intérêts (€)": interets,
            "Amortissement (€)": amort,
            "Capital restant dû (€)": capital_restant
        })

    df = pd.DataFrame(data)
    return df
