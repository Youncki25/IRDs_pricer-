import pandas as pd
from datetime import date

def _periodicite_to_freq(periodicite: str):
    """
    Traduit la périodicité (texte) en :
    - code de fréquence pandas
    - nombre de périodes par an
    """
    if periodicite == "Mensuelle":
        return "MS", 12
    if periodicite == "Trimestrielle":
        return "QS", 4
    return "YS", 1  # Annuelle


def tableau_amortissement_emprunt(
    capital: float,
    date_debut: date,
    date_fin: date,
    periodicite: str = "Mensuelle",
    type_taux: str = "Fixe",   # "Fixe" ou "Variable"
    taux_annuel: float = 4.0,  # taux nominal ou marge sur indice
    indice_ref: str | None = None,
) -> pd.DataFrame:
    """
    Génère un échéancier simple entre date_debut et date_fin.

    - type_taux = "Fixe" :
        -> on applique un taux nominal fixe = taux_annuel (%)
    - type_taux = "Variable" :
        -> on applique un taux = indice_ref (théorique) + marge (taux_annuel)
           (ici on ne met pas de vraie courbe, juste un placeholder constant).

    indice_ref doit être dans :
        ["EURIBOR 1M", "EURIBOR 3M", "EURIBOR 6M", "€STR", "SOFR", "SONIA"]
    """

    freq, n_par_an = _periodicite_to_freq(periodicite)

    dates = pd.date_range(start=date_debut, end=date_fin, freq=freq)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)

    # Pour l'instant, on considère un taux effectif constant par période :
    # - si Fixe : taux_nominal = taux_annuel
    # - si Variable : on pourrait faire (indice + marge), ici on note seulement l'indice.
    taux_par_periode = taux_annuel / 100 / n_par_an

    # Emprunt amortissable à annuités constantes (pour les 2 cas)
    annuite = capital * taux_par_periode / (1 - (1 + taux_par_periode) ** (-n))

    crd = capital
    lignes = []

    for dt_ech in dates:
        interets = crd * taux_par_periode
        amort = annuite - interets
        crd = crd - amort

        ligne = {
            "Date échéance": dt_ech.date(),
            "Annuité": annuite,
            "Intérêts": interets,
            "Amortissement": amort,
            "Capital restant dû": max(crd, 0),
            "Type de taux": type_taux,
        }

        if type_taux == "Variable" and indice_ref is not None:
            ligne["Indice de référence"] = indice_ref

        lignes.append(ligne)

    df = pd.DataFrame(lignes)
    return df
