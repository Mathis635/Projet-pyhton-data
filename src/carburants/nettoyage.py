"""Nettoyage des prix et construction du panel station x carburant x jour.

Chaîne : relevés bruts -> filtres documentés (carburant, prix aberrants, stations
actives) -> prix de fin de journée -> report du dernier prix connu sur les jours
sans changement -> variations journalières.
"""

import pandas as pd

from carburants.prix import CARBURANTS_ROUTIERS, RACINE

DOSSIER_INTERMEDIAIRE = RACINE / "data" / "interim"


def filtrer_releves(
    prix: pd.DataFrame, stations: pd.DataFrame, borne_min: float = 0.5, borne_max: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique les filtres un par un et renvoie (relevés gardés, journal des filtres).

    Le journal compte les lignes retirées à chaque étape : c'est lui qui documente
    le nettoyage dans le rapport.
    """
    journal = [("Relevés bruts", len(prix))]
    prix = prix.assign(id_station=prix["id_station"].astype(str))

    prix = prix[prix["carburant"].isin(CARBURANTS_ROUTIERS)]
    journal.append(("Carburants routiers (Gazole, E10, SP95, SP98)", len(prix)))

    prix = prix.dropna(subset=["prix_ttc", "date_maj"])
    journal.append(("Prix et date renseignés", len(prix)))

    prix = prix[prix["prix_ttc"].between(borne_min, borne_max)]
    journal.append((f"Prix entre {borne_min} et {borne_max} €/L", len(prix)))

    actives = set(stations.loc[stations["active"], "id_station"].astype(str))
    prix = prix[prix["id_station"].isin(actives)]
    journal.append(("Stations actives", len(prix)))

    journal = pd.DataFrame(journal, columns=["étape", "relevés restants"])
    journal["relevés retirés"] = -journal["relevés restants"].diff().fillna(0).astype(int)
    prix = prix.assign(carburant=prix["carburant"].astype(str))
    return prix, journal


def prix_journaliers(prix: pd.DataFrame) -> pd.DataFrame:
    """Panel station x carburant x jour, au prix de fin de journée.

    Un jour sans changement reprend le dernier prix connu : le prix affiché reste
    en vigueur tant qu'il n'est pas modifié. Le report s'arrête au dernier relevé
    de la série, pour ne pas prolonger une station qui a cessé de publier.
    """
    fin_de_jour = (
        prix.assign(jour=prix["date_maj"].dt.floor("D"))
        .sort_values("date_maj")
        .groupby(["jour", "id_station", "carburant"])["prix_ttc"]
        .last()
        .unstack(["id_station", "carburant"])  # une colonne par série station x carburant
    )
    jours = pd.date_range(fin_de_jour.index.min(), fin_de_jour.index.max(), freq="D")
    fin_de_jour = fin_de_jour.reindex(jours)
    reporte = fin_de_jour.ffill().where(fin_de_jour.bfill().notna())  # rien après le dernier relevé
    panel = reporte.rename_axis("jour").stack(["id_station", "carburant"])
    return panel.dropna().rename("prix_ttc").reset_index()


def ajouter_variations(panel: pd.DataFrame) -> pd.DataFrame:
    """Variation journalière du prix de chaque série ; changement = variation non nulle."""
    panel = panel.sort_values(["id_station", "carburant", "jour"])
    panel["variation"] = panel.groupby(["id_station", "carburant"])["prix_ttc"].diff().round(4)
    panel["changement"] = panel["variation"].fillna(0).ne(0)
    return panel.reset_index(drop=True)


def panel_annee(annee: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chaîne complète : écrit data/interim/panel_{annee}.parquet, renvoie (panel, journal)."""
    prix = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"prix_{annee}.parquet")
    stations = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"stations_enseignes_{annee}.parquet")
    gardes, journal = filtrer_releves(prix, stations)
    panel = ajouter_variations(prix_journaliers(gardes))
    panel.to_parquet(DOSSIER_INTERMEDIAIRE / f"panel_{annee}.parquet")
    return panel, journal
