import pandas as pd

from carburants.prix import CARBURANTS_ROUTIERS, DOSSIER_INTERMEDIAIRE


def filtrer_releves(prix, stations, ecart_max=0.4):
    """Filtre les relevés un par un ; renvoie ceux gardés et le nombre restant après chaque filtre."""
    prix = prix.astype({"id_station": str, "carburant": str})
    journal = {"relevés bruts": len(prix)}

    prix = prix[prix["carburant"].isin(CARBURANTS_ROUTIERS)].dropna(subset=["prix_ttc", "date_maj"])
    journal["Gazole, E10, SP95, SP98, prix renseigné"] = len(prix)

    prix = prix[prix["prix_ttc"].between(0.5, 3)]
    journal["prix entre 0,5 et 3 €/L"] = len(prix)

    mediane = prix.groupby([prix["date_maj"].dt.floor("D"), "carburant"])["prix_ttc"].transform("median")
    prix = prix[(prix["prix_ttc"] / mediane - 1).abs() <= ecart_max]
    journal[f"écart à la médiane du jour d'au plus {ecart_max:.0%}"] = len(prix)

    actives = stations.loc[stations["active"], "id_station"].astype(str)
    prix = prix[prix["id_station"].isin(actives)]
    journal["stations actives"] = len(prix)
    return prix, pd.Series(journal, name="relevés restants")


def prix_journaliers(prix):
    """Panel station x carburant x jour : dernier prix de la journée, reporté sur les jours sans changement."""
    prix = prix.assign(jour=prix["date_maj"].dt.floor("D")).sort_values("date_maj")
    large = prix.pivot_table(index="jour", columns=["id_station", "carburant"], values="prix_ttc", aggfunc="last")
    large = large.asfreq("D")
    large = large.ffill().where(large.bfill().notna())
    return large.stack(["id_station", "carburant"]).dropna().rename("prix_ttc").reset_index()


def ajouter_variations(panel):
    """Variation du prix d'un jour à l'autre, pour chaque station et carburant."""
    panel = panel.sort_values(["id_station", "carburant", "jour"]).reset_index(drop=True)
    panel["variation"] = panel.groupby(["id_station", "carburant"])["prix_ttc"].diff().round(4)
    panel["changement"] = panel["variation"].fillna(0) != 0
    return panel


def panel_annee(annee, ecart_max=0.4):
    """Construit et enregistre le panel de l'année ; renvoie le panel et le journal des filtres."""
    prix = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"prix_{annee}.parquet")
    stations = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"stations_enseignes_{annee}.parquet")
    gardes, journal = filtrer_releves(prix, stations, ecart_max)
    panel = ajouter_variations(prix_journaliers(gardes))
    panel.to_parquet(DOSSIER_INTERMEDIAIRE / f"panel_{annee}.parquet")
    return panel, journal
