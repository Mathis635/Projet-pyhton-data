import zipfile
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_BRUT = RACINE / "data" / "raw"
DOSSIER_INTERMEDIAIRE = RACINE / "data" / "interim"
CARBURANTS_ROUTIERS = ["Gazole", "E10", "SP95", "SP98"]


def telecharger_xml(annee):
    """Télécharge l'archive annuelle si besoin et renvoie le chemin du XML."""
    chemin_zip = DOSSIER_BRUT / f"prix_{annee}.zip"
    if not chemin_zip.exists():
        DOSSIER_BRUT.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://donnees.roulez-eco.fr/opendata/annee/{annee}", timeout=120)
        r.raise_for_status()
        chemin_zip.write_bytes(r.content)
    with zipfile.ZipFile(chemin_zip) as z:
        nom = z.namelist()[0]
        z.extract(nom, DOSSIER_BRUT)
    return DOSSIER_BRUT / nom


def lire_xml(chemin_xml):
    """Lit le XML station par station et renvoie la table des stations et celle des prix."""
    stations, prix = [], []
    for _, pdv in etree.iterparse(str(chemin_xml), tag="pdv"):
        stations.append([pdv.get("id"), pdv.get("latitude"), pdv.get("longitude"), pdv.get("cp"),
                         pdv.get("pop"), pdv.findtext("ville"), pdv.findtext("adresse")])
        for p in pdv.iterfind("prix"):
            prix.append([pdv.get("id"), p.get("nom"), p.get("maj"), p.get("valeur")])
        pdv.clear()

    stations = pd.DataFrame(stations, columns=["id_station", "latitude", "longitude", "cp", "pop", "ville", "adresse"])
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce") / 100_000
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce") / 100_000

    prix = pd.DataFrame(prix, columns=["id_station", "carburant", "date_maj", "prix_ttc"])
    prix["prix_ttc"] = pd.to_numeric(prix["prix_ttc"], errors="coerce")
    prix.loc[prix["prix_ttc"] > 10, "prix_ttc"] /= 1000
    prix["date_maj"] = pd.to_datetime(prix["date_maj"], format="ISO8601", errors="coerce")
    return stations.drop_duplicates("id_station"), prix


def prix_annee(annee):
    """Télécharge, lit et enregistre en Parquet les stations et les prix d'une année."""
    stations, prix = lire_xml(telecharger_xml(annee))
    DOSSIER_INTERMEDIAIRE.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(DOSSIER_INTERMEDIAIRE / f"stations_{annee}.parquet")
    prix.to_parquet(DOSSIER_INTERMEDIAIRE / f"prix_{annee}.parquet")


def activite_stations(prix, seuil=50):
    """Nombre de relevés par station et carburant routier ; active à partir de `seuil` relevés."""
    n = pd.crosstab(prix["id_station"].astype(str), prix["carburant"])
    n = n.reindex(columns=CARBURANTS_ROUTIERS, fill_value=0).add_prefix("releves_")
    n["active"] = n.sum(axis=1) >= seuil
    return n
