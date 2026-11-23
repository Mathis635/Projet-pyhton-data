"""Collecte des prix à la pompe : archives annuelles XML de prix-carburants.gouv.fr.

Chaîne : téléchargement du zip -> extraction du XML -> lecture en flux
-> deux tables (stations, prix) -> Parquet.
"""

from pathlib import Path
import zipfile

import pandas as pd
import requests
from lxml import etree

URL_ANNEE = "https://donnees.roulez-eco.fr/opendata/annee/{annee}"
RACINE = Path(__file__).resolve().parents[2]  # prix.py -> carburants -> src -> racine
DOSSIER_BRUT = RACINE / "data" / "raw"
DOSSIER_INTERMEDIAIRE = RACINE / "data" / "interim"


def telecharger_annee(annee: int, dossier: Path = DOSSIER_BRUT) -> Path:
    """Télécharge l'archive zip d'une année, sauf si elle est déjà présente."""
    dossier.mkdir(parents=True, exist_ok=True)
    chemin_zip = dossier / f"prix_{annee}.zip"
    if not chemin_zip.exists():
        reponse = requests.get(URL_ANNEE.format(annee=annee), timeout=120)
        reponse.raise_for_status()
        chemin_zip.write_bytes(reponse.content)
    return chemin_zip


def extraire_xml(chemin_zip: Path) -> Path:
    """Extrait l'unique fichier XML de l'archive et renvoie son chemin."""
    with zipfile.ZipFile(chemin_zip) as archive:
        nom_xml = next(n for n in archive.namelist() if n.endswith(".xml"))
        archive.extract(nom_xml, chemin_zip.parent)
    return chemin_zip.parent / nom_xml


def lire_xml_prix(chemin_xml: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit le XML en flux et renvoie (stations, prix).

    Chaque balise <pdv> (point de vente) porte les attributs de la station
    et contient une balise <prix> par changement de prix dans l'année.
    """
    stations, prix = [], []
    for _, pdv in etree.iterparse(str(chemin_xml), events=("end",), tag="pdv"):
        id_station = pdv.get("id")
        stations.append(
            {
                "id_station": id_station,
                "latitude": pdv.get("latitude"),
                "longitude": pdv.get("longitude"),
                "cp": pdv.get("cp"),
                "pop": pdv.get("pop"),
                "ville": pdv.findtext("ville"),
                "adresse": pdv.findtext("adresse"),
            }
        )
        for p in pdv.iterfind("prix"):
            prix.append(
                {
                    "id_station": id_station,
                    "carburant": p.get("nom"),
                    "date_maj": p.get("maj"),
                    "prix_ttc": p.get("valeur"),
                }
            )
        pdv.clear()  # libère la mémoire de l'élément déjà lu
    return pd.DataFrame(stations), pd.DataFrame(prix)


def typer_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """Convertit les coordonnées en degrés décimaux (stockées multipliées par 100 000)."""
    stations = stations.copy()
    for col in ["latitude", "longitude"]:
        x = pd.to_numeric(stations[col], errors="coerce")
        stations[col] = x.where(x.abs() <= 180, x / 100_000)
    return stations.drop_duplicates("id_station")


def typer_prix(prix: pd.DataFrame) -> pd.DataFrame:
    """Convertit les prix en €/L et les dates en datetime."""
    prix = prix.copy()
    x = pd.to_numeric(prix["prix_ttc"], errors="coerce")
    prix["prix_ttc"] = x.where(x < 10, x / 1000)  # millièmes d'euro -> euros
    prix["date_maj"] = pd.to_datetime(prix["date_maj"], format="ISO8601", errors="coerce")
    prix["carburant"] = prix["carburant"].astype("category")
    return prix


def prix_annee(annee: int, dossier_sortie: Path = DOSSIER_INTERMEDIAIRE) -> None:
    """Chaîne complète pour une année : zip -> XML -> stations et prix en Parquet."""
    chemin_xml = extraire_xml(telecharger_annee(annee))
    stations, prix = lire_xml_prix(chemin_xml)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    typer_stations(stations).to_parquet(dossier_sortie / f"stations_{annee}.parquet")
    typer_prix(prix).to_parquet(dossier_sortie / f"prix_{annee}.parquet")
    