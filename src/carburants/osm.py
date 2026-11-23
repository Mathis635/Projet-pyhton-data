"""Enseigne des stations : appariement avec OpenStreetMap (balise amenity=fuel).

Chaîne : requête Overpass (copie figée en JSON) -> table OSM -> normalisation des
enseignes -> appariement aux stations prix-carburants (identifiant, puis plus
proche voisin, un pour un) -> activité des stations -> nombre de concurrents
d'une autre enseigne parmi les stations actives, dans un rayon donné.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from carburants.prix import activite_stations

URLS_OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
ENTETES_OVERPASS = {
    "User-Agent": "projet-ensae-carburants/0.1 (+https://github.com/Mathis635/Projet-pyhton-data)",
    "Referer": "https://github.com/Mathis635/Projet-pyhton-data",
}
REQUETE_OVERPASS = """
[out:json][timeout:300];
area["ISO3166-1"="FR"]->.fr;
nwr["amenity"="fuel"](area.fr);
out center tags;
"""
RACINE = Path(__file__).resolve().parents[2]
CHEMIN_OSM = RACINE / "data" / "raw" / "osm_stations.json"
RAYON_TERRE_M = 6_371_000

# Variantes d'écriture rencontrées dans OSM -> enseigne normalisée.
ENSEIGNES = {
    "TotalEnergies": ["total", "totalenergies", "total access", "elan", "élan"],
    "Leclerc": ["leclerc"],
    "Carrefour": ["carrefour", "proxi"],
    "Intermarché": ["intermarché", "intermarche", "netto", "groupement des mousquetaires"],
    "Système U": ["systeme u", "système u", "super u", "hyper u", "u express", "u",
                  "station u", "la station u"],
    "Auchan": ["auchan"],
    "Casino": ["casino", "géant", "geant", "spar"],
    "Cora": ["cora", "match"],
    "Esso": ["esso", "mobil"],
    "BP": ["bp"],
    "Shell": ["shell"],
    "Avia": ["avia"],
    "Eni": ["eni", "agip"],
    "Dyneff": ["dyneff"],
    "Vito": ["vito"],
    "AS 24": ["as 24", "as24"],
    "Tamoil": ["tamoil"],
    "Gulf": ["gulf"],
    "Q8": ["q8"],
    "Engen": ["engen"],
    "Fulli": ["fulli"],
}
GRANDES_SURFACES = {"Leclerc", "Carrefour", "Intermarché", "Système U", "Auchan", "Casino", "Cora"}

MAJORS = {"TotalEnergies", "Esso", "BP", "Shell", "Eni"}


def telecharger_osm(chemin: Path = CHEMIN_OSM, essais: int = 3) -> dict:
    """Interroge Overpass et garde une copie figée du résultat (l'API évolue).

    Essaie plusieurs serveurs publics, avec une pause croissante entre les essais :
    les serveurs Overpass sont souvent saturés (erreurs 429, 502, 504).
    """
    if not chemin.exists():
        for essai in range(essais):
            for url in URLS_OVERPASS:
                try:
                    reponse = requests.post(
                        url, data={"data": REQUETE_OVERPASS},
                        headers=ENTETES_OVERPASS, timeout=360,
                    )
                    reponse.raise_for_status()
                    chemin.parent.mkdir(parents=True, exist_ok=True)
                    chemin.write_text(reponse.text, encoding="utf-8")
                    return json.loads(reponse.text)
                except requests.RequestException as erreur:
                    print(f"Échec sur {url} : {erreur}")
            time.sleep(30 * (essai + 1))
        raise RuntimeError("Aucun serveur Overpass n'a répondu.")
    return json.loads(chemin.read_text(encoding="utf-8"))


def table_osm(donnees: dict) -> pd.DataFrame:
    """Une ligne par station OSM ; les surfaces (way, relation) sont réduites à leur centre."""
    lignes = []
    for el in donnees["elements"]:
        centre = el.get("center", el)  # un node a lat/lon, un way a center{lat, lon}
        tags = el.get("tags", {})
        lignes.append(
            {
                "osm_id": f"{el['type']}/{el['id']}",
                "lat": centre.get("lat"),
                "lon": centre.get("lon"),
                "brand": tags.get("brand"),
                "operator": tags.get("operator"),
                "name": tags.get("name"),
                "ref_prix_carburants": tags.get("ref:FR:prix-carburants"),
            }
        )
    return pd.DataFrame(lignes).dropna(subset=["lat", "lon"])


def normaliser_enseigne(texte: str | None) -> str | None:
    """Renvoie l'enseigne normalisée d'un libellé OSM, ou None si inconnue."""
    if not isinstance(texte, str):
        return None
    t = re.sub(r"[^\w]+", " ", texte.lower()).strip()  # "E.Leclerc" -> "e leclerc"
    for enseigne, variantes in ENSEIGNES.items():
        if any(f" {v} " in f" {t} " for v in variantes):  # v présent comme mot entier
            return enseigne
    return None


def ajouter_enseigne(osm: pd.DataFrame) -> pd.DataFrame:
    """Enseigne tirée de brand, à défaut operator, à défaut name ; puis type d'enseigne."""
    osm = osm.copy()
    osm["enseigne"] = (
        osm["brand"].map(normaliser_enseigne)
        .fillna(osm["operator"].map(normaliser_enseigne))
        .fillna(osm["name"].map(normaliser_enseigne))
    )
    osm["type_enseigne"] = np.select(
        [osm["enseigne"].isin(GRANDES_SURFACES), osm["enseigne"].isin(MAJORS), osm["enseigne"].notna()],
        ["grande surface", "major", "autre enseigne"],
        default="inconnue",
    )
    return osm


def apparier(stations: pd.DataFrame, osm: pd.DataFrame, seuil_m: float = 300) -> pd.DataFrame:
    """Associe à chaque station prix-carburants une station OSM.

    1. jointure exacte sur l'identifiant (balise ref:FR:prix-carburants) ;
    2. pour les autres, station OSM libre la plus proche si elle est à moins de seuil_m
       mètres, chaque station OSM ne servant qu'une fois.
    """
    cols = ["osm_id", "enseigne", "type_enseigne"]
    stations = stations.copy()
    stations["id_station"] = stations["id_station"].astype(str)

    par_id = osm.dropna(subset=["ref_prix_carburants"]).drop_duplicates("ref_prix_carburants")
    res = stations.merge(
        par_id[cols + ["ref_prix_carburants"]],
        left_on="id_station", right_on="ref_prix_carburants", how="left",
    ).drop(columns="ref_prix_carburants")
    res["methode"] = np.where(res["osm_id"].notna(), "identifiant", None)
    res["distance_m"] = np.nan

    reste = res["osm_id"].isna() & res["latitude"].notna()
    libres = osm[~osm["osm_id"].isin(res["osm_id"].dropna())]  # OSM pas déjà pris par identifiant
    arbre = BallTree(np.radians(libres[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = arbre.query(np.radians(res.loc[reste, ["latitude", "longitude"]].to_numpy()), k=1)
    candidats = pd.DataFrame(
        {"ligne": res.index[reste], "pos": idx[:, 0], "distance_m": dist[:, 0] * RAYON_TERRE_M}
    )
    # Sous le seuil, et chaque station OSM ne sert qu'une fois : la plus proche la garde
    candidats = (
        candidats[candidats["distance_m"] <= seuil_m]
        .sort_values("distance_m")
        .drop_duplicates("pos")
    )
    lignes = candidats["ligne"].to_numpy()
    res.loc[lignes, cols] = libres.iloc[candidats["pos"]][cols].to_numpy()
    res.loc[lignes, "methode"] = "plus proche voisin"
    res.loc[lignes, "distance_m"] = candidats["distance_m"].to_numpy()
    res["type_enseigne"] = res["type_enseigne"].fillna("non appariée")
    return res


def n_concurrents(stations: pd.DataFrame, rayon_m: float = 5000) -> pd.Series:
    """Nombre de stations d'une autre enseigne dans un rayon de rayon_m mètres.

    Une station d'enseigne inconnue est comptée comme concurrente.
    """
    st = stations.dropna(subset=["latitude", "longitude"])
    coords = np.radians(st[["latitude", "longitude"]].to_numpy())
    voisins = BallTree(coords, metric="haversine").query_radius(coords, r=rayon_m / RAYON_TERRE_M)
    enseigne = st["enseigne"].to_numpy()
    inconnue = pd.isna(enseigne)
    n = [
        sum(1 for j in v if j != i and (inconnue[i] or inconnue[j] or enseigne[j] != enseigne[i]))
        for i, v in enumerate(voisins)
    ]
    return pd.Series(n, index=st.index, name="n_concurrents").reindex(stations.index)


def stations_enseignes(
    annee: int, seuil_m: float = 300, rayon_m: float = 5000, seuil_activite: int = 50
) -> pd.DataFrame:
    """Chaîne complète : stations de l'année + activité + enseigne OSM + concurrents.

    Lit data/interim/stations_{annee}.parquet et prix_{annee}.parquet (produits par
    prix.prix_annee) et écrit data/interim/stations_enseignes_{annee}.parquet.
    Les concurrents ne sont comptés que parmi les stations actives : une station
    fermée ou sans prix publié ne fait pas concurrence.
    """
    dossier = RACINE / "data" / "interim"
    stations = pd.read_parquet(dossier / f"stations_{annee}.parquet")
    prix = pd.read_parquet(dossier / f"prix_{annee}.parquet")
    osm = ajouter_enseigne(table_osm(telecharger_osm()))

    res = apparier(stations, osm, seuil_m=seuil_m)
    res = res.join(activite_stations(prix, seuil=seuil_activite), on="id_station")
    res["active"] = res["active"].fillna(False).astype(bool)  # aucun relevé : inactive
    res["n_concurrents"] = n_concurrents(res[res["active"]], rayon_m=rayon_m)  # NaN si inactive
    res.to_parquet(dossier / f"stations_enseignes_{annee}.parquet")
    return res