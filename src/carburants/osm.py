import json

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from carburants.prix import DOSSIER_BRUT, DOSSIER_INTERMEDIAIRE, activite_stations

RAYON_TERRE = 6_371_000
REQUETE = """[out:json][timeout:300];
area["ISO3166-1"="FR"]->.fr;
nwr["amenity"="fuel"](area.fr);
out center tags;"""

ENSEIGNES = {
    "TotalEnergies": ["total", "totalenergies", "total access", "elan", "élan"],
    "Leclerc": ["leclerc"],
    "Carrefour": ["carrefour", "proxi"],
    "Intermarché": ["intermarché", "intermarche", "netto", "groupement des mousquetaires"],
    "Système U": ["systeme u", "système u", "super u", "hyper u", "u express", "u", "station u", "la station u"],
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


def enseigne(texte):
    """Ramène un libellé OSM (« E.Leclerc ») à une enseigne (« Leclerc »), ou None."""
    if not isinstance(texte, str):
        return None
    lettres = ""
    for lettre in texte.lower():
        if lettre.isalnum() or lettre == "_":
            lettres += lettre
        else:
            lettres += " "
    mots = " " + " ".join(lettres.split()) + " "
    for nom, variantes in ENSEIGNES.items():
        for variante in variantes:
            if " " + variante + " " in mots:
                return nom
    return None


def type_enseigne(nom):
    """Grande surface, major, autre enseigne ou inconnue."""
    if nom in GRANDES_SURFACES:
        return "grande surface"
    if nom in MAJORS:
        return "major"
    return "autre enseigne" if isinstance(nom, str) else "inconnue"


def charger_osm():
    """Stations-service OpenStreetMap de France avec leur enseigne (téléchargées une seule fois)."""
    chemin = DOSSIER_BRUT / "osm_stations.json"
    if not chemin.exists():
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": REQUETE}, timeout=360,
                          headers={"User-Agent": "projet-ensae-carburants",
                                   "Referer": "https://github.com/Mathis635/Projet-pyhton-data"})
        r.raise_for_status()
        chemin.write_text(r.text, encoding="utf-8")

    lignes = []
    for el in json.loads(chemin.read_text(encoding="utf-8"))["elements"]:
        centre, tags = el.get("center", el), el.get("tags", {})
        lignes.append([f"{el['type']}/{el['id']}", centre.get("lat"), centre.get("lon"), tags.get("brand"),
                       tags.get("operator"), tags.get("name"), tags.get("ref:FR:prix-carburants")])
    osm = pd.DataFrame(lignes, columns=["osm_id", "lat", "lon", "brand", "operator", "name", "ref"])
    osm = osm.dropna(subset=["lat", "lon"])
    osm["enseigne"] = osm["brand"].map(enseigne)
    osm["enseigne"] = osm["enseigne"].fillna(osm["operator"].map(enseigne))
    osm["enseigne"] = osm["enseigne"].fillna(osm["name"].map(enseigne))
    osm["type_enseigne"] = osm["enseigne"].map(type_enseigne)
    return osm


def apparier(stations, osm, seuil=300):
    """Relie chaque station à une station OSM : par identifiant, sinon la plus proche libre à moins de `seuil` m."""
    st = stations.assign(id_station=stations["id_station"].astype(str))
    par_ref = osm.dropna(subset=["ref"]).drop_duplicates("ref").set_index("ref")["osm_id"]
    st["osm_id"] = st["id_station"].map(par_ref)
    st["methode"] = np.where(st["osm_id"].notna(), "identifiant", None)

    libres = osm[~osm["osm_id"].isin(st["osm_id"])]
    a_faire = st[st["osm_id"].isna() & st["latitude"].notna()]
    arbre = BallTree(np.radians(libres[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = arbre.query(np.radians(a_faire[["latitude", "longitude"]].to_numpy()))
    proches = pd.DataFrame({"osm_id": libres["osm_id"].to_numpy()[idx[:, 0]], "distance_m": dist[:, 0] * RAYON_TERRE},
                           index=a_faire.index)
    proches = proches[proches["distance_m"] <= seuil].sort_values("distance_m").drop_duplicates("osm_id")

    st.loc[proches.index, "osm_id"] = proches["osm_id"]
    st.loc[proches.index, "methode"] = "plus proche voisin"
    st["distance_m"] = proches["distance_m"]
    st = st.merge(osm[["osm_id", "enseigne", "type_enseigne"]], on="osm_id", how="left")
    st["type_enseigne"] = st["type_enseigne"].fillna("non appariée")
    return st


def n_concurrents(stations, rayon=5000):
    """Nombre de stations d'une autre enseigne à moins de `rayon` m (enseigne inconnue comptée comme concurrente)."""
    st = stations.dropna(subset=["latitude", "longitude"])
    coords = np.radians(st[["latitude", "longitude"]].to_numpy())
    voisins = BallTree(coords, metric="haversine").query_radius(coords, r=rayon / RAYON_TERRE)
    ens = st["enseigne"].fillna("?").to_numpy()
    n = []
    for i in range(len(voisins)):
        compte = 0
        for j in voisins[i]:
            if j != i and (ens[j] != ens[i] or ens[i] == "?"):
                compte += 1
        n.append(compte)
    return pd.Series(n, index=st.index).reindex(stations.index)


def stations_enseignes(annee, seuil=300, rayon=5000):
    """Stations de l'année avec enseigne, activité et nombre de concurrents, enregistrées en Parquet."""
    stations = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"stations_{annee}.parquet")
    prix = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"prix_{annee}.parquet")
    st = apparier(stations, charger_osm(), seuil)
    st = st.join(activite_stations(prix), on="id_station")
    st["active"] = st["active"].eq(True)
    st["n_concurrents"] = n_concurrents(st[st["active"]], rayon)
    st.to_parquet(DOSSIER_INTERMEDIAIRE / f"stations_enseignes_{annee}.parquet")
    return st
