import gzip
import io
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd

PATH_STATIONS = Path("data/interim/stations_enseignes_2022.parquet")
PATH_SORTIE = Path("data/interim/stations_geo_2022.parquet")

URL_CONTOURS_COMMUNES = (
    "https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/latest/geojson/communes-100m.geojson.gz"
)
URL_INSEE_PARQUET = (
    "https://static.data.gouv.fr/resources/recensement-de-la-population-communal-et-filosofi-depuis-2015-france-metropolitaine/20241104-093439/donnees-insee-olap.parquet"
)


def rattacher_communes(df_stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rattache chaque station à son code commune INSEE et calcule la superficie des communes (km²)."""
    df_stations = df_stations.reset_index(drop=True)
    gdf_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["longitude"], df_stations["latitude"]),
        crs="EPSG:4326",
    )

    resp = requests.get(URL_CONTOURS_COMMUNES, timeout=60)
    resp.raise_for_status()
    geojson_bytes = gzip.decompress(resp.content)
    gdf_communes = gpd.read_file(io.BytesIO(geojson_bytes))[["code", "geometry"]]
    gdf_communes = gdf_communes.rename(columns={"code": "code_commune"})

    # Calcul de la superficie en km² en projection Lambert-93 (EPSG:2154)
    communes_l93 = gdf_communes.to_crs(epsg=2154)
    df_superficie = pd.DataFrame({
        "code_commune": communes_l93["code_commune"].astype(str).str.zfill(5),
        "superficie_km2": communes_l93.geometry.area / 1_000_000,
    }).drop_duplicates("code_commune")

    # Jointure spatiale (point dans polygone) et suppression d'éventuels doublons de frontière
    joined = gpd.sjoin(gdf_stations, gdf_communes, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    # Gestion des stations en bordure côtière (plus proche voisin en Lambert-93)
    manquants = (
        joined["code_commune"].isna()
        & gdf_stations.geometry.is_valid
        & ~gdf_stations.geometry.is_empty
    )
    if manquants.any():
        stations_l93 = gdf_stations.loc[manquants].to_crs(epsg=2154)
        nearest = gpd.sjoin_nearest(stations_l93, communes_l93, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[nearest.index, "code_commune"] = nearest["code_commune"].values

    df_out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    return df_out, df_superficie


def charger_donnees_insee(df_superficie: pd.DataFrame) -> pd.DataFrame:
    """Charge la densité de population (hab/km²) et le revenu disponible médian (2021)."""
    df_insee = pd.read_parquet(
        URL_INSEE_PARQUET,
        columns=["code_com", "annee", "clef_json", "valeur"],
    )
    df_2021 = df_insee[
        (df_insee["annee"] == 2021)
        & (df_insee["clef_json"].isin(["revenu_median", "population_menages_p"]))
    ].copy()

    df_2021["code_commune"] = df_2021["code_com"].astype(str).str.zfill(5)
    df_2021["valeur"] = pd.to_numeric(df_2021["valeur"], errors="coerce")

    df_pivot = (
        df_2021.pivot_table(
            index="code_commune",
            columns="clef_json",
            values="valeur",
            aggfunc="first",
        )
        .reset_index()
    )

    df_pivot = df_pivot.merge(df_superficie, on="code_commune", how="left")
    df_pivot["densite"] = df_pivot["population_menages_p"] / df_pivot["superficie_km2"]

    return df_pivot[["code_commune", "densite", "revenu_median"]]


def construire_table_geo(
    path_stations: Path = PATH_STATIONS,
    path_sortie: Path = PATH_SORTIE,
) -> pd.DataFrame:
    """Construit la table géographique enrichie des stations."""
    if not path_stations.exists():
        path_stations = Path("data/interim/stations_2022.parquet")
    df_stations = pd.read_parquet(path_stations)
    df, df_superficie = rattacher_communes(df_stations)

    df["code_commune"] = df["code_commune"].astype(str).str.zfill(5)
    df_insee = charger_donnees_insee(df_superficie)

    df = df.merge(df_insee, on="code_commune", how="left")

    colonnes_attendues = ["id_station", "code_commune", "densite", "revenu_median"]
    df_final = df[colonnes_attendues].copy()

    path_sortie.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(path_sortie, index=False)
    return df_final


if __name__ == "__main__":
    df_geo = construire_table_geo()
    print(df_geo.head())
    print("\nValeurs manquantes :")
    print(df_geo.isna().sum())