import pandas as pd

URL_BRENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
URL_EURUSD = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"
LITRES_PAR_BARIL = 158.987

# Tarifs nationaux de l'accise sur les énergies (ex-TICPE) en €/L HT (2021-2023)
# Source : Direction Générale des Douanes et Droits Indirects (DGDDI) / Art. L. 312-35 du CIBS
# (Tarifs métropole figés depuis 2019 : 59,40 €/hL Gazole, 66,29 €/hL E10, 68,29 €/hL SP95/SP98)
ACCISE_EUR_L = {
    "Gazole": 0.5940,
    "E10": 0.6629,
    "SP95": 0.6829,
    "SP98": 0.6829,
}

# Calendrier officiel des remises carburant 2022 (montants TTC en €/L)
# Sources :
# - État : Décrets n° 2022-423, 2022-1042, 2022-1168 et 2022-1355
# - TotalEnergies : Communiqué de presse du 22 juillet 2022 (12 c€/L sur autoroute du 01/07 au 31/08/2022)
CALENDRIER_REMISES = [
    {
        "debut": "2022-04-01",
        "fin": "2022-06-30",
        "etat": 0.18,
        "total": 0.10,
        "total_autoroute": 0.10,
    },
    {
        "debut": "2022-07-01",
        "fin": "2022-08-31",
        "etat": 0.18,
        "total": 0.10,
        "total_autoroute": 0.12,
    },
    {
        "debut": "2022-09-01",
        "fin": "2022-10-31",
        "etat": 0.30,
        "total": 0.20,
        "total_autoroute": 0.20,
    },
    {
        "debut": "2022-11-01",
        "fin": "2022-11-15",
        "etat": 0.30,
        "total": 0.10,
        "total_autoroute": 0.10,
    },
    {
        "debut": "2022-11-16",
        "fin": "2022-12-31",
        "etat": 0.10,
        "total": 0.10,
        "total_autoroute": 0.10,
    },
]


def charger_serie_fred(url: str, nom_colonne: str) -> pd.DataFrame:
    """Télécharge une série FRED au format CSV et nettoie les valeurs manquantes ('.')."""
    df = pd.read_csv(url, na_values=["."])
    col_date, col_val = df.columns[0], df.columns[1]
    df = df.rename(columns={col_date: "jour", col_val: nom_colonne})
    df["jour"] = pd.to_datetime(df["jour"])
    df[nom_colonne] = pd.to_numeric(df[nom_colonne], errors="coerce")
    return df


def construire_table_couts(
    date_debut: str = "2021-01-01",
    date_fin: str = "2023-12-31",
) -> pd.DataFrame:
    """
    Construit la table quotidienne au format long (jour x carburant) prête à être jointe au panel.

    Colonnes retournées :
        - jour (datetime64[ns])
        - carburant (str) : 'Gazole', 'E10', 'SP95', 'SP98'
        - cout_eur_l (float) : coût du brut en €/L = Brent / (EUR/USD * 158.987)
        - accise_eur_l (float) : tarif national d'accise (TICPE) en €/L HT
        - remise_etat_ttc (float) : remise de l'État en €/L TTC
        - remise_total_ttc (float) : remise TotalEnergies hors autoroute en €/L TTC
        - remise_total_autoroute_ttc (float) : remise TotalEnergies sur autoroute en €/L TTC
    """
    df_brent = charger_serie_fred(URL_BRENT, "brent_usd")
    df_eurusd = charger_serie_fred(URL_EURUSD, "eurusd")

    # Fusion et réindexation sur tous les jours calendaires (propagation des jours fermés)
    df = pd.merge(df_brent, df_eurusd, on="jour", how="outer").sort_values("jour")
    grille_dates = pd.DataFrame(
        {"jour": pd.date_range(df["jour"].min(), df["jour"].max(), freq="D")}
    )
    df = pd.merge(grille_dates, df, on="jour", how="left")
    df[["brent_usd", "eurusd"]] = df[["brent_usd", "eurusd"]].ffill()

    # Filtrage sur la période demandée
    masque_periode = (df["jour"] >= pd.to_datetime(date_debut)) & (
        df["jour"] <= pd.to_datetime(date_fin)
    )
    df = df.loc[masque_periode].copy()

    # Calcul du coût en €/L
    df["cout_eur_l"] = df["brent_usd"] / (df["eurusd"] * LITRES_PAR_BARIL)

    # Application du calendrier des remises via boucle
    df["remise_etat_ttc"] = 0.0
    df["remise_total_ttc"] = 0.0
    df["remise_total_autoroute_ttc"] = 0.0

    for periode in CALENDRIER_REMISES:
        masque = (df["jour"] >= periode["debut"]) & (df["jour"] <= periode["fin"])
        df.loc[masque, "remise_etat_ttc"] = periode["etat"]
        df.loc[masque, "remise_total_ttc"] = periode["total"]
        df.loc[masque, "remise_total_autoroute_ttc"] = periode["total_autoroute"]

    # Passage au format long (produit cartésien jour x carburant)
    df_carburants = pd.DataFrame(
        {
            "carburant": list(ACCISE_EUR_L.keys()),
            "accise_eur_l": list(ACCISE_EUR_L.values()),
        }
    )
    df_long = df.merge(df_carburants, how="cross")

    colonnes_finales = [
        "jour",
        "carburant",
        "eurusd",
        "cout_eur_l",
        "accise_eur_l",
        "remise_etat_ttc",
        "remise_total_ttc",
        "remise_total_autoroute_ttc",
    ]
    return (
        df_long[colonnes_finales]
        .sort_values(["jour", "carburant"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    df_couts = construire_table_couts()
    print(df_couts.head(8))
    print("\nAperçu septembre 2022 (pic des remises) :")
    print(df_couts[df_couts["jour"] == "2022-09-15"])