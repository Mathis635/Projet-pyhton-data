import pandas as pd

URL_BRENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
URL_EURUSD = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"
LITRES_PAR_BARIL = 158.987


def charger_serie_fred(url: str, nom_colonne: str) -> pd.DataFrame:
    """Télécharge une série FRED au format CSV et nettoie les valeurs manquantes ('.')."""
    df = pd.read_csv(url, na_values=["."])
    col_date, col_val = df.columns[0], df.columns[1]
    df = df.rename(columns={col_date: "date", col_val: nom_colonne})
    df["date"] = pd.to_datetime(df["date"])
    df[nom_colonne] = pd.to_numeric(df[nom_colonne], errors="coerce")
    return df


def construire_table_couts(
    date_debut: str = "2007-01-01",
    propager_jours_fermes: bool = True,
) -> pd.DataFrame:
    """Construit la table quotidienne avec date, brent_usd, eurusd et cout_eur_l."""
    df_brent = charger_serie_fred(URL_BRENT, "brent_usd")
    df_eurusd = charger_serie_fred(URL_EURUSD, "eurusd")

    df = pd.merge(df_brent, df_eurusd, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)

    if propager_jours_fermes:
        grille_dates = pd.DataFrame(
            {"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")}
        )
        df = pd.merge(grille_dates, df, on="date", how="left")
        df[["brent_usd", "eurusd"]] = df[["brent_usd", "eurusd"]].ffill()

    df = df[df["date"] >= pd.to_datetime(date_debut)].copy()
    df["cout_eur_l"] = df["brent_usd"] / (df["eurusd"] * LITRES_PAR_BARIL)

    return df.reset_index(drop=True)


def ajouter_remises_2022(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les montants TTC (en €/L) des remises État et TotalEnergies en 2022."""
    df = df.copy()
    df["remise_etat_ttc"] = 0.0
    df["remise_total_ttc"] = 0.0
    df["remise_total_autoroute_ttc"] = 0.0

    m1 = (df["date"] >= "2022-04-01") & (df["date"] <= "2022-08-31")
    df.loc[m1, "remise_etat_ttc"] = 0.18
    df.loc[m1, "remise_total_ttc"] = 0.10
    df.loc[m1, "remise_total_autoroute_ttc"] = 0.12

    m2 = (df["date"] >= "2022-09-01") & (df["date"] <= "2022-10-31")
    df.loc[m2, "remise_etat_ttc"] = 0.30
    df.loc[m2, "remise_total_ttc"] = 0.20
    df.loc[m2, "remise_total_autoroute_ttc"] = 0.20

    m3 = (df["date"] >= "2022-11-01") & (df["date"] <= "2022-11-15")
    df.loc[m3, "remise_etat_ttc"] = 0.30
    df.loc[m3, "remise_total_ttc"] = 0.10
    df.loc[m3, "remise_total_autoroute_ttc"] = 0.10

    m4 = (df["date"] >= "2022-11-16") & (df["date"] <= "2022-12-31")
    df.loc[m4, "remise_etat_ttc"] = 0.10
    df.loc[m4, "remise_total_ttc"] = 0.10
    df.loc[m4, "remise_total_autoroute_ttc"] = 0.10

    return df


if __name__ == "__main__":
    df_couts = construire_table_couts()
    df_couts = ajouter_remises_2022(df_couts)
    print(df_couts.tail())