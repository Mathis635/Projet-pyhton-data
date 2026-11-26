import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from carburants.couts import charger_serie_fred
from carburants.descriptif import COULEUR_BAISSE, COULEUR_HAUSSE
from carburants.prix import DOSSIER_INTERMEDIAIRE

TVA = 0.2
URL_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
COTATIONS = {"Gazole": "DDFUELNYH", "E10": "DGASNYH", "SP95": "DGASNYH", "SP98": "DGASNYH"}
LITRES_PAR_GALLON = 3.78541
REMISES_ETAT = [("2022-04-01", "2022-08-31", 0.18), ("2022-09-01", "2022-11-15", 0.30),
                ("2022-11-16", "2022-12-31", 0.10)]


def remise_etat(jours):
    """Montant TTC de la remise de l'État (€/L), jour par jour."""
    r = pd.Series(0.0, index=jours.index)
    for debut, fin, montant in REMISES_ETAT:
        r[jours.between(debut, fin)] = montant
    return r


def prix_hors_tva(annees, carburant="Gazole"):
    """Prix hors TVA, remise de l'État réintégrée, hors stations TotalEnergies, sur plusieurs années."""
    morceaux = []
    for a in annees:
        p = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"panel_{a}.parquet",
                            columns=["id_station", "carburant", "jour", "prix_ttc"])
        st = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"stations_enseignes_{a}.parquet")
        total = st.loc[st["enseigne"] == "TotalEnergies", "id_station"].astype(str)
        morceaux.append(p[(p["carburant"] == carburant) & ~p["id_station"].isin(total)])
    p = pd.concat(morceaux, ignore_index=True)
    p["prix"] = (p["prix_ttc"] + remise_etat(p["jour"])) / (1 + TVA)
    return p[["id_station", "jour", "prix"]]


def cout_produit(carburant, couts):
    """Cotation quotidienne du produit raffiné à New York (FRED, $/gallon), convertie en €/L."""
    cotation = charger_serie_fred(URL_FRED + COTATIONS[carburant], "usd_gallon")
    base = couts[["jour", "eurusd"]].drop_duplicates("jour")
    d = base.merge(cotation, on="jour", how="left")
    d["usd_gallon"] = d["usd_gallon"].ffill()
    d["cout_eur_l"] = d["usd_gallon"] / (d["eurusd"] * LITRES_PAR_GALLON)
    return d[["jour", "cout_eur_l"]]


def serie_nationale(prix, couts):
    """Prix moyen national hors TVA (p) et coût du produit (c), en €/L, jour par jour."""
    p = prix.groupby("jour")["prix"].mean().rename("p")
    c = couts.set_index("jour")["cout_eur_l"].rename("c")
    return pd.concat([p, c], axis=1, join="inner")

def retards(x, nom, K):
    """Colonnes x_t, x_{t-1}, ..., x_{t-K}."""
    return pd.concat({f"{nom}_{k}": x.shift(k) for k in range(K + 1)}, axis=1)


def modele_asymetrique(serie, K=21, robuste=True):
    """Modèle B asymétrique : Δp_t = α + Σ β⁺_k Δc⁺_{t-k} + Σ β⁻_k Δc⁻_{t-k} + ε_t, écarts-types de Newey-West."""
    dc = serie["c"].diff()
    X = pd.concat([retards(dc.clip(lower=0), "hausse", K), retards(dc.clip(upper=0), "baisse", K)], axis=1)
    d = pd.concat([serie["p"].diff().rename("y"), X], axis=1).dropna()
    modele = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y")))
    return modele.fit(cov_type="HAC", cov_kwds={"maxlags": K}) if robuste else modele.fit()


def transmission_cumulee(res, nom, K):
    """Somme des coefficients β_0 + ... + β_k pour k = 0..K, avec un intervalle de confiance à 95 %."""
    noms = [f"{nom}_{k}" for k in range(K + 1)]
    V = res.cov_params().loc[noms, noms].to_numpy()
    cumul = res.params[noms].cumsum().to_numpy()
    et = np.sqrt([V[: k + 1, : k + 1].sum() for k in range(K + 1)])
    return pd.DataFrame({"cumul": cumul, "bas": cumul - 1.96 * et, "haut": cumul + 1.96 * et})


def tests_asymetrie(res, K):
    """Tests de Wald de l'égalité des effets des hausses et des baisses du coût."""
    somme = " + ".join(f"hausse_{k}" for k in range(K + 1)) + " = " + " + ".join(f"baisse_{k}" for k in range(K + 1))
    hypotheses = {
        "même transmission cumulée sur K jours": somme,
        "mêmes coefficients jour par jour": ", ".join(f"hausse_{k} = baisse_{k}" for k in range(K + 1)),
    }
    lignes = {}
    for nom, h in hypotheses.items():
        w = res.wald_test(h, scalar=True)
        lignes[nom] = {"statistique": float(w.statistic), "p-valeur": float(w.pvalue)}
    return pd.DataFrame(lignes).T.round(3)


def graphique_transmission(res, K):
    """Transmission cumulée d'une hausse et d'une baisse de 1 c€ du coût (modèle B), avec intervalles à 95 %."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nom, couleur in [("hausse", COULEUR_HAUSSE), ("baisse", COULEUR_BAISSE)]:
        t = transmission_cumulee(res, nom, K)
        ax.fill_between(t.index, t["bas"], t["haut"], color=couleur, alpha=0.15)
        ax.plot(t.index, t["cumul"], color=couleur, marker="o", markersize=3, label=f"{nom} du coût")
    ax.axhline(1, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Jours après la variation du coût")
    ax.set_ylabel("Part transmise au prix hors TVA")
    ax.set_title("Transmission cumulée des hausses et des baisses du coût")
    ax.legend(loc="lower right")
    return fig
