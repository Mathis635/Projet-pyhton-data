import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import BallTree

from carburants.descriptif import COULEUR_BAISSE, COULEUR_HAUSSE
from carburants.osm import RAYON_TERRE
from carburants.prix import DOSSIER_INTERMEDIAIRE

TVA = 0.2


def prix_hors_taxes(annees, carburant, couts):
    """Prix hors taxes par station et par jour, hors stations TotalEnergies.

    (prix TTC + remise de l'État) / 1,2 − accise ; remise et accise viennent de la table des coûts."""
    morceaux = []
    for a in annees:
        p = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"panel_{a}.parquet",
                            columns=["id_station", "carburant", "jour", "prix_ttc"])
        st = pd.read_parquet(DOSSIER_INTERMEDIAIRE / f"stations_enseignes_{a}.parquet")
        total = st.loc[st["enseigne"] == "TotalEnergies", "id_station"].astype(str)
        morceaux.append(p[(p["carburant"] == carburant) & ~p["id_station"].isin(total)])
    p = pd.concat(morceaux, ignore_index=True)
    p = p.merge(couts[["jour", "carburant", "accise_eur_l", "remise_etat_ttc"]], on=["jour", "carburant"])
    p["prix"] = (p["prix_ttc"] + p["remise_etat_ttc"]) / (1 + TVA) - p["accise_eur_l"]
    return p[["id_station", "jour", "prix"]]


def serie_nationale(prix, couts, carburant):
    """Prix moyen national hors taxes (p) et coût du pétrole brut (c), en €/L, jour par jour."""
    p = prix.groupby("jour")["prix"].mean().rename("p")
    c = couts[couts["carburant"] == carburant].set_index("jour")["cout_eur_l"].rename("c")
    return pd.concat([p, c], axis=1, join="inner")


def retards(x, nom, K):
    """Colonnes x_t, x_{t-1}, ..., x_{t-K}."""
    colonnes = pd.DataFrame(index=x.index)
    for k in range(K + 1):
        colonnes[f"{nom}_{k}"] = x.shift(k)
    return colonnes


def modele_asymetrique(serie, K=21, robuste=True):
    """Modèle B asymétrique : Δp_t = α + Σ β⁺_k Δc⁺_{t-k} + Σ β⁻_k Δc⁻_{t-k} + ε_t, écarts-types de Newey-West."""
    dc = serie["c"].diff()
    hausses = dc.copy()
    hausses[hausses < 0] = 0
    baisses = dc.copy()
    baisses[baisses > 0] = 0
    X = pd.concat([retards(hausses, "hausse", K), retards(baisses, "baisse", K)], axis=1)
    d = pd.concat([serie["p"].diff().rename("y"), X], axis=1).dropna()
    modele = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y")))
    return modele.fit(cov_type="HAC", cov_kwds={"maxlags": K}) if robuste else modele.fit()


def transmission_cumulee(res, nom, K):
    """Somme des coefficients β_0 + ... + β_k pour k = 0..K, avec son intervalle de confiance à 95 % (t_test de statsmodels)."""
    lignes = []
    for k in range(K + 1):
        noms = [f"{nom}_{j}" for j in range(k + 1)]
        test = res.t_test(" + ".join(noms) + " = 0")
        bas, haut = test.conf_int(alpha=0.05)[0]
        lignes.append({"cumul": test.effect[0], "bas": bas, "haut": haut})
    return pd.DataFrame(lignes)


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


def stations_proches_total(stations, rayon=3000):
    """Pour chaque station active : vrai si une station TotalEnergies active est à moins de `rayon` mètres."""
    st = stations[stations["active"]].dropna(subset=["latitude", "longitude"])
    total = st[st["enseigne"] == "TotalEnergies"]
    arbre = BallTree(np.radians(total[["latitude", "longitude"]].to_numpy()), metric="haversine")
    n = arbre.query_radius(np.radians(st[["latitude", "longitude"]].to_numpy()), r=rayon / RAYON_TERRE,
                           count_only=True)
    return pd.Series(n > 0, index=st["id_station"].astype(str).to_numpy(), name="groupe")


def groupes_revenu(geo):
    """Tiers de revenu médian de la commune : vrai pour le tiers aisé, faux pour le tiers modeste, tiers du milieu écarté."""
    revenu = geo.set_index(geo["id_station"].astype(str))["revenu_median"].dropna()
    bas, haut = revenu.quantile([1 / 3, 2 / 3])
    aise = pd.Series(True, index=revenu[revenu >= haut].index)
    modeste = pd.Series(False, index=revenu[revenu <= bas].index)
    return pd.concat([aise, modeste]).rename("groupe")


def series_groupes(prix, couts, carburant, stations, groupe):
    """Variation moyenne du prix chaque jour dans deux groupes de stations (hors autoroute) et écart entre les deux.

    `groupe` associe à chaque station vrai ou faux. Chaque série est remise en niveau cumulé pour passer
    dans modele_asymetrique."""
    route = stations.loc[stations["pop"] != "A", "id_station"].astype(str)
    p = prix[prix["id_station"].isin(route)].sort_values(["id_station", "jour"])
    p["dp"] = p.groupby("id_station")["prix"].diff()
    p["ecart_jours"] = p.groupby("id_station")["jour"].diff().dt.days
    p["groupe"] = p["id_station"].map(groupe)
    p = p[(p["ecart_jours"] == 1) & p["groupe"].notna()]
    moy = p.groupby(["jour", "groupe"])["dp"].mean().unstack()
    c = couts[couts["carburant"] == carburant].set_index("jour")["cout_eur_l"]
    variations = {"non": moy[False], "oui": moy[True], "écart": moy[True] - moy[False]}
    series = {}
    for nom, dp in variations.items():
        series[nom] = pd.DataFrame({"p": dp.cumsum(), "c": c}).dropna()
    return series


def ecarts_groupes(res_ecart, horizons=(3, 7)):
    """Écart cumulé de transmission entre les deux groupes au bout de k jours, hausses et baisses, et p-valeurs."""
    lignes = {}
    for k in horizons:
        for sens in ["hausse", "baisse"]:
            noms = [f"{sens}_{j}" for j in range(k + 1)]
            test = res_ecart.t_test(" + ".join(noms) + " = 0")
            lignes[(f"{sens}s", f"{k} jours")] = {"écart cumulé": test.effect[0], "p-valeur": float(test.pvalue)}
    return pd.DataFrame(lignes).T.round(3)


def graphique_groupes(res_non, res_oui, K, noms, titre):
    """Transmission cumulée des hausses et des baisses du coût dans deux groupes de stations."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for res, style, groupe in [(res_non, "-", noms[0]), (res_oui, "--", noms[1])]:
        for sens, couleur in [("hausse", COULEUR_HAUSSE), ("baisse", COULEUR_BAISSE)]:
            t = transmission_cumulee(res, sens, K)
            ax.plot(t.index, t["cumul"], style, color=couleur, marker="o", markersize=3, label=f"{sens}, {groupe}")
    ax.set_xlabel("Jours après la variation du coût")
    ax.set_ylabel("Part transmise au prix hors taxes")
    ax.set_title(titre)
    ax.legend(loc="lower right", fontsize=8)
    return fig
