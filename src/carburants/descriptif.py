import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carburants.cartes import COULEURS_TYPE, metropole, preparer_axes
from carburants.prix import CARBURANTS_ROUTIERS

COULEUR_CARBURANT = {"Gazole": "#D55E00", "E10": "#009E73", "SP95": "#0072B2", "SP98": "#CC79A7"}
COULEUR_HAUSSE, COULEUR_BAISSE = "#C0392B", "#2E86C1"
COULEUR_REMISE = "#FFF8E1"
PERIODES_REMISE_ETAT = [
    ("2022-04-01", "2022-08-31", "0,18 €"),
    ("2022-09-01", "2022-11-15", "0,30 €"),
    ("2022-11-16", "2022-12-31", "0,10 €"),
]


def moyennes_par_station(panel, carburant="Gazole"):
    """Prix moyen de chaque station sur la période, pour un carburant."""
    return panel[panel["carburant"] == carburant].groupby("id_station")["prix_ttc"].mean()



def tableau_resume(panel):
    """Par carburant : stations, observations station x jour, moyenne, dispersion, déciles."""
    g = panel.groupby("carburant")["prix_ttc"]
    tab = pd.DataFrame({
        "stations": panel.groupby("carburant")["id_station"].nunique(),
        "observations": g.size(),
        "moyenne": g.mean(),
        "écart-type": g.std(),
        "D1": g.quantile(0.1),
        "médiane": g.median(),
        "D9": g.quantile(0.9),
    })
    return tab.reindex(CARBURANTS_ROUTIERS).round(3)


def tableau_changements(panel):
    """Par carburant : nombre et ampleur moyenne des hausses et des baisses de prix."""
    v = panel[panel["changement"]]
    hausse = v["variation"] > 0
    n_annees = panel["jour"].dt.year.nunique()
    tab = pd.DataFrame({
        "hausses": v[hausse].groupby("carburant").size(),
        "baisses": v[~hausse].groupby("carburant").size(),
        "hausse moyenne (c€)": v[hausse].groupby("carburant")["variation"].mean() * 100,
        "baisse moyenne (c€)": v[~hausse].groupby("carburant")["variation"].mean() * 100,
        "changements par station et par an": v.groupby("carburant").size()
        / panel.groupby("carburant")["id_station"].nunique() / n_annees,
    })
    return tab.reindex(CARBURANTS_ROUTIERS).round(2)



def marquer_remises(ax):
    """Colore les périodes de remise de l'État et indique leur montant."""
    for debut, fin, montant in PERIODES_REMISE_ETAT:
        ax.axvspan(pd.Timestamp(debut), pd.Timestamp(fin), color=COULEUR_REMISE, zorder=0)
        ax.axvline(pd.Timestamp(debut), color="#E0A800", linewidth=0.8, zorder=0)
        ax.text(pd.Timestamp(debut), 1.0, f" remise {montant}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="#8A6D00")


def serie_prix_moyen(panel):
    """Prix moyen national par jour et par carburant, périodes de remise colorées."""
    moy = panel.groupby(["jour", "carburant"])["prix_ttc"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    marquer_remises(ax)
    for carb in CARBURANTS_ROUTIERS:
        ax.plot(moy.index, moy[carb], label=carb, color=COULEUR_CARBURANT[carb], linewidth=1.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix moyen TTC (€/L)")
    ax.set_title("Prix moyen à la pompe en France métropolitaine")
    ax.legend(loc="lower left")
    return fig


def serie_dispersion(panel):
    """Écart D9 - D1 des prix entre stations, moyenne sur 7 jours, un panneau par carburant."""
    g = panel.groupby(["jour", "carburant"])["prix_ttc"]
    ecart = (g.quantile(0.9) - g.quantile(0.1)) * 100
    ecart = ecart.unstack()
    ecart = ecart.rolling(7, center=True, min_periods=1).mean()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, carb in zip(axes.flat, CARBURANTS_ROUTIERS):
        marquer_remises(ax)
        ax.fill_between(ecart.index, 0, ecart[carb], color=COULEUR_CARBURANT[carb], alpha=0.6)
        ax.set_title(carb)
    for ax in axes[:, 0]:
        ax.set_ylabel("D9 - D1 (c€/L)")
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Écart de prix entre les 10 % de stations les moins chères et les 10 % les plus chères")
    fig.tight_layout()
    return fig


def histogramme_variations(panel, carburant="Gazole"):
    """Distribution des variations journalières non nulles, hausses et baisses."""
    v = panel.loc[(panel["carburant"] == carburant) & panel["changement"], "variation"] * 100
    titre = f"Taille des changements de prix : {(v > 0).sum()} hausses, {(v < 0).sum()} baisses"
    fig, ax = plt.subplots(figsize=(9, 5))
    bornes = np.arange(-20, 20.5, 0.5)
    ax.hist(v[(v < 0) & (v >= -20)], bins=bornes, color=COULEUR_BAISSE, edgecolor="white", linewidth=0.3, label="baisses")
    ax.hist(v[(v > 0) & (v <= 20)], bins=bornes, color=COULEUR_HAUSSE, edgecolor="white", linewidth=0.3, label="hausses")
    ax.legend()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"Variation journalière du prix du {carburant.lower()} (c€/L)")
    ax.set_ylabel("Nombre de changements")
    ax.set_title(titre)
    return fig


def boites_type_enseigne(panel, stations, carburant="Gazole"):
    """Prix moyen des stations sur la période, par type d'enseigne."""
    moy = moyennes_par_station(panel, carburant)
    types = stations.set_index(stations["id_station"].astype(str))["type_enseigne"].reindex(moy.index)
    ordre = ["grande surface", "major", "autre enseigne", "inconnue", "non appariée"]
    groupes = [(t, moy[types == t]) for t in ordre if (types == t).any()]
    fig, ax = plt.subplots(figsize=(9, 5))
    boites = ax.boxplot([g for _, g in groupes], tick_labels=[f"{t}\n(n = {len(g)})" for t, g in groupes],
                        showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for boite, (t, _) in zip(boites["boxes"], groupes):
        boite.set_facecolor(COULEURS_TYPE[t])
    ax.set_ylabel(f"Prix moyen du {carburant.lower()} (€/L)")
    ax.set_title("Prix moyen des stations selon le type d'enseigne")
    return fig


def carte_prix(panel, stations, carburant="Gazole"):
    """Carte du prix moyen de chaque station sur la période ; du vert (moins cher) au rouge (plus cher)."""
    moy = moyennes_par_station(panel, carburant).rename("prix")
    st = stations.copy()
    st["id_station"] = st["id_station"].astype(str)
    st = st.join(moy, on="id_station")
    st = metropole(st.dropna(subset=["prix"]))
    bas, haut = st["prix"].quantile([0.05, 0.95])
    fig, ax = plt.subplots(figsize=(8, 8))
    nuage = ax.scatter(st["longitude"], st["latitude"], c=st["prix"], cmap="RdYlGn_r",
                       vmin=bas, vmax=haut, s=3)
    fig.colorbar(nuage, ax=ax, shrink=0.7, label=f"Prix moyen du {carburant.lower()} (€/L)")
    preparer_axes(ax, f"Prix moyen du {carburant.lower()} par station")
    return fig


def infographie_prix(panel, carburant="Gazole"):
    """Prix moyen d'un carburant jour par jour, avec la dispersion entre stations."""
    g = panel[panel["carburant"] == carburant].groupby("jour")["prix_ttc"]
    s = pd.DataFrame({"moy": g.mean(), "et": g.std(), "p01": g.quantile(0.01), "p99": g.quantile(0.99),
                      "d1": g.quantile(0.1), "d9": g.quantile(0.9)})
    c = COULEUR_CARBURANT[carburant]

    fig, ax = plt.subplots(figsize=(11, 6))
    marquer_remises(ax)
    ax.fill_between(s.index, s["d1"], s["d9"], color=c, alpha=0.15, label="D1 - D9 (80 % des stations)")
    ax.fill_between(s.index, s["moy"] - s["et"], s["moy"] + s["et"], color=c, alpha=0.35,
                    label="moyenne ± 1 écart-type")
    ax.plot(s.index, s["moy"], color=c, linewidth=2, label="prix moyen")
    ax.plot(s.index, s["p01"], color=COULEUR_BAISSE, linewidth=0.8, label="1er centile (1 % les moins chères)")
    ax.plot(s.index, s["p99"], color=COULEUR_HAUSSE, linewidth=0.8, label="99e centile (1 % les plus chères)")
    for jour, mot in [(s["moy"].idxmax(), "sommet"), (s["moy"].idxmin(), "creux")]:
        ax.annotate(f"{mot} : {s.loc[jour, 'moy']:.3f} €/L\n{jour:%d/%m/%Y}",
                    xy=(jour, s.loc[jour, "moy"]), xytext=(30, 15 if mot == "sommet" else 30),
                    textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "color": "0.3"})
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix TTC (€/L)")
    ax.set_title(f"{carburant} : prix moyen et dispersion entre stations, jour par jour")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    fig.tight_layout()
    return fig
