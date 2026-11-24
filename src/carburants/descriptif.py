"""Statistiques descriptives et graphiques sur le panel des prix (prix TTC, €/L).

Chaque fonction renvoie un tableau ou une figure matplotlib ; le rapport les affiche
et les commente. Couleurs : une par carburant (palette lisible par les daltoniens),
hausses en rouge et baisses en bleu, types d'enseigne avec les couleurs de cartes.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carburants.cartes import COULEURS_TYPE, metropole, preparer_axes
from carburants.prix import CARBURANTS_ROUTIERS

COULEUR_CARBURANT = {"Gazole": "#D55E00", "E10": "#009E73", "SP95": "#0072B2", "SP98": "#CC79A7"}
COULEUR_HAUSSE, COULEUR_BAISSE = "#C0392B", "#2E86C1"
COULEUR_REMISE = "#FFF8E1"
# Remise TTC de l'État en 2022 (€/L) : 0,18 puis 0,30 puis 0,10
PERIODES_REMISE_ETAT = [
    ("2022-04-01", "2022-08-31", "0,18 €"),
    ("2022-09-01", "2022-11-15", "0,30 €"),
    ("2022-11-16", "2022-12-31", "0,10 €"),
]
CLASSES_CONCURRENTS = [-1, 0, 2, 5, 10, np.inf]
LIBELLES_CONCURRENTS = ["0", "1-2", "3-5", "6-10", "> 10"]


def moyennes_par_station(panel: pd.DataFrame, carburant: str = "Gazole") -> pd.Series:
    """Prix moyen de chaque station sur l'année, pour un carburant."""
    return panel[panel["carburant"] == carburant].groupby("id_station")["prix_ttc"].mean()


# --- Tableaux -------------------------------------------------------------------

def tableau_resume(panel: pd.DataFrame) -> pd.DataFrame:
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


def tableau_changements(panel: pd.DataFrame) -> pd.DataFrame:
    """Par carburant : nombre et ampleur moyenne des hausses et des baisses de prix.

    C'est la première lecture de l'asymétrie : des hausses plus rares mais plus
    fortes que les baisses, ou l'inverse.
    """
    v = panel[panel["changement"]]
    hausse = v["variation"] > 0
    tab = pd.DataFrame({
        "hausses": v[hausse].groupby("carburant").size(),
        "baisses": v[~hausse].groupby("carburant").size(),
        "hausse moyenne (c€)": v[hausse].groupby("carburant")["variation"].mean() * 100,
        "baisse moyenne (c€)": v[~hausse].groupby("carburant")["variation"].mean() * 100,
        "changements par station et par an": v.groupby("carburant").size()
        / panel.groupby("carburant")["id_station"].nunique(),
    })
    return tab.reindex(CARBURANTS_ROUTIERS).round(2)


# --- Figures --------------------------------------------------------------------

def histogrammes_prix(panel: pd.DataFrame) -> plt.Figure:
    """Distribution des prix station x jour, un panneau par carburant, même axe."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    bas, haut = panel["prix_ttc"].quantile([0.001, 0.999])  # axe commun, extrêmes rares exclus
    bornes = np.arange(np.floor(bas * 10) / 10, np.ceil(haut * 10) / 10 + 0.02, 0.02)
    for ax, carb in zip(axes.flat, CARBURANTS_ROUTIERS):
        x = panel.loc[panel["carburant"] == carb, "prix_ttc"]
        ax.hist(x, bins=bornes, color=COULEUR_CARBURANT[carb], edgecolor="white", linewidth=0.3)
        ax.axvline(x.median(), color="black", linewidth=1)
        ax.set_title(f"{carb} (médiane {x.median():.3f} €/L)")
        ax.set_ylabel("Observations station × jour")
    for ax in axes[1]:
        ax.set_xlabel("Prix TTC (€/L)")
    fig.suptitle("Distribution des prix à la pompe")
    fig.tight_layout()
    return fig


def marquer_remises(ax: plt.Axes) -> None:
    """Colore les périodes de remise de l'État et indique leur montant."""
    for debut, fin, montant in PERIODES_REMISE_ETAT:
        ax.axvspan(pd.Timestamp(debut), pd.Timestamp(fin), color=COULEUR_REMISE, zorder=0)
        ax.axvline(pd.Timestamp(debut), color="#E0A800", linewidth=0.8, zorder=0)
        ax.text(pd.Timestamp(debut), 1.0, f" remise {montant}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="#8A6D00")


def serie_prix_moyen(panel: pd.DataFrame) -> plt.Figure:
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


def serie_dispersion(panel: pd.DataFrame) -> plt.Figure:
    """Écart interdécile des prix entre stations (D9 - D1), jour par jour."""
    q = panel.groupby(["jour", "carburant"])["prix_ttc"].quantile([0.1, 0.9]).unstack()
    ecart = (q[0.9] - q[0.1]).unstack() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    marquer_remises(ax)
    for carb in CARBURANTS_ROUTIERS:
        ax.plot(ecart.index, ecart[carb], label=carb, color=COULEUR_CARBURANT[carb], linewidth=1.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Écart D9 - D1 entre stations (c€/L)")
    ax.set_title("Dispersion des prix entre stations")
    ax.legend(loc="lower left")
    return fig


def histogramme_frequence(panel: pd.DataFrame, stations: pd.DataFrame,
                          carburant: str = "Gazole") -> plt.Figure:
    """Nombre de jours de changement de prix par station, selon le type d'enseigne."""
    n = panel[panel["carburant"] == carburant].groupby("id_station")["changement"].sum()
    types = stations.set_index(stations["id_station"].astype(str))["type_enseigne"]
    d = pd.DataFrame({"n": n, "type": types.reindex(n.index)})
    fig, ax = plt.subplots(figsize=(9, 5))
    bornes = np.arange(0, 366, 10)
    for t, sous in d.groupby("type"):
        ax.hist(sous["n"], bins=bornes, histtype="step", linewidth=1.8, color=COULEURS_TYPE.get(t, "black"),
                density=True, label=f"{t} (médiane {sous['n'].median():.0f})")
    ax.set_xlabel(f"Jours de changement du prix du {carburant.lower()} dans l'année")
    ax.set_ylabel("Densité")
    ax.set_title("Fréquence des changements de prix selon le type d'enseigne")
    ax.legend()
    return fig


def histogramme_variations(panel: pd.DataFrame, carburant: str = "Gazole") -> plt.Figure:
    """Distribution des variations journalières non nulles, hausses et baisses."""
    v = panel.loc[(panel["carburant"] == carburant) & panel["changement"], "variation"] * 100
    titre = f"Taille des changements de prix : {(v > 0).sum()} hausses, {(v < 0).sum()} baisses"
    fig, ax = plt.subplots(figsize=(9, 5))
    # axe limité à ±20 c€ : au-delà, les variations sont rares et écraseraient l'histogramme
    bornes = np.arange(-20, 20.5, 0.5)
    ax.hist(v[(v < 0) & (v >= -20)], bins=bornes, color=COULEUR_BAISSE, edgecolor="white", linewidth=0.3, label="baisses")
    ax.hist(v[(v > 0) & (v <= 20)], bins=bornes, color=COULEUR_HAUSSE, edgecolor="white", linewidth=0.3, label="hausses")
    ax.legend()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"Variation journalière du prix du {carburant.lower()} (c€/L)")
    ax.set_ylabel("Nombre de changements")
    ax.set_title(titre)
    return fig


def boites_type_enseigne(panel: pd.DataFrame, stations: pd.DataFrame,
                         carburant: str = "Gazole") -> plt.Figure:
    """Prix moyen annuel des stations, par type d'enseigne."""
    moy = moyennes_par_station(panel, carburant)
    types = stations.set_index(stations["id_station"].astype(str))["type_enseigne"].reindex(moy.index)
    ordre = ["grande surface", "major", "autre enseigne", "inconnue", "non appariée"]
    groupes = [(t, moy[types == t]) for t in ordre if (types == t).any()]
    fig, ax = plt.subplots(figsize=(9, 5))
    boites = ax.boxplot([g for _, g in groupes], tick_labels=[f"{t}\n(n = {len(g)})" for t, g in groupes],
                        showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for boite, (t, _) in zip(boites["boxes"], groupes):
        boite.set_facecolor(COULEURS_TYPE[t])
    ax.set_ylabel(f"Prix moyen annuel du {carburant.lower()} (€/L)")
    ax.set_title("Prix moyen des stations selon le type d'enseigne")
    return fig


def boites_concurrence(panel: pd.DataFrame, stations: pd.DataFrame,
                       carburant: str = "Gazole") -> plt.Figure:
    """Prix moyen annuel des stations selon le nombre de concurrents à 5 km."""
    moy = moyennes_par_station(panel, carburant)
    n = stations.set_index(stations["id_station"].astype(str))["n_concurrents"].reindex(moy.index)
    classe = pd.cut(n, CLASSES_CONCURRENTS, labels=LIBELLES_CONCURRENTS)
    groupes = [(c, moy[classe == c]) for c in LIBELLES_CONCURRENTS]
    fig, ax = plt.subplots(figsize=(9, 5))
    boites = ax.boxplot([g for _, g in groupes], tick_labels=[f"{c}\n(n = {len(g)})" for c, g in groupes],
                        showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for boite, nuance in zip(boites["boxes"], plt.cm.Blues(np.linspace(0.25, 0.9, len(groupes)))):
        boite.set_facecolor(nuance)  # plus foncé = plus de concurrents
    ax.set_xlabel("Stations concurrentes d'une autre enseigne à moins de 5 km")
    ax.set_ylabel(f"Prix moyen annuel du {carburant.lower()} (€/L)")
    ax.set_title("Prix moyen des stations selon la concurrence locale")
    return fig


def carte_prix(panel: pd.DataFrame, stations: pd.DataFrame, carburant: str = "Gazole") -> plt.Figure:
    """Carte du prix moyen annuel de chaque station ; du vert (moins cher) au rouge (plus cher)."""
    moy = moyennes_par_station(panel, carburant).rename("prix")
    st = metropole(stations.assign(id_station=stations["id_station"].astype(str))
                   .join(moy, on="id_station").dropna(subset=["prix"]))
    bas, haut = st["prix"].quantile([0.05, 0.95])  # échelle sur les déciles centraux
    fig, ax = plt.subplots(figsize=(8, 8))
    nuage = ax.scatter(st["longitude"], st["latitude"], c=st["prix"], cmap="RdYlGn_r",
                       vmin=bas, vmax=haut, s=3)
    fig.colorbar(nuage, ax=ax, shrink=0.7, label=f"Prix moyen annuel du {carburant.lower()} (€/L)")
    preparer_axes(ax, f"Prix moyen du {carburant.lower()} par station")
    return fig


def infographie_prix(panel: pd.DataFrame, carburant: str = "Gazole") -> plt.Figure:
    """Vue d'ensemble d'un carburant sur l'année, en deux panneaux alignés sur le temps.

    Haut : prix moyen entre stations, bande moyenne ± un écart-type, bande D1-D9,
    1er et 99e centiles du jour, sommet et creux du prix moyen annotés.
    Bas : part des stations qui augmentent (au-dessus de zéro) ou baissent
    (en dessous) leur prix ce jour-là.
    """
    d = panel[panel["carburant"] == carburant]
    g = d.groupby("jour")["prix_ttc"]
    s = pd.DataFrame({"moy": g.mean(), "et": g.std(), "p01": g.quantile(0.01), "p99": g.quantile(0.99),
                      "d1": g.quantile(0.1), "d9": g.quantile(0.9)})
    stations_du_jour = d.groupby("jour").size()
    part_hausse = d[d["variation"] > 0].groupby("jour").size().reindex(s.index, fill_value=0) / stations_du_jour * 100
    part_baisse = d[d["variation"] < 0].groupby("jour").size().reindex(s.index, fill_value=0) / stations_du_jour * 100

    fig, (haut, bas) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.2]})
    marquer_remises(haut)
    c = COULEUR_CARBURANT[carburant]
    haut.fill_between(s.index, s["d1"], s["d9"], color=c, alpha=0.15, label="D1 - D9 (80 % des stations)")
    haut.fill_between(s.index, s["moy"] - s["et"], s["moy"] + s["et"], color=c, alpha=0.35,
                      label="moyenne ± 1 écart-type")
    haut.plot(s.index, s["moy"], color=c, linewidth=2, label="prix moyen")
    haut.plot(s.index, s["p01"], color=COULEUR_BAISSE, linewidth=0.8, label="1er centile (1 % les moins chères)")
    haut.plot(s.index, s["p99"], color=COULEUR_HAUSSE, linewidth=0.8, label="99e centile (1 % les plus chères)")
    for jour, mot in [(s["moy"].idxmax(), "sommet"), (s["moy"].idxmin(), "creux")]:
        haut.annotate(f"{mot} : {s.loc[jour, 'moy']:.3f} €/L\n{jour:%d/%m/%Y}",
                      xy=(jour, s.loc[jour, "moy"]), xytext=(30, 15 if mot == "sommet" else 30),
                      textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "color": "0.3"})
    haut.set_ylabel("Prix TTC (€/L)")
    haut.set_title(f"{carburant} : prix moyen et dispersion entre stations, jour par jour")
    haut.legend(loc="lower left", fontsize=8, ncols=2)

    bas.bar(s.index, part_hausse, width=1, color=COULEUR_HAUSSE, label="stations en hausse")
    bas.bar(s.index, -part_baisse, width=1, color=COULEUR_BAISSE, label="stations en baisse")
    bas.axhline(0, color="black", linewidth=0.6)
    bas.set_ylabel("Part des stations (%)")
    bas.set_xlabel("Date")
    bas.legend(loc="upper left", fontsize=8, ncols=2)
    fig.tight_layout()
    return fig
