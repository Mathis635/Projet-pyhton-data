"""Cartes des stations-service (France métropolitaine)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Emprise de la France métropolitaine (degrés)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 41.0, 51.5, -5.5, 10.0
COULEURS_TYPE = {
    "grande surface": "tab:green",
    "major": "tab:red",
    "autre enseigne": "tab:blue",
    "inconnue": "tab:orange",
    "non appariée": "lightgrey",
}


def metropole(stations: pd.DataFrame) -> pd.DataFrame:
    """Garde les stations dont les coordonnées tombent en France métropolitaine."""
    return stations[
        stations["latitude"].between(LAT_MIN, LAT_MAX)
        & stations["longitude"].between(LON_MIN, LON_MAX)
    ]


def preparer_axes(ax: plt.Axes, titre: str) -> None:
    """Axes nommés, titre, et rapport d'aspect corrigé de la latitude moyenne."""
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(titre)
    # 1° de longitude vaut cos(latitude) × 1° de latitude : sans correction la carte est aplatie
    ax.set_aspect(1 / np.cos(np.radians((LAT_MIN + LAT_MAX) / 2)))


def carte_types_enseigne(stations: pd.DataFrame) -> plt.Figure:
    """Stations colorées par type d'enseigne ; l'effectif de chaque type figure en légende."""
    st = metropole(stations)
    fig, ax = plt.subplots(figsize=(8, 8))
    for type_enseigne, couleur in COULEURS_TYPE.items():
        d = st[st["type_enseigne"] == type_enseigne]
        ax.scatter(d["longitude"], d["latitude"], s=2, c=couleur,
                   label=f"{type_enseigne} ({len(d)})")
    preparer_axes(ax, "Stations-service selon le type d'enseigne (appariement OpenStreetMap)")
    ax.legend(markerscale=5, loc="lower left")
    return fig