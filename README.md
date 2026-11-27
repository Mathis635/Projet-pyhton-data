# Transmission du prix du pétrole aux prix à la pompe en France

Les prix à la pompe répercutent-ils de la même façon les hausses et les baisses du coût du pétrole, et cela dépend-il du marché local (présence d'une station TotalEnergies, niveau de vie de la commune) ?

Projet du cours « Python pour la data science » (ENSAE 2A).

## Auteurs

Mathis Chartier, Hugues Woto

## Installation

```bash
uv sync
```

## Exécution du rapport

```bash
uv run quarto render rapport/index.qmd
```

Le rapport télécharge les données depuis leurs sources (prix-carburants.gouv.fr, OpenStreetMap, FRED, data.gouv.fr) et produit `rapport/index.html`.

## Arborescence

- `src/carburants/` : logique du projet, un module par étape
- `rapport/` : rapport Quarto
- `data/` : données téléchargées (non versionné)
