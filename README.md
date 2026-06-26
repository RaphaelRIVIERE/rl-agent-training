# Eagle-1 : Pilote automatique LunarLander

Projet OpenClassrooms AI Engineer. Agent DQN entraîné sur LunarLander-v3 via Stable-Baselines3, exposé via une API FastAPI et visualisé avec Streamlit.

## Prérequis

- Python 3.11
- [UV](https://docs.astral.sh/uv/) pour la gestion de l'environnement virtuel

## Installation

```bash
uv sync
```

## Lancer l'API

L'API doit être démarrée en premier. Le GUI et le dashboard en dépendent.

```bash
uv run uvicorn api.main:app --reload
```

L'API est disponible sur `http://localhost:8000`.
La documentation interactive (Swagger) est sur `http://localhost:8000/docs`.

## Lancer le GUI

Dans un second terminal, après avoir démarré l'API :

```bash
uv run streamlit run gui/app.py
```

Le GUI joue un épisode complet en appelant l'API à chaque step et affiche l'animation de l'atterrissage en temps réel.

## Lancer le tableau de bord

Dans un terminal séparé, après avoir démarré l'API :

```bash
uv run streamlit run dashboard/app.py
```

Le dashboard permet de lancer plusieurs épisodes via la sidebar et affiche les courbes de récompense, le taux de succès et le détail filtrable par épisode.

## Lancer les tests

```bash
uv run pytest tests/ -v
```

3 tests couverts : `/health`, observation valide (8 floats), observation invalide (mauvaise taille).

## Structure du projet

```
api/          chargement du modèle et endpoint /play
gui/          animation de l'atterrissage en temps réel
dashboard/    courbes de récompense et métriques
tests/        tests unitaires de l'API
notebook/     exploration, entraînement et optimisation (Colab)
  models/     modèles sauvegardés (dqn_baseline.zip, dqn_final.zip)
```

## Variable d'environnement

Par défaut l'API charge le modèle depuis `notebook/models/dqn_final.zip`. Pour utiliser un autre modèle :

```bash
MODEL_PATH=chemin/vers/modele.zip uv run uvicorn api.main:app --reload
```
