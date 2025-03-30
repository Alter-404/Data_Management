# Gestionnaire de Portefeuille Multi-Actifs

Un système complet de gestion de portefeuille pour créer, optimiser et surveiller des portefeuilles d'investissement à travers de multiples classes d'actifs.

## Aperçu

Le Gestionnaire de Portefeuille Multi-Actifs est une application de bureau conçue pour aider les gestionnaires de portefeuille et les investisseurs à créer et optimiser des portefeuilles d'investissement. L'application propose des stratégies d'optimisation de portefeuille pour différents profils de risque, une visualisation interactive des performances, et des outils d'analyse comparative.

## Fonctionnalités

- **Création de Portefeuille**: Interface intuitive pour la création et la gestion de portefeuilles d'investissement.
- **Gestion des Données**: Récupération, stockage et analyse de données de marché pour diverses classes d'actifs.
- **Construction de Portefeuille**: Construction de portefeuille optimisée avec plusieurs stratégies (Faible Risque, Faible Rotation, Actions à Haut Rendement).
- **Comparaison de Portefeuilles**: Outils pour comparer les performances de différents portefeuilles et stratégies.
- **Intégration avec Base de Données**: Stockage sécurisé des portefeuilles, des transactions et des métriques de performance.
- **Backtesting**: Évaluation des stratégies sur des données historiques pour mesurer les performances potentielles.

## Installation

### Prérequis

- Python 3.8+
- SQLite3

### Dépendances

Installez les dépendances requises:

```bash
pip install -r requirements.txt
```

Les principales dépendances incluent:
- numpy
- pandas
- matplotlib
- tkinter
- scipy
- yfinance

## Structure du Projet

```
multi_asset_portfolio_manager/
├── gui/                    # Composants de l'interface graphique (Tkinter)
│   ├── components/         # Composants individuels de l'interface
│   └── app.py              # Point d'entrée principal de l'application
├── src/                    # Fonctionnalités principales
│   ├── data_management/    # Récupération et stockage des données
│   └── portfolio_optimization/ # Stratégies et optimisation de portefeuille
├── outputs/                # Fichiers de sortie et base de données
├── docs/                   # Documentation
│   ├── en/                 # Documentation en anglais
│   └── fr/                 # Documentation en français
└── README.md               # Ce fichier
```

## Utilisation

Pour démarrer l'application, exécutez:

```bash
python -m gui.app
```

## Composants

### Interface Utilisateur (GUI)

L'interface utilisateur se compose de quatre onglets principaux:
- **Portefeuille**: Création et gestion de portefeuilles
- **Données**: Récupération et visualisation de données de marché
- **Construction**: Optimisation et construction de portefeuille
- **Comparaison**: Comparaison des performances entre différents portefeuilles

### Modules Principaux

- **DatabaseManager**: Gère le stockage et la récupération des données
- **DataCollector**: Récupère les données de marché de sources externes
- **PortfolioOptimizer**: Implémente diverses stratégies d'optimisation
- **PortfolioVisualizer**: Génère des visualisations pour l'analyse de portefeuille

## Stratégies de Portefeuille

### Stratégie à Faible Risque
Minimise la volatilité du portefeuille tout en maintenant des rendements acceptables, idéale pour les investisseurs conservateurs.

### Stratégie à Faible Rotation
Réduit les coûts de transaction en minimisant le nombre de transactions, adaptée aux portefeuilles sensibles aux coûts.

### Stratégie Actions à Haut Rendement
Maximise les rendements en se concentrant sur les actions à fort dividende et à croissance potentielle.

## Schéma de Base de Données

L'application utilise une base de données SQLite avec les tables principales suivantes:
- **Products**: Stocke les informations sur les actifs disponibles
- **Clients**: Informations sur les clients
- **Managers**: Informations sur les gestionnaires de portefeuille
- **Portfolios**: Définitions des portefeuilles
- **Deals**: Transactions exécutées pour les portefeuilles
- **PerformanceMetrics**: Métriques de performance pour les portefeuilles

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Contributeurs

- [Nom du Contributeur 1](https://github.com/utilisateur1)
- [Nom du Contributeur 2](https://github.com/utilisateur2)

Pour contribuer au projet, veuillez consulter notre [Guide de Contribution](contributing.md). 