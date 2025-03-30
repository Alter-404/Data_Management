# Documentation du Schéma de Base de Données

Ce document fournit une vue d'ensemble complète du schéma de base de données utilisé dans l'application Gestionnaire de Portefeuille Multi-Actifs. L'application utilise SQLite comme système de gestion de base de données.

## Aperçu

La base de données se compose de plusieurs tables qui stockent des informations sur les produits (actifs), les clients, les gestionnaires de portefeuille, les portefeuilles, les transactions (deals), les positions de portefeuille et les métriques de performance. Les tables sont conçues pour maintenir l'intégrité référentielle grâce aux relations de clé étrangère.

## Définitions des Tables

### Products (anciennement Assets)

Stocke les informations sur les produits financiers disponibles pour le trading dans le système.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour le produit |
| symbol | TEXT | NOT NULL | Symbole boursier pour le produit |
| name | TEXT | | Nom complet du produit |
| type | TEXT | | Type de produit (ex., Action, Obligation, ETF) |
| region | TEXT | | Région géographique du produit |
| sector | TEXT | | Secteur industriel du produit |
| is_active | INTEGER | DEFAULT 1 | Si le produit est actif et disponible pour le trading |

### Managers (anciennement PortfolioManagers)

Stocke les informations sur les gestionnaires de portefeuille qui gèrent les portefeuilles des clients.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour le gestionnaire |
| name | TEXT | NOT NULL | Nom du gestionnaire de portefeuille |
| email | TEXT | | Adresse email du gestionnaire de portefeuille |
| phone | TEXT | | Numéro de téléphone du gestionnaire de portefeuille |
| date_joined | TEXT | | Date à laquelle le gestionnaire a rejoint |

### Clients

Stocke les informations sur les clients qui possèdent des portefeuilles.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour le client |
| name | TEXT | NOT NULL | Nom du client |
| risk_profile | TEXT | | Niveau de tolérance au risque du client |
| email | TEXT | | Adresse email du client |
| phone | TEXT | | Numéro de téléphone du client |

### Portfolios

Stocke les informations sur les portefeuilles d'investissement gérés dans le système.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour le portefeuille |
| client_id | INTEGER | FOREIGN KEY | Référence au client qui possède le portefeuille |
| manager_id | INTEGER | FOREIGN KEY | Référence au gestionnaire qui gère le portefeuille |
| name | TEXT | NOT NULL | Nom du portefeuille |
| strategy | TEXT | | Stratégie d'investissement appliquée au portefeuille |
| asset_universe | TEXT | | Univers d'actifs disponibles pour ce portefeuille |
| creation_date | TEXT | | Date à laquelle le portefeuille a été créé |
| last_updated | TEXT | | Date à laquelle le portefeuille a été mis à jour pour la dernière fois |
| cash_balance | REAL | DEFAULT 0.0 | Solde de trésorerie actuel dans le portefeuille |

### Deals (anciennement Trades)

Stocke les informations sur toutes les transactions exécutées dans le système.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour la transaction |
| portfolio_id | INTEGER | FOREIGN KEY | Référence au portefeuille où la transaction a été exécutée |
| product_id | INTEGER | FOREIGN KEY | Référence au produit négocié |
| date | TEXT | NOT NULL | Date à laquelle la transaction a été exécutée |
| action | TEXT | NOT NULL | Achat ou Vente |
| shares | REAL | NOT NULL | Nombre d'actions négociées |
| price | REAL | NOT NULL | Prix par action au moment de la transaction |
| amount | REAL | NOT NULL | Valeur monétaire totale de la transaction |
| period | TEXT | | Période de trading (ex., "entraînement", "évaluation") |

### MarketData

Stocke les données de marché historiques pour les produits.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour l'enregistrement de données de marché |
| product_id | INTEGER | FOREIGN KEY | Référence au produit |
| date | TEXT | NOT NULL | Date de l'enregistrement de données de marché |
| open | REAL | | Prix d'ouverture |
| high | REAL | | Prix le plus élevé pendant la période |
| low | REAL | | Prix le plus bas pendant la période |
| close | REAL | | Prix de clôture |
| volume | INTEGER | | Volume de trading |
| adjusted_close | REAL | | Prix de clôture ajusté |

### PerformanceMetrics

Stocke les métriques de performance pour les portefeuilles au fil du temps.

| Nom de Colonne | Type de Données | Contraintes | Description |
|----------------|-----------------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Identifiant unique pour l'enregistrement des métriques de performance |
| portfolio_id | INTEGER | FOREIGN KEY | Référence au portefeuille |
| date | TEXT | NOT NULL | Date de l'enregistrement de performance |
| total_value | REAL | | Valeur totale du portefeuille y compris la trésorerie |
| daily_return | REAL | | Pourcentage de rendement quotidien |
| cumulative_return | REAL | | Pourcentage de rendement cumulatif depuis le début |
| volatility | REAL | | Volatilité du portefeuille (écart-type des rendements) |
| sharpe_ratio | REAL | | Ratio de Sharpe (rendement ajusté au risque) |
| max_drawdown | REAL | | Drawdown maximal (plus grande baisse de pic à creux) |
| winning_days | INTEGER | | Nombre de jours avec des rendements positifs |

## Relations

Le diagramme suivant illustre les relations entre les tables:

```
Clients (1) ---> (N) Portfolios (1) ---> (N) Deals
                      |
                      | (1)
                      v
                      (N)
Managers (1) -----> Portfolios (1) ---> (N) PerformanceMetrics
                      |
                      | (N)
                      v
Products (1) -------> (N) Deals
      |
      | (1)
      v
      (N)
MarketData
```

## Exemples de Requêtes SQL

### Création d'un Nouveau Portefeuille

```sql
INSERT INTO Portfolios (client_id, manager_id, name, strategy, asset_universe, creation_date, cash_balance)
VALUES (1, 2, 'Portefeuille de Croissance', 'Actions à Haut Rendement', 'Actions US', '2023-01-15', 100000.0);
```

### Récupération des Performances de Portefeuille

```sql
SELECT p.name, pm.date, pm.total_value, pm.daily_return, pm.cumulative_return, pm.volatility, pm.sharpe_ratio
FROM Portfolios p
JOIN PerformanceMetrics pm ON p.id = pm.portfolio_id
WHERE p.id = 1
ORDER BY pm.date DESC;
```

### Récupération de Toutes les Transactions pour un Portefeuille

```sql
SELECT d.date, pr.symbol, d.action, d.shares, d.price, d.amount
FROM Deals d
JOIN Products pr ON d.product_id = pr.id
WHERE d.portfolio_id = 1
ORDER BY d.date DESC;
```

### Calcul de l'Allocation de Portefeuille

```sql
SELECT pr.symbol, pr.type, pr.sector,
       SUM(CASE WHEN d.action = 'BUY' THEN d.shares ELSE -d.shares END) as total_shares,
       SUM(CASE WHEN d.action = 'BUY' THEN d.amount ELSE -d.amount END) as total_invested
FROM Deals d
JOIN Products pr ON d.product_id = pr.id
WHERE d.portfolio_id = 1
GROUP BY pr.id
HAVING total_shares > 0;
```

### Obtenir les Portefeuilles d'un Gestionnaire

```sql
SELECT p.id, p.name, c.name as client_name, p.strategy, p.asset_universe, p.cash_balance
FROM Portfolios p
JOIN Clients c ON p.client_id = c.id
WHERE p.manager_id = 1;
```

## Initialisation de la Base de Données

La base de données est initialisée avec des produits prédéfinis, y compris des symboles boursiers courants comme AAPL, MSFT, GOOGL, etc. L'initialisation crée toutes les tables nécessaires avec des contraintes de clé étrangère appropriées et les remplit avec des données initiales selon les besoins.

## Migrations

Lors de la modification du schéma de base de données, il est important de créer des scripts de migration qui gèrent:

1. La création de nouvelles tables
2. L'ajout de nouvelles colonnes aux tables existantes
3. Le renommage de tables ou de colonnes
4. La mise à jour des relations de clé étrangère
5. La migration des données de l'ancien schéma vers le nouveau schéma

Cela garantit que les données existantes sont préservées lorsque le schéma change.

## Considérations de Performance

Pour une performance optimale, la base de données inclut des index sur:

- `symbol` dans la table Products
- `portfolio_id` et `date` dans la table Deals
- `product_id` et `date` dans la table MarketData
- `portfolio_id` et `date` dans la table PerformanceMetrics

Ces index améliorent les performances des requêtes pour les opérations courantes comme la récupération de données historiques, le calcul de métriques de portefeuille et la génération de rapports. 