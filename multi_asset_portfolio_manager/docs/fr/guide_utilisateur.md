# Guide d'Utilisateur

Ce guide fournit des instructions détaillées sur l'utilisation efficace de l'application Multi-Asset Portfolio Manager.

## Premiers Pas

Après avoir installé les dépendances, vous pouvez démarrer l'application en exécutant:

```bash
python -m main
```

L'application s'ouvrira avec l'interface principale contenant quatre onglets:
- Portefeuille
- Données
- Construction
- Comparaison

## Onglet Portefeuille

L'onglet Portefeuille vous permet de créer et gérer des portefeuilles d'investissement.

### Création d'un Nouveau Portefeuille

1. Cliquez sur le bouton "Nouveau Portefeuille"
2. Remplissez les informations requises:
   - Nom du Portefeuille: Un nom unique pour votre portefeuille
   - Client: Sélectionnez un client existant ou créez-en un nouveau
   - Stratégie: Choisissez parmi Faible Risque, Faible Rotation, ou Actions à Haut Rendement
   - Univers d'Actifs: Sélectionnez parmi les univers d'actifs disponibles
3. Cliquez sur "Créer" pour créer le portefeuille

### Gestion des Portefeuilles

- **Voir le Portefeuille**: Sélectionnez un portefeuille dans la liste déroulante pour voir ses détails
- **Modifier le Portefeuille**: Cliquez sur le bouton "Modifier" pour changer les paramètres du portefeuille
- **Supprimer le Portefeuille**: Cliquez sur le bouton "Supprimer" pour enlever un portefeuille

## Onglet Données

L'onglet Données est utilisé pour récupérer et gérer les données de marché pour différents actifs.

### Récupération des Données de Marché

1. Sélectionnez l'Univers d'Actifs dans la liste déroulante (ex. Actions US, Actions Mondiales, Cryptomonnaies)
2. Sélectionnez les actifs spécifiques pour lesquels vous souhaitez récupérer des données
3. Définissez la période en utilisant le sélecteur de période
4. Cliquez sur "Récupérer Données" pour obtenir les données de marché des sources externes

### Visualisation des Données de Marché

- Les données récupérées sont affichées dans un tableau avec les dates et les informations de prix
- Les graphiques montrent l'historique des prix et les rendements pour les actifs sélectionnés
- L'onglet Rendements montre les rendements quotidiens et cumulatifs calculés

### Gestion des Données

- Les données sont automatiquement stockées dans la base de données pour une utilisation future
- Vous pouvez rafraîchir les données existantes en utilisant le bouton "Rafraîchir"
- Exportez les données en utilisant le bouton "Exporter" (format CSV)

## Onglet Construction

L'onglet Construction vous permet de construire et d'optimiser des portefeuilles basés sur les stratégies sélectionnées.

### Construction d'un Portefeuille

1. Sélectionnez un portefeuille dans la liste déroulante
2. Choisissez les paramètres de construction:
   - Capital Initial: Montant d'investissement initial
   - Période d'Entraînement: Période pour l'entraînement de la stratégie
   - Période d'Évaluation: Période pour tester la stratégie
   - Positions Maximales: Nombre maximum d'actifs à détenir
3. Cliquez sur "Construire Portefeuille" pour lancer l'optimisation

### Analyse de Portefeuille

Après la construction, les informations suivantes sont affichées:

- **Performance**: Valeur du portefeuille au fil du temps avec les métriques clés
- **Positions**: Positions actuelles du portefeuille avec les pondérations
- **Transactions**: Historique des transactions montrant les achats et les ventes
- **Rendements**: Rendements quotidiens et cumulatifs
- **Métriques**: Métriques de performance comme le ratio de Sharpe, la volatilité et le drawdown

### Sauvegarde des Résultats

- Les résultats sont automatiquement sauvegardés dans la base de données
- Vous pouvez sauvegarder manuellement des résultats spécifiques en utilisant le bouton "Sauvegarder Résultats"

## Onglet Comparaison

L'onglet Comparaison vous permet de comparer plusieurs portefeuilles et leurs métriques de performance.

### Sélection de Portefeuilles pour Comparaison

1. Dans la liste des Portefeuilles Disponibles, sélectionnez un ou plusieurs portefeuilles
2. Cliquez sur "Ajouter >" pour les déplacer vers la liste des Portefeuilles Sélectionnés
3. Définissez la période pour la comparaison
4. Sélectionnez les métriques que vous souhaitez comparer
5. Cliquez sur "Comparer Portefeuilles" pour générer la comparaison

### Vues de Comparaison

- **Séries Temporelles**: Montre la valeur du portefeuille ou les rendements au fil du temps pour tous les portefeuilles sélectionnés
- **Graphique à Barres**: Compare des métriques spécifiques comme le rendement total, la volatilité, le ratio de Sharpe
- **Risque-Rendement**: Graphique de dispersion montrant le profil risque-rendement de chaque portefeuille
- **Tableau de Synthèse**: Tableau détaillé des métriques pour tous les portefeuilles

## Stratégies de Portefeuille

L'application prend en charge trois principales stratégies d'optimisation de portefeuille:

### Stratégie à Faible Risque

- Se concentre sur la minimisation de la volatilité du portefeuille
- Priorise les actifs avec une volatilité historique plus faible
- Convient aux investisseurs conservateurs

### Stratégie à Faible Rotation

- Limite les échanges à un maximum de deux transactions par mois
- Utilise une sélection basée sur le momentum pour prioriser les transactions
- Réduit les coûts de transaction et les implications fiscales

### Stratégie Actions à Haut Rendement

- Maximise les rendements sans contraintes sur la volatilité ou la rotation
- Se concentre sur le momentum ajusté au risque pour la sélection d'actifs
- Convient aux investisseurs agressifs cherchant des rendements élevés

## Conseils et Bonnes Pratiques

1. **Commencez par les Données**: Assurez-vous toujours d'avoir suffisamment de données de marché avant de construire des portefeuilles
2. **Comparez les Stratégies**: Utilisez l'onglet Comparaison pour évaluer différentes stratégies
3. **Mises à Jour Régulières**: Rafraîchissez régulièrement les données de marché pour une optimisation précise du portefeuille
4. **Sélection de Stratégie**: Choisissez des stratégies qui s'alignent avec vos objectifs d'investissement:
   - Faible Risque pour la préservation du capital
   - Faible Rotation pour l'efficacité fiscale
   - Actions à Haut Rendement pour des rendements maximaux
5. **Périodes**: Utilisez des périodes d'entraînement et d'évaluation appropriées:
   - Entraînement: Des périodes plus longues capturent davantage de cycles de marché
   - Évaluation: Des périodes récentes testent la performance de la stratégie dans les conditions actuelles 