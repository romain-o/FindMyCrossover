# FindMyCrossover - Contexte et Lignes Directrices du Projet

## 1. Approche Algorithmique : Programmation Génétique (GP)
Nous utilisons une représentation en arbre binaire pour faire évoluer la topologie du circuit (et pas seulement les valeurs).
- Les nœuds internes sont des opérateurs de connexion.
- Les feuilles (terminaux) sont les composants physiques et les haut-parleurs.

## 2. Structure de l'Arbre (Nœuds)
- `SeriesNode(left, right)` : Connecte deux sous-arbres en série.
- `ParallelNode(left, right)` : Connecte deux sous-arbres en parallèle (flottant, partageant les mêmes bornes d'entrée/sortie).
- `ShuntNode(component)` : Branche un composant en dérivation vers la masse (GND), permettant de créer les pentes d'atténuation.
- `ComponentNode` : Classe mère pour `Resistor`, `Capacitor` et `Inductor`, possédant une valeur numérique modifiable par mutation.
- `DriverNode(label, frd_path, zma_path)` : Représente le haut-parleur. Il charge les fichiers de mesure réels (.zma pour l'impédance, .frd pour la réponse acoustique).

## 3. Évaluation du Circuit (Sans SPICE)
Pour des raisons de performance, nous n'utilisons pas de simulateur externe comme SPICE.
- L'évaluation se fait via des calculs matriciels en nombres complexes avec `numpy` sur un vecteur de fréquences (ex: de 20 Hz à 20 kHz).
- Les fichiers `.zma` et `.frd` sont convertis en tableaux `numpy` de nombres complexes (Magnitude + Phase via la formule d'Euler).
- L'impédance de chaque composant classique (R, L, C) est également calculée sous forme de tableau complexe vectorisé (Z_R, Z_L, Z_C).
- L'arbre est parcouru de manière récursive (bottom-up) pour calculer l'impédance équivalente de chaque bloc, puis utiliser la règle du pont diviseur de tension pour trouver le signal électrique (complexe) aux bornes de chaque `DriverNode`.
- La réponse finale de chaque voie est le produit : `Tension_aux_bornes(f) * Reponse_Acoustique_FRD(f)`.

## 4. État Actuel du Projet et Prochaine Étape
- Nous venons de valider la structure globale et l'abandon de SPICE au profit de l'évaluation matricielle complexe avec Numpy.
- Le développement principal se fait dans `circuit.py`.
