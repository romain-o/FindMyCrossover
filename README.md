# crossover_ai — conception automatique de filtres 2 voies (Woofer/Mid) par IA

Système d'apprentissage qui **choisit une topologie** de crossover adaptée à un couple
de haut-parleurs (à partir de leurs FRD/ZMA), puis **règle les valeurs** des composants
par optimisation différentiable sur GPU. Conçu pour s'entraîner lui-même, en réutilisant
votre code existant (`src/nodes.py`, `catalog_manager`, `schematic`, `vituix_exporter`).

## Principe (optimisation bi-niveau)
- **Boucle externe (discrète)** : une politique autorégressive pose la topologie token
  par token (deux branches combinées en `Parallel`), conditionnée par un encodeur des
  courbes des drivers.
- **Boucle interne (continue)** : `torch_sim` (portage différentiable de votre
  `evaluator.py`) optimise les valeurs des composants par Adam + gradients exacts,
  des centaines de candidats en parallèle (CUDA-ready).
- **Apprentissage** : imitation d'un professeur (templates évalués, ou votre GA) puis
  fine-tuning RL avec récompense = `-fitness`.

## Données attendues
```
data/
  Woofer/   X.frd  X.zma   (FRD = freq, SPL dB, phase deg ; ZMA = freq, |Z|, phase deg)
  Mid/      Y.frd  Y.zma   (commentaires '*' ignorés, comme dans vos fichiers)
  Tweeter/  ...            (ignoré pour le 2 voies Woofer/Mid)
```
Chaque driver = une paire FRD+ZMA de même nom de base. Le pipeline rééchantillonne tout
sur une grille `geomspace(20, 20000, 400)` (dB + phase déroulée), exactement comme votre
`_prepare_driver`. Pointez le pipeline via `CROSSOVER_DATA=/chemin/vers/data`.

## Gestion des données / entraînement
- **Split honnête** : des drivers réels sont réservés à la validation et **jamais
  augmentés** (mesure de généralisation réelle).
- **Augmentation intra-classe** (woofer×woofer uniquement) : interpolation en dB+phase,
  décalage de niveau, warp fréquentiel léger, bruit de mesure. Multiplie une poignée de
  HP réels en milliers de tâches plausibles.
- **Cache disque** des courbes rééchantillonnées (`.cache_crossover/`).

## Installation
```
pip install torch numpy scipy        # + vos deps existantes (pandas, schemdraw, matplotlib)
# placer le dossier crossover_ai/ À CÔTÉ de votre dossier src/
```

## Ordre d'exécution
```bash
python -m crossover_ai.smoke_test        # vérifie toute la chaîne (données synthétiques)
python -m crossover_ai.train_imitation   # Phase B : behavior cloning (warm start)
python -m crossover_ai.train_rl          # Phase C : fine-tuning RL
```

## Fichiers
| Fichier | Rôle |
|---|---|
| `torch_sim.py` | simulateur différentiable batché (= evaluator.py en torch, CUDA) |
| `torch_reward.py` | reward différentiable (sous-ensemble fidèle de `fitness()`) |
| `data_pipeline.py` | chargement FRD/ZMA, resample, augmentation, tâches, cache |
| `tree_actions.py` | grammaire tokens ↔ arbre Series/Parallel/Shunt + templates |
| `driver_encoder.py` | CNN1D courbes → embedding système |
| `policy.py` | politique autorégressive (construction de topologie) |
| `inner_optim.py` | optimisation des valeurs (Adam batché, vrai reward) |
| `envs.py` | environnement : 1 topologie = 1 épisode |
| `ga_harvester.py` | professeur (templates évalués) + stub `GATeacher` |
| `train_imitation.py` / `train_rl.py` | entraînements |
| `bridge.py` | conversion vers/depuis vos arbres `nodes.py` (+ snap/schematic/vituix) |
| `config.py` | réglages centraux |

## Brancher votre GA comme professeur (qualité maximale)
Dans `ga_harvester.GATeacher` : lancez `src.optimizer.CrossoverOptimizer.run()`, récupérez
le champion (arbre `nodes.py`), puis `bridge.tree_to_tokens(champion)` → `(low_ids, high_ids)`.

## Limites / honnêteté
- Le reward porté est un **sous-ensemble** de votre `fitness` (les pénalités médium 3 voies,
  thermique fine, etc. restent à porter pour le 3 voies).
- La grammaire couvre ordres 1–4 + Zobel/L-pad/notch simple ; les topologies exotiques
  hors grammaire ne sont pas atteignables (par conception, pour garantir la validité).
- Battre un GA bien réglé est l'objectif **ambitieux** ; le gain immédiat certain est la
  proposition instantanée + l'inner-loop différentiable (≫ scipy diff. finies).
