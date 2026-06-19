"""
config.py — Configuration centrale du projet crossover_ai.
Tout ce qui est "réglage global" vit ici pour éviter les constantes éparpillées.
"""
from __future__ import annotations
import os
import torch

# ------------------------------------------------------------------ device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ données
# Arborescence supportée (récursive + tolérante singulier/pluriel/casse) :
#   data/
#     Woofers/FRD/0deg/nom.frd   +   Woofers/ZMA/0deg/nom.zma   (votre cas)
#   ... ou bien la forme plate :
#     Woofer/nom.frd  +  Woofer/nom.zma
# L'appariement se fait par NOM DE BASE de fichier (le même pour le .frd et le .zma).
DATA_ROOT = os.environ.get("CROSSOVER_DATA", "data")

# Pour le 2-voies : le Woofer est la voie BASSE, le Mid la voie HAUTE.
# (les noms sont résolus de façon tolérante : "Woofers"/"Woofer", casse indifférente)
ROLE_LOW = os.environ.get("CROSSOVER_LOW", "Woofers")     # voie basse
ROLE_HIGH = os.environ.get("CROSSOVER_HIGH", "Tweeters")  # voie haute (Tweeters ou Mids)
CLASSES = [ROLE_LOW, ROLE_HIGH]

# Si plusieurs angles de mesure existent (0deg, 15deg, ...), on ne garde que l'axe.
ON_AXIS_TAG = "0deg"

FRD_EXTS = (".frd", ".txt")
ZMA_EXTS = (".zma",)

# Noms de dossiers réels (override si différents des noms logiques).
# Ex. arborescence : data/Woofers/FRD/0deg/*.frd  +  data/Woofers/ZMA/0deg/*.zma
# Laisser None : la découverte matche par préfixe (Woofer -> "Woofers") automatiquement.
CLASS_DIRS = {
    ROLE_LOW: os.environ.get("CROSSOVER_LOW_DIR"),    # ex: "Woofers"
    ROLE_HIGH: os.environ.get("CROSSOVER_HIGH_DIR"),  # ex: "Mids"
}

CACHE_DIR = os.environ.get("CROSSOVER_CACHE", ".cache_crossover")

# ------------------------------------------------------------------ grille fréquentielle
F_MIN, F_MAX = 20.0, 20000.0
N_FREQ = 400                          # comme le mode "exploration" de votre optimizer

# ------------------------------------------------------------------ bornes composants (= optimizer.py)
BOUNDS = {"R": (0.1, 500.0), "C": (0.1e-6, 300e-6), "L": (0.05e-3, 5e-3)}

# Valeurs d'initialisation par type (pour les slots créés par la grammaire)
INIT_VALUE = {"R": 8.0, "C": 6.8e-6, "L": 0.8e-3}

# ------------------------------------------------------------------ complexité
# Nombre MAX de composants PAR BRANCHE (voie). Total crossover ~ 2x cette valeur.
# L'IA exploite ce budget de façon ADAPTATIVE (parcimonie du reward) : peu de
# composants pour des drivers simples, plus pour des drivers difficiles.
MAX_COMPONENTS_PER_BRANCH = int(os.environ.get("CROSSOVER_MAX_COMP_BRANCH", "6"))

# ------------------------------------------------------------------ modèle électrique (= evaluator.py)
ESR_C = 0.01            # ESR série condensateur
DCR_COEFF = 400.0       # DCR self ~ value*400

# ------------------------------------------------------------------ poids du reward (miroir de WEIGHTS)
WEIGHTS = {
    "mse_sum": 1.0,
    "crossover": 3.25,          # sur-pondération de la zone de coupure
    "overshoot": 10.0,          # anti-annulation de phase / bosse acoustique
    "impedance": 142.9,
    "high_low_leak": 48.0,      # le HP haut ne doit pas jouer en bas
    "low_high_leak": 10.0,      # le HP bas ne doit pas jouer en haut
    "components": 0.15,
    "n_comps_free": 6,          # composants "gratuits" ; au-delà -> pénalité croissante
    "resistors": 0.3,
    "thermal": 20.0,            # pénalité de dissipation dans les résistances (= GA)
}

# ------------------------------------------------------------------ puissance / thermique (= optimizer.py)
POWER_TEST_V = 28.28    # tension de test (28.28 V RMS = 100 W sur 8 Ω)
RES_POWER_MAX = 17.0    # W : une résistance ne doit pas dissiper plus que ça
WASTE_FRAC_MAX = 0.20   # < 1 kHz : pas plus de 20% de la puissance HP gaspillée en R
WASTE_FREQ_MAX = 1000.0 # bande où l'on surveille le gaspillage (graves)

# Cible SPL : on vise (sensibilité mini des 2 voies) - offset
TARGET_OFFSET_DB = 0.0

# Bande utile par défaut (affinée automatiquement par voie dans data_pipeline)
BAND_MIN, BAND_MAX = 80.0, 18000.0

# ------------------------------------------------------------------ surcharge config.json
# Si un config.json existe (écrit par app.py), il surcharge les réglages ci-dessus.
# Permet de régler les poids / la complexité depuis l'interface graphique.
try:
    import json as _json
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as _f:
            _cfg = _json.load(_f)
        WEIGHTS.update(_cfg.get("weights", {}))
        TARGET_OFFSET_DB = float(_cfg.get("spl_settings", {}).get("target_offset_db", TARGET_OFFSET_DB))
        if "complexity" in _cfg:
            MAX_COMPONENTS_PER_BRANCH = int(_cfg["complexity"].get("max_comp_per_branch",
                                                                   MAX_COMPONENTS_PER_BRANCH))
except Exception as _e:
    print(f"[config] config.json ignoré ({_e})")