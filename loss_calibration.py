import optuna
import numpy as np
import time
from src.optimizer import CrossoverOptimizer, WayConfig
from src.nodes import ComponentNode

# =====================================================================
# 1. BASE DE DONNÉES DES TESTS (À remplir avec vos mesures)
# =====================================================================
DRIVER_COMBINATIONS = [
    [
        WayConfig("Woofer", r"data\Woofers\FRD\0deg\SIG180-4_0deg.frd", r"data\Woofers\ZMA\SIG180-4.zma"),
        WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\RST28F-4_0deg.frd", r"data\Tweeters\ZMA\RST28F-4.zma")
    ],
    [
        WayConfig("Woofer", r"data\Woofers\FRD\0deg\SIG150-4_0deg.frd", r"data\Woofers\ZMA\SIG150-4.zma"),
        WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\RST28F-4_0deg.frd", r"data\Tweeters\ZMA\RST28F-4.zma")
    ],
    [
        WayConfig("Woofer", r"data\Woofers\FRD\0deg\RS150-8_0deg.frd", r"data\Woofers\ZMA\RS150-8.zma"),
        WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\RST28A-4_0deg.frd", r"data\Tweeters\ZMA\RST28A-4.zma")
    ],
    [
        WayConfig("Woofer", r"data\Woofers\FRD\0deg\DS135-8_0deg.frd", r"data\Woofers\ZMA\DS135-8.zma"),
        WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\ND25FW-4_0deg.frd", r"data\Tweeters\ZMA\ND25FW-4.zma")
    ]
 
]

def get_driver_combinations():
    return [
                [
                    WayConfig("Woofer", r"data\Woofers\FRD\0deg\SIG180-4_0deg.frd", r"data\Woofers\ZMA\SIG180-4.zma"),
                    WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\RST28F-4_0deg.frd", r"data\Tweeters\ZMA\RST28F-4.zma")
                ],
                [
                    WayConfig("Woofer", r"data\Woofers\FRD\0deg\SIG150-4_0deg.frd", r"data\Woofers\ZMA\SIG150-4.zma"),
                    WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\RST28F-4_0deg.frd", r"data\Tweeters\ZMA\RST28F-4.zma")
                ],
                [
                    WayConfig("Woofer", r"data\Woofers\FRD\0deg\RS150-8_0deg.frd", r"data\Woofers\ZMA\RS150-8.zma"),
                    WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\RST28A-4_0deg.frd", r"data\Tweeters\ZMA\RST28A-4.zma")
                ],
                [
                    WayConfig("Woofer", r"data\Woofers\FRD\0deg\DS135-8_0deg.frd", r"data\Woofers\ZMA\DS135-8.zma"),
                    WayConfig("Tweeter", r"data\Tweeters\FRD\0deg\ND25FW-4_0deg.frd", r"data\Tweeters\ZMA\ND25FW-4.zma")
                ]
 
            ]

TARGET_FC = 0.0

# =====================================================================
# 2. LE JUGE HUMAIN (MÉTA-LOSS)
# =====================================================================
def evaluate_human_score(opt, individual):
    """
    Ceci est la vérité absolue. Peu importe les poids testés par Optuna, 
    cette fonction juge le résultat final comme vous le feriez visuellement.
    """
    root = individual['tree']
    res = opt.evaluator.evaluate(root)
    score = 0.0
    
    # 1. Erreur SPL pure (La base du score)
    p_sum = np.zeros_like(opt.freqs, dtype=complex)
    for way in opt.ways:
        p_sum += res.get(way.label, {}).get("P_acoustic", np.zeros_like(opt.freqs))
    spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
    
    # On calcule la variance/erreur par rapport à la cible (Moyenne quadratique pure)
    mse_spl = np.mean((spl_sum[opt.mask_flat] - opt.target_spl)**2)
    score += mse_spl * 10.0 # Facteur d'échelle pour Optuna
    
    # 2. Lignes rouges absolues (Sécurité)
    # Impédance
    Z_in = opt.evaluator.get_impedance(root)
    min_Z = np.min(np.abs(Z_in))
    if min_Z < 3.2:
        score += 1000.0 + (3.2 - min_Z) * 5000.0 # Impardonnable
        
    # Thermique
    V_amp_test = np.full_like(opt.freqs, 28.28, dtype=complex)
    max_p = opt._get_max_power_dissipation(root, V_amp_test)
    if max_p > 20.0:
        score += 1000.0 + (max_p - 20.0) * 100.0 # Impardonnable
        
    # 3. Compacité du filtre
    comps = [n for n in root.get_all_nodes() if isinstance(n, ComponentNode)]
    n_comps = len(comps)
    if n_comps > 8:
        score += (n_comps - 8) * 50.0 # Très punitif
        
    return score

# =====================================================================
# 3. LE MOTEUR OPTUNA
# =====================================================================
def objective(trial):
    # Optuna va générer des poids à tester dans ces fourchettes
    test_weights = {
        'mse_sum': 1.0,           # Reste fixe (point de repère)
        'mean_spl': trial.suggest_float('mean_spl', 0.001, 0.1, log=True),
        'crossover': trial.suggest_float('crossover', 1.0, 20.0),
        'n_comps': 8,             # Reste fixe (objectif)
        
        # Optuna teste les pentes des gradients (Échelle Logarithmique car la plage est vaste)
        'fc_err': trial.suggest_float('fc_err', 10.0, 1000.0, log=True),
        'impedance': trial.suggest_float('impedance', 10.0, 1000.0, log=True),
        'tweeter_low': trial.suggest_float('tweeter_low', 1.0, 200.0, log=True),
        'woofer_attenuation': trial.suggest_float('woofer_attenuation', 100.0, 5000.0, log=True),
        'thermal': trial.suggest_float('thermal', 0.01, 5.0, log=True),
        'components': trial.suggest_float('components', 0.1, 10.0, log=True),
        'resistors': trial.suggest_float('resistors', 0.01, 1.0, log=True)
    }
    
    total_score = 0.0
    driver_combinations = get_driver_combinations()
    
    # On teste ces poids sur TOUS les combos de haut-parleurs
    for config in driver_combinations:
        opt = CrossoverOptimizer(config, target_fc=TARGET_FC, weights=test_weights)
        
        # Entraînement "Rapide" pour tester les poids (ex: 20 générations)
        # Inutile d'aller jusqu'à 100, on veut juste voir si la direction est bonne
        best_ind = opt.run(generations=20, pop_size=100)
        
        # On passe le meilleur individu devant le Juge Humain
        human_score = evaluate_human_score(opt, best_ind)
        total_score += human_score
        
    # On renvoie la moyenne des scores. Optuna cherchera à minimiser cette valeur.
    return total_score / len(DRIVER_COMBINATIONS)

# =====================================================================
# 4. LANCEMENT DE L'ÉTUDE
# =====================================================================
if __name__ == "__main__":
    print("[+] Lancement de la calibration Optuna...")
    start_time = time.time()
    
    # Création de l'étude (direction='minimize' car on veut réduire l'erreur humaine)
    study = optuna.create_study(direction="minimize")
    
    # On lance 50 essais de combinaisons de poids
    # (Attention, cela peut prendre du temps selon le nombre de combos !)
    study.optimize(objective, n_trials=50)
    
    print("\n=========================================")
    print("[!] CALIBRATION TERMINÉE !")
    print(f"Temps total : {(time.time() - start_time) / 60:.1f} minutes")
    print("=========================================\n")
    
    print(">> Copiez-collez ce dictionnaire dans votre fichier optimizer.py :\n")
    best_weights = study.best_params
    best_weights['mse_sum'] = 1.0
    best_weights['n_comps'] = 8
    
    # Formatage propre pour le copier-coller
    print("WEIGHTS = {")
    for k, v in best_weights.items():
        if isinstance(v, float):
            print(f"    '{k}': {v:.4f},")
        else:
            print(f"    '{k}': {v},")
    print("}")