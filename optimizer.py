import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from nodes import DriverNode, SeriesNode, ParallelNode, ShuntNode, Capacitor, Inductor, Resistor, Node, ComponentNode
from evaluator import CircuitEvaluator
from mutator import TreeMutator
from scipy.optimize import minimize
import itertools
from schematic import SchematicRenderer
import time

# Multiprocessing
from concurrent.futures import ProcessPoolExecutor

BOUNDS_R = (0.1, 50.0)
BOUNDS_C = (0.1e-6, 150e-6)
BOUNDS_L = (0.05e-3, 15e-3)

E24_SERIES = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 
                       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])

WEIGHTS = {
                'mse_sum': 1.0,      # Poids de base (référence)
                'ripple': 2.0,       # Le ripple est 2x plus puni que la pente globale
                'phase': 0.0,        # La phase départage, mais ne doit pas détruire le SPL
                'n_comps': 14,      # Nombre de composants avant pénalité
                'components': 0.05,   # 5% du score actuel en pénalité par composant au-delà du seuil
                'mean_spl': 0.0,     # Poids pour la moyenne du SPL
                'crossover': 5,  # Poids pour la pénalité de croisement (désactivé pour l'instant)
                'resistors': 0.08    # Poids pour les résistances (favorise les solutions sans résistances)
            }

def snap_to_e24(val):
    """Arrondit une valeur mathématique à la valeur E24 la plus proche."""
    if val <= 0: return val
    power = np.floor(np.log10(val))
    norm = val / (10 ** power)
    idx = np.abs(E24_SERIES - norm).argmin()
    return E24_SERIES[idx] * (10 ** power)

class WayConfig:
    """Configuration d'une voie acoustique (Grave, Médium, Aigu, etc.)"""
    def __init__(self, label, frd_path, zma_path, target_type='LP', order=4, z_offset_m=0.0):
        self.label = label
        self.frd_path = frd_path
        self.zma_path = zma_path
        self.target_type = target_type # 'LP', 'BP', 'HP'
        self.order = order
        self.z_offset_m = z_offset_m
        self.driver = DriverNode(label, frd_path, zma_path)

class CrossoverOptimizer:
    def __init__(self, ways_configs):
        self.freqs = np.geomspace(20, 20000, 400)
        self.ways = ways_configs
        self.evaluator = CircuitEvaluator(self.freqs)
        self.mutator = TreeMutator(
            prob_value_mut=0.2, 
            prob_type_mut=0.2, 
            prob_topology_mut=0.2, 
            prob_add_node=0.30,
            prob_remove_node=0.10
        )
                            
        self.ripple_mask = (self.freqs >= 400) & (self.freqs <= 12000)
        self.mask_ref = (self.freqs > 300) & (self.freqs < 1000)
        
        # Préparation des drivers (Interpolation + Z-Offset)
        for way in self.ways:
            self._prepare_driver(way)
        
        # =========================================================
        # NOUVEAU : Verrouillage de la cible SPL sur le Woofer brut
        # =========================================================
        raw_w_mag = np.abs(self.ways[0].driver.H_acoustic)
        raw_w_spl = 20 * np.log10(raw_w_mag + 1e-12)
        
        # On calcule la moyenne dans la zone de référence (souvent 300-1000Hz)
        raw_avg = np.mean(raw_w_spl[self.mask_ref]) if np.any(self.mask_ref) else self.target_spl
        
        # On fixe la cible 1.5 dB sous la sensibilité naturelle (pertes normales d'un filtre passif)
        self.target_spl = raw_avg - 1.5
        print(f"[+] Cible SPL verrouillée à {self.target_spl:.1f} dB (Woofer brut: {raw_avg:.1f} dB)")
        
        # =========================================================
        # 2. NOUVEAU : Détection automatique de la plage utile (mask_flat)
        # =========================================================
        # On superpose les capacités maximales de TOUS les haut-parleurs bruts
        max_raw_spl = np.zeros_like(self.freqs)
        for way in self.ways:
            raw_mag = np.abs(way.driver.H_acoustic)
            raw_spl = 20 * np.log10(raw_mag + 1e-12)
            max_raw_spl = np.maximum(max_raw_spl, raw_spl)
            
        # On cherche la plage où le système "nu" est capable de jouer à au moins -10dB de la cible
        playable_mask = max_raw_spl >= (self.target_spl - 10.0)
        valid_indices = np.where(playable_mask)[0]
        
        if len(valid_indices) > 0:
            idx_min = valid_indices[0]
            idx_max = valid_indices[-1]
            
            # On resserre de quelques crans pour ne pas inclure la "chute" (roll-off) brutale dans le calcul MSE
            f_min = self.freqs[min(idx_min + 5, len(self.freqs)-1)]
            f_max = self.freqs[max(idx_max - 5, 0)]
            # Sécurité
            f_min = max(f_min, 100.0)
        else:
            f_min, f_max = 100, 18000 # Sécurité si les fichiers FRD sont vides
            
        self.mask_flat = (self.freqs >= f_min) & (self.freqs <= f_max)
        print(f"[+] Plage d'optimisation auto-détectée : {int(f_min)} Hz - {int(f_max)} Hz")

    def _prepare_driver(self, way):
        d = way.driver
        d.model_name = os.path.basename(way.frd_path).split('.')[0].split('@')[0] # Extraction du nom du fichier sans extension
        # Interpolation Magnitude et Phase (avec unwrap pour éviter les sauts de 2pi)
        mag_db = 20 * np.log10(np.abs(d.H_acoustic) + 1e-10)
        ph_unwrapped = np.unwrap(np.angle(d.H_acoustic))
        
        mag_interp = np.interp(self.freqs, d.frd_freqs, mag_db)
        ph_interp = np.interp(self.freqs, d.frd_freqs, ph_unwrapped)
        d.H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
        
        # Application du Z-Offset (Délai acoustique)
        delay_s = way.z_offset_m / 343.0
        phase_delay = np.exp(-1j * 2 * np.pi * self.freqs * delay_s)
        d.H_acoustic *= phase_delay

        # Impédance
        z_mag = np.abs(d.Z_complex)
        z_ph = np.unwrap(np.angle(d.Z_complex))
        d.Z_complex = np.interp(self.freqs, d.zma_freqs, z_mag) * np.exp(1j * np.interp(self.freqs, d.zma_freqs, z_ph))

    def _get_lr4_transfer(self, f_target, type='LP'):
        s = 1j * (self.freqs / f_target)
        poly = (s**2 + np.sqrt(2)*s + 1)**2
        if type == 'LP': return 1 / poly
        if type == 'HP': return (s**4) / poly
        return np.ones_like(self.freqs)

    def fitness(self, individual):
        root = individual['tree']

        
        if not isinstance(root, ParallelNode): 
            return 1e9

    
        res = self.evaluator.evaluate(root)
        
        dynamic_spl = self.target_spl
        
        total_mse = 0.0

        # --- 2. ÉVALUATION DYNAMIQUE DES POLARITÉS ET DE LA PHASE ---
        best_score_sum = float('inf')
        best_polarities = [1.0] * len(self.ways)
        
        # On génère toutes les combinaisons possibles (+1 ou -1)
        pol_combinations = list(itertools.product([1.0, -1.0], repeat=len(self.ways)-1))

        for pols in pol_combinations:
            current_pols = [1.0] + list(pols) # Le woofer est toujours à +1.0
            p_sum_test = np.zeros_like(self.freqs, dtype=complex)
            p_ways = [] # Pour stocker la pression de chaque voie individuellement
            
            for i, way in enumerate(self.ways):
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_adj = p_real * current_pols[i] # Pression avec la polarité testée
                p_ways.append(p_adj)
                p_sum_test += p_adj 
            
            spl_sum_test = 20 * np.log10(np.abs(p_sum_test) + 1e-12)
            diff = spl_sum_test - dynamic_spl
            
            # =========================================================
            # --- NOUVEAU : Détection dynamique du point de croisement ---
            # =========================================================
            dynamic_weight = np.zeros_like(self.freqs)
            dynamic_weight[self.mask_flat] = 1.0 # Poids de base
            
            for j in range(len(self.ways) - 1):
                mag1 = 20 * np.log10(np.abs(p_ways[j]) + 1e-12)
                mag2 = 20 * np.log10(np.abs(p_ways[j+1]) + 1e-12)
                
                # On restreint la recherche pour éviter de trouver des faux croisements 
                # dans les extrêmes graves ou aigus (ex: entre 500Hz et 8000Hz)
                search_mask = (self.freqs > 500) & (self.freqs < 4000)
                
                if np.any(search_mask):
                    # Le point de croisement est là où la différence de volume entre les 2 HP est la plus petite
                    idx_cross = np.argmin(np.abs(mag1[search_mask] - mag2[search_mask]))
                    f_cross = self.freqs[search_mask][idx_cross]
                    
                    # On booste le poids de l'erreur autour de cette fréquence (ex: +/- 1 octave)
                    dynamic_weight[(self.freqs > f_cross / 2.0) & (self.freqs < f_cross * 2.0)] = WEIGHTS['crossover']
            # =========================================================
            
            # 1. NORMALISATION : On calcule les erreurs brutes d'abord
            raw_mse = np.mean(np.where(diff > 0, (diff**2) * 5.0, diff**2) * dynamic_weight)
            
            raw_ripple = 0.0
            if np.any(self.ripple_mask):
                raw_ripple = np.std(spl_sum_test[self.ripple_mask]) ** 2
            
            # raw_phase_penalty = 0.0
            # for j in range(len(self.ways) - 1):
            #     p1, p2 = p_ways[j], p_ways[j+1]
            #     phase_diff_rad = np.abs(np.angle(p1 / (p2 + 1e-12)))
            #     overlap_weight = np.abs(p1) * np.abs(p2) / (np.max(np.abs(p1) * np.abs(p2)) + 1e-12)
            #     raw_phase_penalty += np.mean(np.maximum(0, phase_diff_rad - 0.52) * overlap_weight)
            
            mean_spl = np.mean(spl_sum_test[self.mask_flat])
            
            # 2. APPLICATION DES POIDS RELATIFS
            score_sum = (raw_mse * WEIGHTS['mse_sum'])
                        #(raw_ripple * WEIGHTS['ripple'])
                        #(-mean_spl * WEIGHTS['mean_spl']) 
                        #(raw_phase_penalty * 100.0 * WEIGHTS['phase'])
                        

            
            if score_sum < best_score_sum:
                best_score_sum = score_sum
                best_polarities = current_pols
        
        total_mse += best_score_sum
        individual['best_polarities'] = best_polarities

        # --- 3. CONTRAINTES LÉTALES (On ne négocie pas la physique) ---
        penalty = 0.0
        Z_in = self.evaluator.get_impedance(root)
        min_Z = np.min(np.abs(Z_in))
        if min_Z < 3.2: 
            penalty += 10000.0 + (3.2 - min_Z) * 5000.0 # Mur de brique
        
        last_way_v = res.get(self.ways[-1].label, {}).get("V_complex", np.zeros_like(self.freqs))
        
        v_low = np.abs(last_way_v)[self.freqs < 1000.0]
        if np.any(v_low > 0.1):
            penalty += 10000.0 + np.sum(v_low) * 1000.0 # Mur de brique
        
        # --- 4. COMPOSANTS ---
        comps = [n for n in root.get_all_nodes() if isinstance(n, ComponentNode)]
        n_comps = len(comps)
        if n_comps <= WEIGHTS['n_comps']:
            comp_penalty = 0.0
        else:
            comp_penalty = (n_comps - WEIGHTS['n_comps']) * (total_mse * WEIGHTS['components'])

        # NOUVEAU : Taxe spécifique sur les résistances
        n_resistors = len([c for c in comps if isinstance(c, Resistor)])
        # Chaque résistance ajoute un malus de X% sur le score
        resistor_penalty = n_resistors * (total_mse * WEIGHTS['resistors'])

        return total_mse + penalty + comp_penalty + resistor_penalty


    def _elite_worker(self, args):
        ind, max_opt, snap = args
        
        # Si on est en phase continue et que l'individu est déjà optimisé, on passe
        if not snap and ind.get('is_optimized', False):
            return (self.fitness(ind), ind)
            
        if not snap:
            self.optimize_values(ind, max_iter=max_opt)
            ind['is_optimized'] = True # On le marque comme "propre"
        else:
            for comp in ind['tree'].get_all_nodes():
                if isinstance(comp, ComponentNode):
                    comp.value = snap_to_e24(comp.value)
            self.optimize_e24_values(ind)
            ind['is_optimized'] = False # En phase E24, on réévalue toujours
            
        return (self.fitness(ind), ind)

    def _lamarckian_worker(self, child_ind):
        if random.random() < 0.25:
            self.optimize_values(child_ind, max_iter=3)
            child_ind['is_optimized'] = True # Marqué comme pré-entraîné
        return child_ind

    def optimize_e24_values(self, individual):
        """
        Recherche locale discrète : teste les valeurs E24 voisines pour chaque composant
        afin d'affiner le score sans casser la topologie.
        """
        comps = [n for n in individual['tree'].get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        best_score = self.fitness(individual)
        improved = True
        
        # Tant qu'on trouve des améliorations en bougeant d'un cran, on continue
        while improved:
            improved = False
            for comp in comps:
                original_val = snap_to_e24(comp.value)
                
                # Calcul de l'index actuel dans la série E24
                power = np.floor(np.log10(original_val))
                norm = original_val / (10 ** power)
                idx = np.abs(E24_SERIES - norm).argmin()
                
                best_comp_val = original_val
                
                # On teste le cran en dessous (-1) et le cran au dessus (+1)
                for step in [-1, 1]:
                    new_idx = idx + step
                    new_power = power
                    
                    # Gestion des passages de dizaines (ex: 9.1 -> 10.0)
                    if new_idx < 0:
                        new_idx = len(E24_SERIES) - 1
                        new_power -= 1
                    elif new_idx >= len(E24_SERIES):
                        new_idx = 0
                        new_power += 1
                        
                    test_val = E24_SERIES[new_idx] * (10 ** new_power)
                    comp.value = test_val
                    
                    new_score = self.fitness(individual)
                    if new_score < best_score:
                        best_score = new_score
                        best_comp_val = test_val
                        improved = True
                
                # On applique la meilleure valeur (qui peut être l'originale si pas d'amélioration)
                comp.value = best_comp_val
                
        return individual

    def optimize_values(self, individual, max_iter=5):
        root = individual['tree']
        comps = [n for n in root.get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        init = [np.log10(np.clip(c.value, 1e-12, 1e2)) for c in comps]
        bounds = [(np.log10(BOUNDS_R[0]), np.log10(BOUNDS_R[1])) if isinstance(c, Resistor) else 
                  (np.log10(BOUNDS_C[0]), np.log10(BOUNDS_C[1])) if isinstance(c, Capacitor) else 
                  (np.log10(BOUNDS_L[0]), np.log10(BOUNDS_L[1])) for c in comps]
                  
        def obj(x_log):
            for i, v in enumerate(x_log): comps[i].value = 10**v
            return self.fitness(individual)
            
        res = minimize(
            obj, init, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,   # Tolérance sur la fonction relâchée
                'eps': 1e-3     # Pas de calcul du gradient plus large = beaucoup plus rapide
            }
        )
        for i, v in enumerate(res.x): comps[i].value = 10**v
        return individual

    def run(self, generations=50, pop_size=60):
        population = []
        
        # Tentative de chargement du champion
        if os.path.exists("00_best_crossover.json"):
            try:
                with open("00_best_crossover.json", "r") as f: data = json.load(f)
                tree = Node.from_dict(data["tree"])
                # Réassignation des drivers
                for n in tree.get_all_nodes():
                    if isinstance(n, DriverNode):
                        way = next(w for w in self.ways if w.label == n.label)
                        n.H_acoustic, n.Z_complex = way.driver.H_acoustic, way.driver.Z_complex
                
                # Chargement de la nouvelle structure
                best_pols = data.get("best_polarities", [1.0] * len(self.ways))
                population.append({'tree': tree, 'best_polarities': best_pols})
                print("[+] Champion chargé.")
            except Exception as e: print(f"Erreur chargement: {e}")
        
        else:
            try:
                # Graine 3ème ordre (18dB/octave) : beaucoup plus plat naturellement
                w_branch = SeriesNode(Inductor(1.5e-3), ParallelNode(Capacitor(10e-6), SeriesNode(Inductor(0.5e-3), self.ways[0].driver.copy())))
                t_branch = SeriesNode(Capacitor(4.7e-6), ParallelNode(Inductor(0.3e-3), SeriesNode(Capacitor(10e-6), self.ways[1].driver.copy())))
                seed_tree = ParallelNode(w_branch, t_branch)
                population.append({'tree': seed_tree})
                print("[+] Graine (Template 3ème ordre) injectée.")
            except Exception as e:
                pass

        # Remplissage
        while len(population) < pop_size:
            branches = []
            for way in self.ways:
                branches.append(self.mutator.generate_random_tree(way.driver.copy(), max_depth=2))
            
            root = branches[0]
            for b in branches[1:]:
                root = ParallelNode(root, b)
            
            population.append({'tree': root})

        best_score = float('inf')
        best_ind = population[0]

        best_score = float('inf')
        best_ind = population[0]

        # --- MODIFICATION : Ouverture du Pool de Processus ---
        # Le Pool gère automatiquement le nombre de cœurs de votre processeur (ex: 8, 12, ou 16)
        with ProcessPoolExecutor() as executor:
            
            for gen in range(generations):
                
                # ==========================================
                # 1. PARALLÉLISATION DE LA FITNESS (Avec Chunksize)
                # ==========================================
                # Le chunksize=10 envoie les individus par paquets de 10, divisant le temps de transfert par 10
                fitness_results = list(executor.map(self.fitness, population, chunksize=10))
                scores = [(fit, ind) for fit, ind in zip(fitness_results, population)]
                
                scores.sort(key=lambda x: x[0])
                
                if gen < int(generations * 0.4):      
                    max_opt_iter = 2
                    snap_to_standard = False
                elif gen < int(generations * 0.9):    
                    max_opt_iter = 12
                    snap_to_standard = False
                else:                                 
                    max_opt_iter = 20
                    snap_to_standard = True
                    
                if gen == int(generations * 0.9):
                    print("Passage en PHASE 3")
                    best_score = float('inf') 

                # ==========================================
                # NOUVEAU : PARALLÉLISATION DE L'ÉLITE
                # ==========================================
                elite_count = max(2, pop_size // 10)
                
                # On prépare les paquets de travail pour chaque élite
                elite_args = [(scores[i][1], max_opt_iter, snap_to_standard) for i in range(elite_count)]
                
                # On envoie tout aux cœurs !
                optimized_elites = list(executor.map(self._elite_worker, elite_args))
                
                # On met à jour les scores avec les résultats revenus des workers
                for i in range(elite_count):
                    scores[i] = optimized_elites[i]
                
                scores.sort(key=lambda x: x[0])
                
                if scores[0][0] < best_score:
                    best_score = scores[0][0]
                    best_ind = scores[0][1]
                    
                    save_tree = best_ind['tree'].copy()
                    for comp in save_tree.get_all_nodes():
                        if isinstance(comp, ComponentNode):
                            comp.value = snap_to_e24(comp.value)
                    
                    n_comps = len([n for n in best_ind['tree'].get_all_nodes() if isinstance(n, ComponentNode)])
                    
                    if gen % 5 == 0:
                        print(f"Gen {gen}: Record {best_score:.2f} | Composants: {n_comps}")
                    
                    with open("00_best_crossover.json", "w") as f:
                        json.dump({
                            "tree": save_tree.to_dict(), 
                            "best_polarities": best_ind.get('best_polarities', [1.0] * len(self.ways))
                        }, f, indent=4)

                # ==========================================
                # 2. PARALLÉLISATION DE L'ENTRAÎNEMENT DES ENFANTS
                # ==========================================
                new_pop = [best_ind]
                for i in range(1, elite_count):
                    new_pop.append(scores[i][1])
                    
                # A. Génération rapide (séquentielle) de tous les enfants "bruts"
                raw_children = []
                while len(new_pop) + len(raw_children) < pop_size:
                    parent = random.choice(scores[:pop_size//3])[1]
                    child_tree = self.mutator.mutate(parent['tree'].copy())

                    raw_children.append({'tree': child_tree, 'is_optimized': False})
                
                # B. Entraînement Lamarckien (lourd) distribué en parallèle sur tous les cœurs
                if gen < int(generations * 0.9):
                    trained_children = list(executor.map(self._lamarckian_worker, raw_children))
                    new_pop.extend(trained_children)
                else:
                    new_pop.extend(raw_children)
                
                population = new_pop

        # Polissage final
        # On ne fait de l'optimisation continue QUE si on n'est pas en mode E24
        if not snap_to_standard:
            self.optimize_values(best_ind, max_iter=150)
        else:
            # Sinon on donne un dernier coup de Hill-Climber E24
            self.optimize_e24_values(best_ind)
            
        # CLIP SÉCURITÉ OBLIGATOIRE AVANT AFFICHAGE
        for comp in best_ind['tree'].get_all_nodes():
            if isinstance(comp, ComponentNode):
                comp.value = snap_to_e24(comp.value)
            elif isinstance(comp, DriverNode):
                # On retrouve la configuration de la voie correspondante
                way = next(w for w in self.ways if w.label == comp.label)
                # On réassigne le nom extrait
                comp.model_name = way.driver.model_name
                
        self.plot_result(best_ind)
    
        renderer = SchematicRenderer(best_ind['tree'])
        renderer.save("00_crossover_schematic.png")
        
        return best_ind

    def plot_result(self, individual, filename=["00_crossover_response.png","00_filter.png"]):
        root = individual['tree']
        # On n'a plus besoin de fx_list pour les cibles
        res = self.evaluator.evaluate(root)
        
        # Le graphique doit utiliser la même cible stricte que l'optimiseur
        dynamic_spl = self.target_spl
        
        plt.figure(figsize=(12, 8))
        p_sum = np.zeros_like(self.freqs, dtype=complex)
        best_pols = individual.get('best_polarities', [1.0] * len(self.ways))
        
        for i, way in enumerate(self.ways):
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_sum += p_real * best_pols[i] 
            
            spl_real = 20 * np.log10(np.abs(p_real) + 1e-10)
            label_suffix = " (inv)" if best_pols[i] < 0 else ""
            plt.semilogx(self.freqs, spl_real, label=f"Réel {way.label}{label_suffix}", linewidth=2)

        plt.semilogx(self.freqs, 20 * np.log10(np.abs(p_sum) + 1e-10), label="Somme", color='red', linewidth=3)

            
        plt.axhline(dynamic_spl, color='green', linestyle='--', alpha=0.5)
        plt.ylim(dynamic_spl - 40, dynamic_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.title(f"Réponse {len(self.ways)}-voies (Score: {self.fitness(individual):.2f})")
        plt.savefig(filename[0])
        plt.close()
        
        # Fonctions de transfert
        plt.figure(figsize=(12, 8))
        for i, way in enumerate(self.ways):
            # On récupère la tension électrique pure aux bornes du haut-parleur (sans la réponse acoustique)
            v_complex = res.get(way.label, {}).get("V_complex", np.zeros_like(self.freqs))
            
            # L'entrée (Ampli) est calculée à 1V (0 dB). On trace donc l'atténuation du filtre en dB.
            filter_db = 20 * np.log10(np.abs(v_complex) + 1e-12)
            
            plt.semilogx(self.freqs, filter_db, label=f"Filtre {way.label}", linewidth=2)
            
        plt.axhline(0, color='black', linestyle='-', alpha=0.5, label="0 dB (Signal Ampli brut)")
        
        # Affichage du point de croisement électrique
        plt.ylim(-40, 5) # L'échelle s'arrête à +5dB pour voir les petites surtensions éventuelles
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.title("Fonction de transfert électrique du filtre (V_out / V_in)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Atténuation (dB)")
        plt.savefig(filename[1])
        plt.close()
        
        

if __name__ == "__main__":
    # CONFIGURATION 2-VOIES PERSONNALISÉE
    start_time = time.time()
    config = [
        WayConfig("Woofer", "Driver_Data/RS225-8@0.frd", "Driver_Data/RS225-8.zma", target_type='LP', z_offset_m=0.03),
        WayConfig("Tweeter", "Driver_Data/SEAS_27TDFC_tweeter_SPL.frd", "Driver_Data/SEAS_27TDFC_tweeter_ZR.zma", target_type='HP', z_offset_m=0.0)
    ]

    opt = CrossoverOptimizer(config)
    best = opt.run(generations=150, pop_size=120)
    best['tree'].display()
    end_time = time.time()
    print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")

