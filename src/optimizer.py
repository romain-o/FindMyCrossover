import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from src.nodes import DriverNode, SeriesNode, ParallelNode, ShuntNode, Capacitor, Inductor, Resistor, Node, ComponentNode
from src.evaluator import CircuitEvaluator
from src.mutator import TreeMutator
from scipy.optimize import minimize
from src.schematic import SchematicRenderer
import time

# ============================================================
# OPTIMISATION #1 : Fonctions module-level pour multiprocessing
# → self est pickled UNE SEULE FOIS par worker via initializer,
#   pas une fois par tâche comme avec ProcessPoolExecutor.
# ============================================================
from multiprocessing import Pool, cpu_count

_pool_optimizer = None  # Global par worker process

def _pool_init(opt):
    global _pool_optimizer
    _pool_optimizer = opt

def _pool_fitness(ind):
    return _pool_optimizer.fitness(ind)

def _pool_elite(args):
    return _pool_optimizer._elite_worker(args)

def _pool_lamarckian(child):
    return _pool_optimizer._lamarckian_worker(child)


BOUNDS_R = (0.1, 50.0)
BOUNDS_C = (0.1e-6, 150e-6)
BOUNDS_L = (0.05e-3, 15e-3)

E24_SERIES = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 
                       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])

WEIGHTS = {
                'mse_sum': 1.0,
                'phase': 0.0,
                'n_comps': 8,
                'components': 0.05,
                'mean_spl': 0.0,
                'crossover': 5,
                'resistors': 0.01
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
    def __init__(self, label, frd_path, zma_path, order=4, z_offset=0.0, y_offset=0.0, x_offset=0.0):
        self.label = label
        self.frd_path = frd_path
        self.zma_path = zma_path
        self.order = order
        self.z_offset = z_offset
        self.y_offset = y_offset 
        self.x_offset = x_offset
        self.driver = DriverNode(label, frd_path, zma_path)

class CrossoverOptimizer:
    def __init__(self, ways_configs):
        self.freqs = np.geomspace(20, 20000, 400)
        self.ways = ways_configs
        self.evaluator = CircuitEvaluator(self.freqs)
        self.mutator = TreeMutator(
            prob_value_mut=0.2, 
            prob_type_mut=0.1, 
            prob_topology_mut=0.3, 
            prob_add_node=0.35,
            prob_remove_node=0.05
        )
                            
        self.ripple_mask = (self.freqs >= 400) & (self.freqs <= 12000)
        self.mask_ref = (self.freqs > 300) & (self.freqs < 1000)
        
        for way in self.ways:
            self._prepare_driver(way)
        
        raw_w_mag = np.abs(self.ways[0].driver.H_acoustic)
        raw_w_spl = 20 * np.log10(raw_w_mag + 1e-12)
        raw_avg = np.mean(raw_w_spl[self.mask_ref]) if np.any(self.mask_ref) else self.target_spl
        self.target_spl = raw_avg - 1.0
        print(f"[+] Cible SPL verrouillée à {self.target_spl:.1f} dB (Woofer brut: {raw_avg:.1f} dB)")
        
        max_raw_spl = np.zeros_like(self.freqs)
        for way in self.ways:
            raw_mag = np.abs(way.driver.H_acoustic)
            raw_spl = 20 * np.log10(raw_mag + 1e-12)
            max_raw_spl = np.maximum(max_raw_spl, raw_spl)
            
        playable_mask = max_raw_spl >= (self.target_spl - 10.0)
        valid_indices = np.where(playable_mask)[0]
        
        if len(valid_indices) > 0:
            idx_min = valid_indices[0]
            idx_max = valid_indices[-1]
            f_min = self.freqs[min(idx_min + 5, len(self.freqs)-1)]
            f_max = self.freqs[max(idx_max - 5, 0)]
            f_min = max(f_min, 100.0)
        else:
            f_min, f_max = 100, 18000
            
        self.mask_flat = (self.freqs >= f_min) & (self.freqs <= f_max)
        print(f"[+] Plage d'optimisation auto-détectée : {int(f_min)} Hz - {int(f_max)} Hz")

        # ============================================================
        # OPTIMISATION #2 : Pré-calcul du poids statique de base
        # → Evite de reconstruire dynamic_weight depuis zéro à chaque
        #   appel de fitness() (appelé des milliers de fois).
        # ============================================================
        self._base_weight = np.zeros(len(self.freqs))
        self._base_weight[self.mask_flat] = 1.0

    def _prepare_driver(self, way):
        d = way.driver
        d.model_name = os.path.basename(way.frd_path).split('.')[0].split('@')[0]
        mag_db = 20 * np.log10(np.abs(d.H_acoustic) + 1e-10)
        ph_unwrapped = np.unwrap(np.angle(d.H_acoustic))
        
        mag_interp = np.interp(self.freqs, d.frd_freqs, mag_db)
        ph_interp = np.interp(self.freqs, d.frd_freqs, ph_unwrapped)
        d.H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
        
        delay_s = np.linalg.norm([way.x_offset, way.y_offset, way.z_offset + 2]) / 343.0
        phase_delay = np.exp(-1j * 2 * np.pi * self.freqs * delay_s)
        d.H_acoustic *= phase_delay

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
        # ============================================================
        # OPTIMISATION #3 : Cache du score de fitness
        # → Un individu non modifié n'est JAMAIS réévalué.
        #   Le cache est invalidé par optimize_values() et
        #   optimize_e24_values() dès qu'une valeur change.
        #   Les nouveaux enfants (mutés/croisés) n'ont pas la clé
        #   '_cached_score', donc sont toujours évalués.
        # ============================================================

        root = individual['tree']

        if not isinstance(root, ParallelNode): 
            return 1e9

        res = self.evaluator.evaluate(root)
        dynamic_spl = self.target_spl
        total_mse = 0.0

        best_score_sum = float('inf')
        
        p_sum_test = np.zeros_like(self.freqs, dtype=complex)
        p_ways = []
        
        for i, way in enumerate(self.ways):
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_ways.append(p_real)
            p_sum_test += p_real
            
        spl_sum_test = 20 * np.log10(np.abs(p_sum_test) + 1e-12)
        diff = spl_sum_test - dynamic_spl
        
        # ============================================================
        # OPTIMISATION #4 : Partir du poids pré-calculé (copie rapide)
        # au lieu de recrér un array de zéros + le remplir
        # ============================================================
        dynamic_weight = self._base_weight.copy()
        
        for j in range(len(self.ways) - 1):
            mag1 = 20 * np.log10(np.abs(p_ways[j]) + 1e-12)
            mag2 = 20 * np.log10(np.abs(p_ways[j+1]) + 1e-12)
            
            if len(self.ways) == 2:
                search_mask = (self.freqs > 800) & (self.freqs < 5000)
            else:
                if j == 0:
                    search_mask = (self.freqs > 150) & (self.freqs < 1000)
                elif j == 1:
                    search_mask = (self.freqs > 1000) & (self.freqs < 8000)
                else:
                    search_mask = (self.freqs > 2000) & (self.freqs < 12000)
            
            if np.any(search_mask):
                idx_cross = np.argmin(np.abs(mag1[search_mask] - mag2[search_mask]))
                f_cross = self.freqs[search_mask][idx_cross]
                dynamic_weight[(self.freqs > f_cross / 2.0) & (self.freqs < f_cross * 2.0)] = WEIGHTS['crossover']
        
        raw_mse = np.mean(np.where(diff > 0, (diff**2) * 5.0, diff**2) * dynamic_weight)
        
        raw_ripple = 0.0
        if np.any(self.ripple_mask):
            raw_ripple = np.std(spl_sum_test[self.ripple_mask]) ** 2
        
        mean_spl = np.mean(spl_sum_test[self.mask_flat])
        
        score_sum = (raw_mse * WEIGHTS['mse_sum'])

        if score_sum < best_score_sum:
            best_score_sum = score_sum
        
        total_mse += best_score_sum

        penalty = 0.0
        
        Z_in = self.evaluator.get_impedance(root)
        min_Z = np.min(np.abs(Z_in))
        if min_Z < 3.2: 
            penalty += 10000.0 + (3.2 - min_Z) * 5000.0
        
        last_way_v = res.get(self.ways[-1].label, {}).get("V_complex", np.zeros_like(self.freqs))
        
        v_low = np.abs(last_way_v)[self.freqs < 1000.0]
        if np.any(v_low > 0.1):
            penalty += 10000.0 + np.sum(v_low) * 1000.0

        # ============================================================
        # OPTIMISATION #5 : get_all_nodes() appelé une seule fois
        # → On réutilise `all_nodes` pour comps ET n_resistors
        #   au lieu de traverser l'arbre deux fois.
        # ============================================================
        all_nodes = root.get_all_nodes()
        comps = [n for n in all_nodes if isinstance(n, ComponentNode)]
        n_comps = len(comps)

        if n_comps <= WEIGHTS['n_comps']:
            comp_penalty = 0.0
        else:
            comp_penalty = (n_comps - WEIGHTS['n_comps']) * (total_mse * WEIGHTS['components'])

        n_resistors = sum(1 for c in comps if isinstance(c, Resistor))
        resistor_penalty = n_resistors * (total_mse * WEIGHTS['resistors'])

        final_score = total_mse + penalty + comp_penalty + resistor_penalty

        return final_score


    def _elite_worker(self, args):
        ind, max_opt, snap = args
        
        if not snap and ind.get('is_optimized', False):
            return (self.fitness(ind), ind)
            
        if not snap:
            self.optimize_values(ind, max_iter=max_opt)
            ind['is_optimized'] = True
        else:
            for comp in ind['tree'].get_all_nodes():
                if isinstance(comp, ComponentNode):
                    comp.value = snap_to_e24(comp.value)
            self.optimize_e24_values(ind)
            ind['is_optimized'] = False
            
        return (self.fitness(ind), ind)

    def _lamarckian_worker(self, child_ind):
        if random.random() < 0.40:
            self.optimize_values(child_ind, max_iter=8)
            child_ind['is_optimized'] = True
        return child_ind

    def optimize_e24_values(self, individual):
        """
        Recherche locale discrète : teste les valeurs E24 voisines pour chaque composant.
        """
        # ============================================================
        # OPTIMISATION #3 (suite) : Invalidation du cache
        # → Toute modification de valeur rend le score précédent caduc.
        # ============================================================
        individual.pop('_cached_score', None)

        comps = [n for n in individual['tree'].get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        best_score = self.fitness(individual)
        improved = True
        
        while improved:
            improved = False
            for comp in comps:
                original_val = snap_to_e24(comp.value)
                
                power = np.floor(np.log10(original_val))
                norm = original_val / (10 ** power)
                idx = np.abs(E24_SERIES - norm).argmin()
                
                best_comp_val = original_val
                
                for step in [-1, 1]:
                    new_idx = idx + step
                    new_power = power
                    
                    if new_idx < 0:
                        new_idx = len(E24_SERIES) - 1
                        new_power -= 1
                    elif new_idx >= len(E24_SERIES):
                        new_idx = 0
                        new_power += 1
                        
                    test_val = E24_SERIES[new_idx] * (10 ** new_power)
                    comp.value = test_val
                    
                    # Forcer le recalcul à chaque test de valeur
                    individual.pop('_cached_score', None)
                    new_score = self.fitness(individual)
                    if new_score < best_score:
                        best_score = new_score
                        best_comp_val = test_val
                        improved = True
                
                comp.value = best_comp_val
                individual.pop('_cached_score', None)
                
        return individual

    def optimize_values(self, individual, max_iter=5):
        # ============================================================
        # OPTIMISATION #3 (suite) : Invalidation du cache avant optim
        # ============================================================
        individual.pop('_cached_score', None)

        root = individual['tree']
        comps = [n for n in root.get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        init = [np.log10(np.clip(c.value, 1e-12, 1e2)) for c in comps]
        bounds = [(np.log10(BOUNDS_R[0]), np.log10(BOUNDS_R[1])) if isinstance(c, Resistor) else 
                  (np.log10(BOUNDS_C[0]), np.log10(BOUNDS_C[1])) if isinstance(c, Capacitor) else 
                  (np.log10(BOUNDS_L[0]), np.log10(BOUNDS_L[1])) for c in comps]
        
                  
        def obj(x_log):
            for i, v in enumerate(x_log): comps[i].value = 10**v
            # Invalider le cache à chaque itération de l'optimiseur (valeurs qui changent)
            individual.pop('_cached_score', None)
            return self.fitness(individual)
            
        res = minimize(
            obj, init, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,
                'eps': 1e-3
            }
        )
        for i, v in enumerate(res.x): comps[i].value = 10**v
        # Le cache est déjà invalidé par le dernier appel obj() dans minimize
        return individual

    def run(self, generations=50, pop_size=60, checkpoint_path=None):
        """
        Lance l'optimisation. 
        Accepte un checkpoint_path optionnel pour ne pas écraser les autres projets.
        """
        population = []
        
        # Chargement intelligent du checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f: data = json.load(f)
                tree = Node.from_dict(data["tree"])
                for n in tree.get_all_nodes():
                    if isinstance(n, DriverNode):
                        way = next(w for w in self.ways if w.label == n.label)
                        n.H_acoustic, n.Z_complex = way.driver.H_acoustic, way.driver.Z_complex
                
                population.append({'tree': tree})
                print(f"[+] Champion chargé depuis {checkpoint_path}")
            except Exception as e: 
                print(f"Erreur chargement du checkpoint: {e}")
        else:
            try:
               if len(self.ways) == 2:
                    w_branch = SeriesNode(Inductor(1.5e-3), ParallelNode(Capacitor(10e-6), SeriesNode(Inductor(0.5e-3), self.ways[0].driver.copy())))
                    t_branch = SeriesNode(Capacitor(4.7e-6), ParallelNode(Inductor(0.3e-3), SeriesNode(Capacitor(10e-6), self.ways[1].driver.copy())))
                    seed_tree = ParallelNode(w_branch, t_branch)
                    population.append({'tree': seed_tree})
                    print("[+] Graine (Template 3ème ordre) injectée.")
            except Exception as e:
                pass

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

        n_workers = cpu_count()
        chunksize = max(1, pop_size // (n_workers * 4))

        with Pool(processes=n_workers, initializer=_pool_init, initargs=(self,)) as pool:
            for gen in range(generations):
                fitness_results = pool.map(_pool_fitness, population, chunksize=chunksize)
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
                    print("Passage en PHASE 3 (Standardisation E24)")
                    best_score = float('inf') 

                elite_count = max(2, pop_size // 10)
                elite_args = [(scores[i][1], max_opt_iter, snap_to_standard) for i in range(elite_count)]
                optimized_elites = pool.map(_pool_elite, elite_args, chunksize=1)
                
                for i in range(elite_count):
                    scores[i] = optimized_elites[i]
                scores.sort(key=lambda x: x[0])
                
                if scores[0][0] < best_score:
                    best_score = scores[0][0]
                    best_ind = scores[0][1]
                    
                    # Sauvegarde dynamique du checkpoint si le chemin est fourni
                    if checkpoint_path:
                        save_tree = best_ind['tree'].copy()
                        for comp in save_tree.get_all_nodes():
                            if isinstance(comp, ComponentNode):
                                comp.value = snap_to_e24(comp.value)
                        with open(checkpoint_path, "w") as f:
                            json.dump({
                                "tree": save_tree.to_dict(), 
                            }, f, indent=4)

                new_pop = [best_ind]
                for i in range(1, elite_count):
                    new_pop.append(scores[i][1])
                    
                raw_children = []
                while len(new_pop) + len(raw_children) < pop_size:
                    def tournament():
                        competitors = random.sample(scores, 3)
                        return min(competitors, key=lambda x: x[0])[1]

                    parent1 = tournament()
                    if random.random() < 0.30:
                        parent2 = tournament()
                        child_tree = self.mutator.crossover(parent1['tree'], parent2['tree'])
                    else:
                        child_tree = self.mutator.mutate(parent1['tree'].copy())

                    raw_children.append({'tree': child_tree, 'is_optimized': False})
                
                if gen < int(generations * 0.9):
                    trained_children = pool.map(_pool_lamarckian, raw_children, chunksize=chunksize)
                    new_pop.extend(trained_children)
                else:
                    new_pop.extend(raw_children)
                
                population = new_pop

        if not snap_to_standard:
            self.optimize_values(best_ind, max_iter=150)
        else:
            self.optimize_e24_values(best_ind)
            
        for comp in best_ind['tree'].get_all_nodes():
            if isinstance(comp, ComponentNode):
                comp.value = snap_to_e24(comp.value)
            elif isinstance(comp, DriverNode):
                way = next(w for w in self.ways if w.label == comp.label)
                comp.model_name = way.driver.model_name

        # On retourne juste le meilleur individu sans tracer les graphiques ici !
        return best_ind

    def plot_result(self, individual, filename_response="response.png", filename_filter="filter.png", last_score=None):
        """Génère les graphiques avec des chemins personnalisés."""
        root = individual['tree']
        res = self.evaluator.evaluate(root)
        dynamic_spl = self.target_spl
        
        # 1. Graphique de Réponse SPL
        plt.figure(figsize=(12, 8))
        p_sum = np.zeros_like(self.freqs, dtype=complex)
        
        for i, way in enumerate(self.ways):
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_sum += p_real
            spl_real = 20 * np.log10(np.abs(p_real) + 1e-10)
            plt.semilogx(self.freqs, spl_real, label=f"Réel {way.label}", linewidth=2)

        plt.semilogx(self.freqs, 20 * np.log10(np.abs(p_sum) + 1e-10), label="Somme", color='red', linewidth=3)
        plt.axhline(dynamic_spl, color='green', linestyle='--', alpha=0.5)
        plt.ylim(dynamic_spl - 40, dynamic_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        
        score_display = last_score if last_score is not None else self.fitness(individual)
        plt.title(f"Réponse {len(self.ways)}-voies (Score: {score_display:.2f})")
        plt.savefig(filename_response)
        plt.close()
        
        # 2. Graphique de Fonction de Transfert Électrique
        plt.figure(figsize=(12, 8))
        for i, way in enumerate(self.ways):
            v_complex = res.get(way.label, {}).get("V_complex", np.zeros_like(self.freqs))
            filter_db = 20 * np.log10(np.abs(v_complex) + 1e-12)
            plt.semilogx(self.freqs, filter_db, label=f"Filtre {way.label}", linewidth=2)
            
        plt.axhline(0, color='black', linestyle='-', alpha=0.5, label="0 dB (Signal Ampli brut)")
        plt.ylim(-40, 5)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.title("Fonction de transfert électrique du filtre (V_out / V_in)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Atténuation (dB)")
        plt.savefig(filename_filter)
        plt.close()

    def draw_schematic(self, individual, filename="schematic.png"):
        """Méthode dédiée pour sauvegarder le schéma où on le souhaite."""
        renderer = SchematicRenderer(individual['tree'])
        renderer.save(filename)
        
    def _get_off_axis_H(self, way, angle):
        """Charge un fichier FRD d'un autre angle et calcule le H_acoustic 3D"""
        # On remplace magiquement '0deg' par '15deg', '30deg', etc. dans le chemin
        off_axis_path = way.frd_path.replace('0deg', angle)
        
        if not os.path.exists(off_axis_path):
            return None
            
        try:
            # 1. Chargement des données brutes
            frd_data = np.loadtxt(off_axis_path)
            frd_freqs = frd_data[:, 0]
            mag_db = frd_data[:, 1]
            ph_unwrapped = np.unwrap(np.deg2rad(frd_data[:, 2]))
            
            # 2. Interpolation sur nos 400 fréquences
            mag_interp = np.interp(self.freqs, frd_freqs, mag_db)
            ph_interp = np.interp(self.freqs, frd_freqs, ph_unwrapped)
            H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
            
            # 3. Application du Moteur Géométrique 3D
            x_mm = getattr(way, 'x_offset', 0.0)
            y_mm = getattr(way, 'y_offset', 0.0)
            z_mm = getattr(way, 'z_offset', 0.0)
            listen_dist_mm = 2000.0
            
            dist_to_mic_mm = np.sqrt(x_mm**2 + y_mm**2 + (listen_dist_mm - z_mm)**2)
            path_diff_m = (dist_to_mic_mm - listen_dist_mm) / 1000.0
            delay_s = path_diff_m / 343.0
            
            phase_delay = np.exp(-1j * 2 * np.pi * self.freqs * delay_s)
            H_acoustic *= phase_delay
            
            return H_acoustic
        except Exception as e:
            print(f"[-] Erreur lecture {off_axis_path}: {e}")
            return None

    def plot_directivity(self, individual, filename="directivity.png"):
        """Évalue et dessine la réponse globale pour tous les angles"""
        plt.figure(figsize=(12, 8))
        root = individual['tree']
        angles = ['0deg', '15deg', '30deg', '45deg']
        colors = ['red', 'orange', 'green', 'blue']
        alphas = [1.0, 0.8, 0.6, 0.5]
        labels = ['0° (Axe)', '15°', '30°', '45°']
        
        # 1. Sauvegarde des H_acoustic originaux (0deg) pour ne pas casser le circuit
        original_H = {way.label: way.driver.H_acoustic.copy() for way in self.ways}
        
        for idx, angle in enumerate(angles):
            valid_angle = True
            
            # 2. Remplacement des H_acoustic dans l'arbre pour cet angle
            for way in self.ways:
                new_H = original_H[way.label] if angle == '0deg' else self._get_off_axis_H(way, angle)
                if new_H is None:
                    valid_angle = False
                    break
                    
                for node in root.get_all_nodes():
                    if isinstance(node, DriverNode) and node.label == way.label:
                        node.H_acoustic = new_H
                        
            if not valid_angle:
                continue # On passe cet angle s'il manque des fichiers
                
            # 3. Évaluation du circuit avec les courbes modifiées
            res = self.evaluator.evaluate(root)
            p_sum = np.zeros_like(self.freqs, dtype=complex)
            for way in self.ways:
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_sum += p_real
                
            spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
            plt.semilogx(self.freqs, spl_sum, label=labels[idx], color=colors[idx], linewidth=3 if idx==0 else 2, alpha=alphas[idx])
            
        # 4. Restauration de l'état d'origine
        for way in self.ways:
            for node in root.get_all_nodes():
                if isinstance(node, DriverNode) and node.label == way.label:
                    node.H_acoustic = original_H[way.label]
                    
        plt.axhline(self.target_spl, color='black', linestyle='--', alpha=0.3, label="Cible SPL")
        plt.ylim(self.target_spl - 40, self.target_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.title(f"Réponse en directivité SPL (Off-Axis)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("SPL (dB)")
        plt.savefig(filename)
        plt.close()
        
        

if __name__ == "__main__":
    start_time = time.time()
    config = [
        WayConfig("Woofer", r"crossovers\ER18RNX+27TDFC\SEAS_H1456-08_ER18RNX_SPL.frd", r"crossovers\ER18RNX+27TDFC\SEAS_H1456-08_ER18RNX_ZR.zma",
                  z_offset_m=0.00, y_offset=-100, x_offset=0.00),
        WayConfig("Tweeter", r"crossovers\ER18RNX+27TDFC\Tweeter_SPL.frd", r"crossovers\ER18RNX+27TDFC\Tweeter_ZR.zma")
    ]

    opt = CrossoverOptimizer(config)
    best = opt.run(generations=100, pop_size=120)
    best['tree'].display()
    end_time = time.time()
    print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")