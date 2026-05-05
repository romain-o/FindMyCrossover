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


BOUNDS_R = (0.1, 220.0)
BOUNDS_C = (0.1e-6, 300e-6)
BOUNDS_L = (0.05e-3, 5e-3)

E24_SERIES = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 
                       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])

# WEIGHTS = {
#     # Poids de la cible principale
#     'mse_sum': 1.0,           # Ancre de calcul (1 point = 1 dB d'erreur quadratique)
#     'mean_spl': 0.01,         # Bonus très léger pour les sensibilités globales élevées
#     'crossover': 3.0,         # Multiplicateur d'importance de la zone de raccord
    
#     # Seuils fixes
#     'n_comps': 8,             # Nombre idéal max de composants
    
#     # Multiplicateurs des Gradients (Pentes de pénalité)
#     'fc_err': 100.0,          # Pénalité pour l'écart de fréquence de coupure (octave_err ^ 2)
#     'impedance': 100.0,       # Pénalité d'impédance sous 3.2 Ohms (diff ^ 3)
#     'tweeter_low': 20.0,      # Pénalité pour le tweeter qui joue du grave (diff ^ 2)
#     'woofer_attenuation': 2500.0, # Pénalité si le woofer est bridé (diff ^ 3)
#     'thermal': 0.1,           # Pénalité thermique au-delà de 20W (diff ^ 2)
#     'components': 0.5,        # Pénalité par composant supplémentaire (excess ^ 2.5)
#     'resistors': 0.1          # Pénalité linéaire par résistance présente
# }

WEIGHTS = {
    'crossover': 3.2549,
    'fc_err': 20.0,
    'impedance': 142.9232,
    'tweeter_low': 48.4022,
    'woofer_attenuation': 218.2858,
    'thermal': 0.1216,
    'components': 0.6808,
    'resistors': 0.6518,
    'mse_sum': 1.0000,
    'n_comps': 8,
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
    def __init__(self, ways_configs, target_fc=0.0, weights=None):
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
        
        global WEIGHTS
        self.weights = weights if weights is not None else WEIGHTS
                            
        self.mask_ref = (self.freqs > 200) & (self.freqs < 1000)
        
        for way in self.ways:
            self._prepare_driver(way)
        
        raw_w_mag = np.abs(self.ways[0].driver.H_acoustic)
        raw_w_spl = 20 * np.log10(raw_w_mag + 1e-12)
        raw_avg = np.mean(raw_w_spl[self.mask_ref]) if np.any(self.mask_ref) else self.target_spl
        self.target_spl = raw_avg - 0.7
        self.target_fc = target_fc
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
            f_min = max(f_min, 80.0)
            f_max = min(f_max, 18000.0)
        else:
            f_min, f_max = 80, 18000
            
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
        raw_name = os.path.basename(way.frd_path).split('.')[0].split('@')[0]
        d.model_name = raw_name.replace('_0deg', '')
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

    def fitness(self, individual, return_components=False):
        # ============================================================
        # OPTIMISATION #3 : Cache du score de fitness
        # → Un individu non modifié n'est JAMAIS réévalué.
        #   Le cache est invalidé par optimize_values() et
        #   optimize_e24_values() dès qu'une valeur change.
        #   Les nouveaux enfants (mutés/croisés) n'ont pas la clé
        #   '_cached_score', donc sont toujours évalués.
        # ============================================================

        root = individual['tree']
        
        for comp in root.get_all_nodes():
            if isinstance(comp, Resistor):
                comp.value = float(np.clip(comp.value, BOUNDS_R[0], BOUNDS_R[1]))
            elif isinstance(comp, Capacitor):
                comp.value = float(np.clip(comp.value, BOUNDS_C[0], BOUNDS_C[1]))
            elif isinstance(comp, Inductor):
                comp.value = float(np.clip(comp.value, BOUNDS_L[0], BOUNDS_L[1]))

        if not isinstance(root, ParallelNode): 
            return 1e9

        res = self.evaluator.evaluate(root)
        total_mse = 0.0

        best_score_sum = float('inf')
        
        p_sum_test = np.zeros_like(self.freqs, dtype=complex)
        p_ways = []
        
        for i, way in enumerate(self.ways):
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_ways.append(p_real)
            p_sum_test += p_real
            
        spl_sum_test = 20 * np.log10(np.abs(p_sum_test) + 1e-12)
        dynamic_spl = self.target_spl
        # dynamic_spl = spl_sum_test[self.mask_ref].mean()
        diff = spl_sum_test - dynamic_spl
        
        # --- NOUVEAU : DICTIONNAIRE DE TRACKING ---
        comps_track = {
            'MSE_SPL': 0.0,
            'FC_Penalty': 0.0,
            'Impedance_Penalty': 0.0,
            'Tweeter_LowFreq_Penalty': 0.0,
            'Woofer_Attenuation_Penalty': 0.0, # <-- NOUVEAU
            'Thermal_Penalty': 0.0,            # <-- NOUVEAU
            'Component_Count_Penalty': 0.0,
            'Resistor_Count_Penalty': 0.0
        }
        
        dynamic_weight = self._base_weight.copy()
        
        for j in range(len(self.ways) - 1):
            mag1 = 20 * np.log10(np.abs(p_ways[j]) + 1e-12)
            mag2 = 20 * np.log10(np.abs(p_ways[j+1]) + 1e-12)
            
            if len(self.ways) == 2:
                search_mask = (self.freqs > 800) & (self.freqs < 5000)
            else:
                if j == 0: search_mask = (self.freqs > 150) & (self.freqs < 1000)
                elif j == 1: search_mask = (self.freqs > 1000) & (self.freqs < 8000)
                else: search_mask = (self.freqs > 2000) & (self.freqs < 12000)
            
            if np.any(search_mask):
                m1_sub = mag1[search_mask]
                m2_sub = mag2[search_mask]
                f_sub = self.freqs[search_mask]
                
                cross_points = np.where(m2_sub > m1_sub)[0]
                if len(cross_points) > 0: idx_cross = cross_points[0]
                else: idx_cross = np.argmin(np.abs(m1_sub - m2_sub))
                    
                f_cross = f_sub[idx_cross]
                dynamic_weight[(self.freqs > f_cross / 2.0) & (self.freqs < f_cross * 2.0)] = self.weights['crossover']

                if getattr(self, 'target_fc', 0.0) > 0.0:
                    octave_err = (np.log2(f_cross / self.target_fc))
                    # Rangement dans le dictionnaire
                    comps_track['FC_Penalty'] = (octave_err ** 2) * self.weights['fc_err']
                
        raw_mse = np.mean(np.where(diff > 0, (diff**2) * 1.5, diff**2) * dynamic_weight)
        comps_track['MSE_SPL'] = (raw_mse * self.weights['mse_sum'])
        
        # ==========================================
        # 1. GRADIENT D'IMPÉDANCE (Limite 3.2 Ω)
        # ==========================================
        Z_in = self.evaluator.get_impedance(root)
        min_Z = np.min(np.abs(Z_in))
        if min_Z < 3.2: 
            # --- MODIFIÉ ---
            comps_track['Impedance_Penalty'] += ((3.2 - min_Z) ** 3) * self.weights['impedance']
        
        # ==========================================
        # 2. GRADIENT DE SÉCURITÉ DU TWEETER
        # ==========================================
        last_way_v = res.get(self.ways[-1].label, {}).get("V_complex", np.zeros_like(self.freqs))
        v_low = np.abs(last_way_v)[self.freqs < 1000.0]
        v_excess = np.maximum(0, v_low - 0.1)
        if np.any(v_excess > 0):
            # --- MODIFIÉ ---
            comps_track['Tweeter_LowFreq_Penalty'] += np.sum(v_excess ** 2) * self.weights['tweeter_low']

        # ==========================================
        # 3. GRADIENT ANTI-ATTÉNUATION WOOFER
        # ==========================================
        v_woofer = res.get(self.ways[0].label, {}).get("V_complex", np.zeros_like(self.freqs))
        max_w_gain = np.max(np.abs(v_woofer))
        if max_w_gain < 0.95:  
            # --- MODIFIÉ ---
            comps_track['Woofer_Attenuation_Penalty'] += ((0.95 - max_w_gain) ** 3) * self.weights['woofer_attenuation']

        # ==========================================
        # 4. GRADIENT THERMIQUE (Limite 20W)
        # ==========================================
        if hasattr(self, '_get_max_power_dissipation'):
            V_amp_test = np.full_like(self.freqs, 28.28, dtype=complex)
            max_resistor_power = self._get_max_power_dissipation(root, V_amp_test)
            
            if max_resistor_power > 20.0:
                # --- MODIFIÉ ---
                comps_track['Thermal_Penalty'] += ((max_resistor_power - 20.0) ** 2) * self.weights['thermal']

        # ==========================================
        # 5. GRADIENT DU NOMBRE DE COMPOSANTS
        # ==========================================
        all_nodes = root.get_all_nodes()
        comps = [n for n in all_nodes if isinstance(n, ComponentNode)]
        n_comps = len(comps)

        if n_comps <= self.weights['n_comps']:
            comps_track['Component_Count_Penalty'] = 0.0
        else:
            excess = n_comps - self.weights['n_comps']
            # --- MODIFIÉ ---
            comps_track['Component_Count_Penalty'] = (excess ** 2.5) * self.weights['components']

        n_resistors = sum(1 for c in comps if isinstance(c, Resistor))
        # --- MODIFIÉ ---
        comps_track['Resistor_Count_Penalty'] = n_resistors * self.weights['resistors']

        # ==========================================
        # CALCUL FINAL
        # ==========================================
        final_score = sum(comps_track.values())

        if return_components:
            return final_score, comps_track
            
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
            if len(self.ways) == 2:
                try:
                    seeds = []
                    
                    # Raccourcis pour la lisibilité
                    w_drv = lambda: self.ways[0].driver.copy()
                    t_drv = lambda: self.ways[1].driver.copy()

                    # Template 1 : 1er Ordre (Très simple, 6dB/oct)
                    w1 = SeriesNode(Inductor(1.0e-3), w_drv())
                    t1 = SeriesNode(Capacitor(4.7e-6), t_drv())
                    seeds.append(ParallelNode(w1, t1))

                    # Template 2 : 2ème Ordre Classique (12dB/oct)
                    w2 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(10e-6), w_drv()))
                    t2 = SeriesNode(Capacitor(5.6e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w2, t2))

                    # Template 3 : 3ème Ordre (La graine d'origine)
                    w3 = SeriesNode(Inductor(1.5e-3), ParallelNode(Capacitor(10e-6), SeriesNode(Inductor(0.5e-3), w_drv())))
                    t3 = SeriesNode(Capacitor(4.7e-6), ParallelNode(Inductor(0.3e-3), SeriesNode(Capacitor(10e-6), t_drv())))
                    seeds.append(ParallelNode(w3, t3))

                    # Template 4 : L'Arme Secrète (Filtre Bouchon en série sur le Woofer + 2ème Ordre)
                    notch = ParallelNode(Capacitor(15e-6), Inductor(0.1e-3))
                    w4 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(8.2e-6), SeriesNode(notch, w_drv())))
                    t4 = SeriesNode(Capacitor(5.6e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w4, t4))

                    # Template 5 : Tweeter Atténué (L-Pad) + 2ème Ordre
                    lpad_t = SeriesNode(Resistor(3.3), ParallelNode(Resistor(10.0), t_drv()))
                    w5 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(10e-6), w_drv()))
                    t5 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.4e-3), lpad_t))
                    seeds.append(ParallelNode(w5, t5))

                    # Injection des graines dans la population
                    print(f"[+] Injection de {len(seeds)} templates fondamentaux dans la population.")
                    for s in seeds:
                        # On ajoute la graine pure
                        population.append({'tree': s, 'is_optimized': False})
                        
                        # On ajoute 10 versions légèrement mutées de cette graine 
                        # (change les valeurs, ajoute/retire un composant)
                        for _ in range(10):
                            mutated_s = self.mutator.mutate(s.copy())
                            population.append({'tree': mutated_s, 'is_optimized': False})
                            
                    print(f"[+] {len(seeds) * 10} mutants de première génération créés avec succès.")
                            
                except Exception as e:
                    print(f"[-] Erreur lors de l'injection des graines : {e}")
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
                
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                
                # On prend les meilleurs 10%
                top_10_count = max(1, pop_size // 10)
                top_inds = [s[1] for s in scores[:top_10_count]]
                
                # On re-calcule silencieusement pour obtenir les dictionnaires
                gen_comps = []
                for ind in top_inds:
                    _, comps = self.fitness(ind, return_components=True)
                    gen_comps.append(comps)
                
                # On fait la moyenne de chaque pénalité pour cette génération
                avg_comps = {k: np.mean([c[k] for c in gen_comps]) for k in gen_comps[0].keys()}
                self.loss_history.append(avg_comps)
                # ----------------------------------------------------
                
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

    def plot_result(self, individual, filename_response="response.png", filename_filter="filter.png"):
        """Génère les graphiques SPL et de transfert électrique en anglais"""
        root = individual['tree']
        res = self.evaluator.evaluate(root)
        
        # --- GRAPH 1 : SPL RESPONSE ---
        plt.figure(figsize=(12, 7))
        p_sum = np.zeros_like(self.freqs, dtype=complex)
        
        for way in self.ways:
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_sum += p_real
            spl_real = 20 * np.log10(np.abs(p_real) + 1e-12)
            
            # Utilisation du nom exact du haut-parleur (ex: "RS150-8")
            plt.semilogx(self.freqs, spl_real, label=way.driver.model_name, linewidth=2)

        spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
        plt.semilogx(self.freqs, spl_sum, label="System Sum", color='red', linewidth=3)
        plt.axhline(self.target_spl, color='green', linestyle='--', alpha=0.5, label="Target SPL")
            
        # --- NOUVEAU : Calcul et affichage de l'écart Min-Max (300Hz - 17kHz) ---
        mask_range = (self.freqs >= 300) & (self.freqs <= 17000)
        if np.any(mask_range):
            spl_in_range = spl_sum[mask_range]
            spl_diff = np.max(spl_in_range) - np.min(spl_in_range)
            
            # L'argument 'transform=plt.gca().transAxes' permet d'utiliser des pourcentages (0 à 1) 
            # de la taille de l'écran plutôt que les vraies valeurs des axes X et Y.
            plt.text(0.02, 0.95, f"Ripple (300Hz-17kHz): {spl_diff:.1f} dB", 
                     transform=plt.gca().transAxes, 
                     fontsize=11, fontweight='bold', color='black',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
        # -------------------------------------------------------------------------
        
        plt.title(f"System SPL Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.xlim(20, 20000)
        plt.ylim(self.target_spl - 30, self.target_spl + 10)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.savefig(filename_response)
        plt.close()

        # --- GRAPH 2 : ELECTRICAL TRANSFER ---
        plt.figure(figsize=(12, 8))
        for way in self.ways:
            v_complex = res.get(way.label, {}).get("V_complex", np.zeros_like(self.freqs))
            filter_db = 20 * np.log10(np.abs(v_complex) + 1e-12)
            plt.semilogx(self.freqs, filter_db, label=f"{way.driver.model_name} Filter", linewidth=2)
            
        plt.title("Electrical Transfer Functions")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.xlim(20, 20000)
        plt.ylim(-40, 5)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.savefig(filename_filter)
        plt.close()

    def plot_directivity(self, individual, filename="directivity.png"):
        """Génère le graphique de directivité avec légendes anglaises"""
        plt.figure(figsize=(12, 7))
        root = individual['tree']
        angles = ['0deg', '15deg', '30deg', '45deg']
        colors = ['red', 'orange', 'green', 'blue']
        alphas = [1.0, 0.8, 0.6, 0.5]
        
        # Légendes traduites
        labels = ['0° (On-Axis)', '15°', '30°', '45°']
        
        original_H = {way.label: way.driver.H_acoustic.copy() for way in self.ways}
        
        for idx, angle in enumerate(angles):
            valid_angle = True
            
            for way in self.ways:
                new_H = original_H[way.label] if angle == '0deg' else self._get_off_axis_H(way, angle)
                if new_H is None:
                    valid_angle = False
                    break
                for node in root.get_all_nodes():
                    if isinstance(node, DriverNode) and node.label == way.label:
                        node.H_acoustic = new_H
                        
            if not valid_angle:
                continue 
                
            res = self.evaluator.evaluate(root)
            p_sum = np.zeros_like(self.freqs, dtype=complex)
            for way in self.ways:
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_sum += p_real
                
            spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
            plt.semilogx(self.freqs, spl_sum, label=labels[idx], color=colors[idx], linewidth=3 if idx==0 else 2, alpha=alphas[idx])
            
        for way in self.ways:
            for node in root.get_all_nodes():
                if isinstance(node, DriverNode) and node.label == way.label:
                    node.H_acoustic = original_H[way.label]
                    
        plt.axhline(self.target_spl, color='black', linestyle='--', alpha=0.3, label="Target SPL")
        plt.ylim(self.target_spl - 30, self.target_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        
        # Titres traduits
        plt.title("Off-Axis SPL Directivity")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        
        plt.savefig(filename)
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
        
    def plot_sonogram(self, individual, filename="sonogram.png"):
        """Génère un Sonogramme Ultra-Haute Définition, clone parfait de VituixCAD"""
        import matplotlib.ticker as ticker
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.interpolate import RectBivariateSpline
        import numpy as np
        
        root = individual['tree']
        
        # 1. On récolte les angles disponibles
        test_angles = [0, 15, 30, 45]
        valid_data = {}
        original_H = {way.label: way.driver.H_acoustic.copy() for way in self.ways}
        
        # 2. Simulation des réponses pour chaque angle
        for angle in test_angles:
            angle_str = f"{angle}deg"
            valid_angle = True
            
            for way in self.ways:
                new_H = original_H[way.label] if angle == 0 else self._get_off_axis_H(way, angle_str)
                if new_H is None:
                    valid_angle = False
                    break
                for node in root.get_all_nodes():
                    if isinstance(node, DriverNode) and node.label == way.label:
                        node.H_acoustic = new_H
                        
            if valid_angle:
                res = self.evaluator.evaluate(root)
                p_sum = np.zeros_like(self.freqs, dtype=complex)
                for way in self.ways:
                    p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                    p_sum += p_real
                spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
                valid_data[angle] = spl_sum
                
        # Restauration de l'état du circuit
        for way in self.ways:
            for node in root.get_all_nodes():
                if isinstance(node, DriverNode) and node.label == way.label:
                    node.H_acoustic = original_H[way.label]
                    
        if len(valid_data) < 2:
            print("[-] Pas assez de données angulaires pour dessiner le Sonogramme HD.")
            return

        # 3. Création du miroir symétrique (de -45° à +45°)
        angles_raw = []
        spl_raw = []
        for angle in sorted(valid_data.keys(), reverse=True):
            if angle != 0:
                angles_raw.append(-angle)
                spl_raw.append(valid_data[angle])
        for angle in sorted(valid_data.keys()):
            angles_raw.append(angle)
            spl_raw.append(valid_data[angle])
            
        angles_raw = np.array(angles_raw)
        spl_matrix = np.array(spl_raw)

        # 4. INTERPOLATION MATHÉMATIQUE ULTRA-HD
        # kx=2 (Quadratique) : Lissage parfait sans exagérer les courbes entre les angles
        spline = RectBivariateSpline(angles_raw, self.freqs, spl_matrix, kx=2, ky=3)
        
        # Génération d'une grille dense (500 lignes d'angles x 1000 colonnes de fréquences)
        angles_hd = np.linspace(angles_raw[0], angles_raw[-1], 500)
        freqs_hd = np.geomspace(100, 20000, 1000)
        spl_hd = spline(angles_hd, freqs_hd)

        # 5. CRÉATION DE LA PALETTE DE COULEURS VITUIXCAD (512 Niveaux)
        vituix_colors = [
            (0.00, '#000000'),  # Noir pur (Silences)
            (0.14, '#000088'),  # Bleu Marine
            (0.28, '#0000FF'),  # Bleu
            (0.42, '#00FFFF'),  # Cyan
            (0.57, '#00FF00'),  # Vert
            (0.71, '#FFFF00'),  # Jaune
            (0.85, '#FF0000'),  # Rouge
            (1.00, '#550000')   # Bordeaux Foncé (Pics SPL)
        ]
        vituix_cmap = LinearSegmentedColormap.from_list('vituix_pro', vituix_colors, N=512)

        # Échelles de décibels calquées sur Vituix (Dynamique de ~56 dB)
        max_spl = self.target_spl + 6
        min_spl = max_spl - 40
        
        # 6. DESSIN DU GRAPHIQUE
        fig, ax = plt.subplots(figsize=(12, 6))
        X, Y = np.meshgrid(freqs_hd, angles_hd)
        
        # Le fond coloré ultra-lissé (200 paliers thermiques pour détruire l'effet de "bandes")
        c = ax.contourf(X, Y, spl_hd, levels=np.linspace(min_spl, max_spl, 200), 
                        cmap=vituix_cmap, extend='both', antialiased=True)
        
        # Les isocontours stricts (lignes noires fines tous les 3 dB)
        levels_3db = np.arange(int(min_spl), int(max_spl) + 1, 3)
        ax.contour(X, Y, spl_hd, levels=levels_3db, colors='black', linewidths=0.5, alpha=0.7, antialiased=True)

        # 7. STYLISATION VITUIXCAD STRICTE
        ax.set_xscale('log')
        ax.set_xlim(100, 20000)
        ax.set_ylim(angles_raw[0], angles_raw[-1])
        
        # Axe X : Fréquences formatées (1k, 2k...)
        ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        def format_freq(x, pos):
            if x == 100: return '100Hz'
            elif x >= 1000: return f'{int(x/1000)}k'
            else: return f'{int(x)}'
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_freq))
        
        # Axe Y : Angles placés à droite
        ax.set_yticks(angles_raw)
        ax.yaxis.tick_right() 
        ax.set_ylabel('deg', loc='top', rotation=0, labelpad=-20)
        
        # Grilles subtiles
        ax.grid(True, which='major', color='white', alpha=0.3, linewidth=0.5)
        ax.grid(True, which='minor', color='white', alpha=0.1, linewidth=0.3)
        ax.set_title('Directivity (hor)', pad=10)

        # 8. COLORBAR À GAUCHE
        plt.subplots_adjust(left=0.15, right=0.95) 
        cbar_ax = fig.add_axes([0.05, 0.15, 0.02, 0.7]) # [Gauche, Bas, Largeur, Hauteur]
        cbar = fig.colorbar(c, cax=cbar_ax, ticks=np.arange(int(min_spl), int(max_spl), 8))
        cbar.ax.set_title('dB', pad=10)
        cbar.ax.yaxis.set_ticks_position('left')
        
        # Export en haute résolution
        plt.savefig(filename, dpi=200, bbox_inches='tight') 
        plt.close()
        
    def plot_loss_history(self, filename="loss_history.png"):
        """Génère un graphique à barres empilées de l'évolution des composantes de la loss."""
        if not hasattr(self, 'loss_history') or not self.loss_history:
            print("[-] Aucune donnée de loss à afficher.")
            return

        generations = np.arange(len(self.loss_history))
        keys = list(self.loss_history[0].keys())
        
        plt.figure(figsize=(12, 7))
        bottom = np.zeros(len(generations))
        
        # Palette de couleurs professionnelle et distincte
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, key in enumerate(keys):
            values = np.array([gen[key] for gen in self.loss_history])
            # width=1.0 permet aux barres de se toucher et de créer un effet "bloc" (stacked area continu)
            plt.bar(generations, values, bottom=bottom, width=1.0, 
                    label=key.replace('_', ' '), color=colors[i % len(colors)], edgecolor='none')
            bottom += values
            
        plt.title("Evolution of Loss Components (Top 10% Average)", fontsize=14, fontweight='bold')
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Absolute Loss Score", fontsize=12)
        
        # --- Cadrage Intelligent ---
        # On ignore les pics massifs des toutes premières générations pour rendre la fin lisible
        focus_idx = max(0, int(len(generations) * 0.15)) 
        if len(bottom) > focus_idx:
            max_y = np.max(bottom[focus_idx:]) * 1.5 
            plt.ylim(0, max_y)
        
        plt.xlim(0, len(generations) - 1)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # On sort la légende du graphique pour ne pas cacher les données
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _calc_node_impedance(self, node):
        """Calcule l'impédance complexe équivalente d'un nœud ou sous-arbre."""
        name = node.__class__.__name__
        
        # Composants passifs
        if name == "Resistor":
            return np.full_like(self.freqs, node.value, dtype=complex)
        elif name == "Capacitor":
            return 1.0 / (1j * 2 * np.pi * self.freqs * node.value + 1e-15)
        elif name == "Inductor":
            return 1j * 2 * np.pi * self.freqs * node.value
            
        # Nœud Haut-Parleur : On récupère la vraie courbe ZMA mesurée
        elif name == "DriverNode":
            for way in self.ways:
                if way.label == node.label:
                    if hasattr(way.driver, 'Z_complex'): return way.driver.Z_complex
                    if hasattr(way.driver, 'Z'): return way.driver.Z
            return np.full_like(self.freqs, 8.0, dtype=complex) # Fallback 8 Ohms
            
        # Lois de Kirchhoff (Série et Parallèle)
        elif name == "SeriesNode":
            return self._calc_node_impedance(node.left) + self._calc_node_impedance(node.right)
        elif name == "ParallelNode":
            z1 = self._calc_node_impedance(node.left)
            z2 = self._calc_node_impedance(node.right)
            return 1.0 / (1.0 / (z1 + 1e-15) + 1.0 / (z2 + 1e-15))
            
        return np.full_like(self.freqs, 1e6, dtype=complex)
    
    def _get_max_power_dissipation(self, node, V_in):
        """
        Calcule la tension aux bornes de chaque composant de manière récursive 
        et retourne la puissance dissipée maximale (en Watts) par une résistance du circuit.
        """
        name = node.__class__.__name__
        
        if name == "Resistor":
            # P = |U|^2 / R (On prend la pire dissipation sur toute la plage de fréquences)
            power_array = (np.abs(V_in)**2) / node.value
            return np.max(power_array)
            
        elif name in ["Capacitor", "Inductor", "DriverNode"]:
            # Les composants purement réactifs et les HP ne "brûlent" pas le filtre
            return 0.0
            
        elif name == "ParallelNode":
            # En parallèle, la tension est identique sur les deux branches
            p_left = self._get_max_power_dissipation(node.left, V_in)
            p_right = self._get_max_power_dissipation(node.right, V_in)
            return max(p_left, p_right)
            
        elif name == "SeriesNode":
            # En série, on a un diviseur de tension complexe
            z_left = self._calc_node_impedance(node.left)
            z_right = self._calc_node_impedance(node.right)
            z_tot = z_left + z_right + 1e-15
            
            v_left = V_in * (z_left / z_tot)
            v_right = V_in * (z_right / z_tot)
            
            p_left = self._get_max_power_dissipation(node.left, v_left)
            p_right = self._get_max_power_dissipation(node.right, v_right)
            return max(p_left, p_right)
            
        return 0.0

    def plot_impedance(self, individual, filename="impedance.png"):
        """Génère le graphique de l'impédance globale du système (Module) en anglais."""
        import matplotlib.ticker as ticker
        
        root = individual['tree']
        
        # Calcul de l'impédance totale à l'entrée du filtre
        Z_in = self._calc_node_impedance(root)
        mag_Z = np.abs(Z_in)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 1. Axe principal : Module (Ohms)
        color1 = '#0077BB' # Bleu Pro
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Impedance (Ω)', color=color1, fontweight='bold')
        ax1.semilogx(self.freqs, mag_Z, color=color1, linewidth=2.5, label="Magnitude")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim(20, 20000)
        
        # Échelle intelligente (plafond à 60 Ohms max pour éviter d'écraser le graphe si forte résonance)
        y_max = min(60, max(20, np.max(mag_Z) * 1.1))
        ax1.set_ylim(0, y_max)
        
        # Formatage de l'axe X (100Hz, 1k, etc.)
        def format_freq(x, pos):
            if x == 100: return '100Hz'
            elif x >= 1000: return f'{int(x/1000)}k'
            else: return f'{int(x)}'
        ax1.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_freq))
        
        # Regroupement des légendes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        ax1.legend(lines_1, labels_1, loc='upper right')
            
        plt.title(f"System Impedance")
        fig.tight_layout()  
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