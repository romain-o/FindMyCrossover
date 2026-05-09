import numpy as np
import random
import json
import os
import itertools # <-- NOUVEL IMPORT POUR LE BRUTE FORCE
import csv
import matplotlib.pyplot as plt
from src.nodes import DriverNode, SeriesNode, ParallelNode, ShuntNode, Capacitor, Inductor, Resistor, Node, ComponentNode
from src.evaluator import CircuitEvaluator
from src.mutator import TreeMutator
from src.catalog_manager import CatalogManager
from scipy.optimize import minimize
from src.schematic import SchematicRenderer
import pandas as pd
import time

# ============================================================
# OPTIMISATION #1 : Fonctions module-level pour multiprocessing
# ============================================================
from multiprocessing import Pool, cpu_count

_pool_optimizer = None

def _pool_init(opt):
    global _pool_optimizer
    _pool_optimizer = opt

def _pool_fitness(ind):
    # NOUVEAU : On retourne aussi le wiring trouvé par le worker
    score = _pool_optimizer.fitness(ind)
    return score, ind.get('wiring', {})

def _pool_elite(args):
    return _pool_optimizer._elite_worker(args)

def _pool_lamarckian(child):
    return _pool_optimizer._lamarckian_worker(child)


BOUNDS_R = (0.1, 220.0)
BOUNDS_C = (0.1e-6, 300e-6)
BOUNDS_L = (0.05e-3, 5e-3)

CATALOG = CatalogManager()

WEIGHTS = {
    'crossover': 3.2549,
    'fc_err': 20.0,
    'impedance': 142.9232,
    'tweeter_low': 48.4022,
    'woofer_high': 10.0,
    'woofer_attenuation': 218.2858,
    'thermal': 0.1216,
    #'components': 0.6808,
    'components': 0.5,
    'resistors': 0.6518,
    'mse_sum': 1.0000,
    'n_comps': 9,
}

class WayConfig:
    """Configuration d'une voie acoustique (Grave, Médium, Aigu, etc.)"""
    # NOUVEAU : Ajout de l'argument 'count'
    def __init__(self, label, frd_path, zma_path, order=4, z_offset=0.0, y_offset=0.0, x_offset=0.0, count=1):
        self.label = label
        self.frd_path = frd_path
        self.zma_path = zma_path
        self.order = order
        self.z_offset = z_offset
        self.y_offset = y_offset 
        self.x_offset = x_offset
        self.count = count # <-- Nombre de drivers pour cette voie
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
        self.mask_ref_t = (self.freqs > 4000) & (self.freqs < 14000)
        
        for way in self.ways:
            self._prepare_driver(way)
            
        # On calcule le SPL de base en prenant en compte le count potentiel
        self.apply_wiring({w.label: 'parallel' for w in self.ways if getattr(w, 'count', 1) > 1})
        raw_w_mag = np.abs(self.ways[0].driver.H_acoustic)
        raw_w_spl = 20 * np.log10(raw_w_mag + 1e-12)
        raw_avg = np.mean(raw_w_spl[self.mask_ref]) if np.any(self.mask_ref) else self.target_spl
        
        raw_t_mag = np.abs(self.ways[1].driver.H_acoustic)
        raw_t_spl = 20 * np.log10(raw_t_mag + 1e-12)
        raw_t_avg = np.mean(raw_t_spl[self.mask_ref_t])
        
        self.target_spl = min(raw_avg, raw_t_avg) - 0.7
        self.target_fc = target_fc
        self.apply_wiring({}) # On restaure à 1
        
        print(f"[+] Cible SPL verrouillée à {self.target_spl:.1f} dB (Woofer brut estimé: {raw_avg:.1f} dB)")
        
        max_raw_spl = np.zeros_like(self.freqs)
        for way in self.ways:
            raw_mag = np.abs(way.driver.H_base)
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
            f_max = min(f_max, 20000.0)
        else:
            f_min, f_max = 80, 20000
        self.mask_flat = (self.freqs >= f_min) & (self.freqs <= f_max)
        print(f"[+] Plage d'optimisation auto-détectée : {int(f_min)} Hz - {int(f_max)} Hz")

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

        # NOUVEAU : Sauvegarde des réponses de base pour les calculs Série/Parallèle
        d.H_base = d.H_acoustic.copy()
        d.Z_base = d.Z_complex.copy()

    def apply_wiring(self, wiring_dict):
        """Applique dynamiquement les modifications Z et H selon le câblage choisi."""
        for way in self.ways:
            d = way.driver
            count = getattr(way, 'count', 1)
            if count > 1:
                w_type = wiring_dict.get(way.label, 'parallel')
                if w_type == 'series':
                    d.Z_complex = d.Z_base * count
                    d.H_acoustic = d.H_base * 1.0  # La sensibilité en tension (2.83V) reste identique
                else: # parallel
                    d.Z_complex = d.Z_base / count
                    d.H_acoustic = d.H_base * count # +6dB par doublement à tension constante
            else:
                if hasattr(d, 'Z_base'):
                    d.Z_complex = d.Z_base
                    d.H_acoustic = d.H_base

    def fitness(self, individual, return_components=False):
        root = individual['tree']
        
        # 1. Vérification des limites physiques
        for comp in root.get_all_nodes():
            if isinstance(comp, Resistor):
                comp.value = float(np.clip(comp.value, BOUNDS_R[0], BOUNDS_R[1]))
            elif isinstance(comp, Capacitor):
                comp.value = float(np.clip(comp.value, BOUNDS_C[0], BOUNDS_C[1]))
            elif isinstance(comp, Inductor):
                comp.value = float(np.clip(comp.value, BOUNDS_L[0], BOUNDS_L[1]))

        if not isinstance(root, ParallelNode): 
            return 1e9

        # 2. Détermination des combinaisons de câblage à tester (Brute Force)
        multi_ways = [w for w in self.ways if getattr(w, 'count', 1) > 1]
        if not multi_ways:
            combos = [{}]
        else:
            labels = [w.label for w in multi_ways]
            # options = [['series', 'parallel'] for _ in multi_ways]
            # combos = [dict(zip(labels, c)) for c in itertools.product(*options)]
            options = [['parallel'] for _ in multi_ways] 
            combos = [dict(zip(labels, c)) for c in itertools.product(*options)]

        best_final_score = float('inf')
        best_comps_track = None
        best_wiring = None

        # 3. Évaluation de chaque configuration
        for wiring in combos:
            # On applique les modifications Z et H virtuelles au DriverNode
            self.apply_wiring(wiring)
            
            res = self.evaluator.evaluate(root)
            
            p_sum_test = np.zeros_like(self.freqs, dtype=complex)
            p_ways = []
            
            for i, way in enumerate(self.ways):
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_ways.append(p_real)
                p_sum_test += p_real
                
            spl_sum_test = 20 * np.log10(np.abs(p_sum_test) + 1e-12)
            dynamic_spl = self.target_spl
            diff = spl_sum_test - dynamic_spl
            
            comps_track = {
                'MSE_SPL': 0.0,
                'FC_Penalty': 0.0,
                'Impedance_Penalty': 0.0,
                'Tweeter_LowFreq_Penalty': 0.0,
                'Woofer_HighFreq_Penalty': 0.0,
                'Woofer_Attenuation_Penalty': 0.0,
                'Thermal_Penalty': 0.0,
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
                        comps_track['FC_Penalty'] = (octave_err ** 2) * self.weights['fc_err']
                    
            raw_mse = np.mean(np.where(diff > 0, (diff**2)*3, diff**2) * dynamic_weight)
            comps_track['MSE_SPL'] = (raw_mse * self.weights['mse_sum'])
            
            Z_in = self.evaluator.get_impedance(root)
            min_Z = np.min(np.abs(Z_in))
            if min_Z < 3.2: 
                comps_track['Impedance_Penalty'] += ((3.2 - min_Z) ** 3) * self.weights['impedance']
            
            last_way_v = res.get(self.ways[-1].label, {}).get("V_complex", np.zeros_like(self.freqs))
            v_low = np.abs(last_way_v)[self.freqs < 1000.0]
            v_excess = np.maximum(0, v_low - 0.1)
            if np.any(v_excess > 0):
                comps_track['Tweeter_LowFreq_Penalty'] += np.sum(v_excess ** 2) * self.weights['tweeter_low']

            # ========================================== # 2. GRADIENT DE SÉCURITÉ DU WOOFER # ========================================== 
            last_way_v = res.get(self.ways[0].label, {}).get("V_complex", np.zeros_like(self.freqs))
            v_high = np.abs(last_way_v)[self.freqs > 2000 * 1.2]
            v_excess = np.maximum(0, v_high - 0.1)
            if np.any(v_excess > 0):
                comps_track['Woofer_HighFreq_Penalty'] += np.sum(v_excess ** 2) * self.weights['woofer_high']

            v_woofer = res.get(self.ways[0].label, {}).get("V_complex", np.zeros_like(self.freqs))
            max_w_gain = np.max(np.abs(v_woofer))
            if max_w_gain < 0.95:  
                comps_track['Woofer_Attenuation_Penalty'] += ((0.95 - max_w_gain) ** 3) * self.weights['woofer_attenuation']

            if hasattr(self, '_get_max_power_dissipation'):
                V_amp_test = np.full_like(self.freqs, 28.28, dtype=complex)
                max_resistor_power = self._get_max_power_dissipation(root, V_amp_test)
                
                if max_resistor_power > 20.0:
                    comps_track['Thermal_Penalty'] += ((max_resistor_power - 20.0) ** 2) * self.weights['thermal']

            all_nodes = root.get_all_nodes()
            comps = [n for n in all_nodes if isinstance(n, ComponentNode)]
            n_comps = len(comps)

            if n_comps <= self.weights['n_comps']:
                comps_track['Component_Count_Penalty'] = 0.0
            else:
                excess = n_comps - self.weights['n_comps']
                comps_track['Component_Count_Penalty'] = (excess ** 2.5) * self.weights['components']

            n_resistors = sum(1 for c in comps if isinstance(c, Resistor))
            comps_track['Resistor_Count_Penalty'] = n_resistors * self.weights['resistors']

            final_score = sum(comps_track.values())
            
            # 4. On garde le meilleur câblage
            if final_score < best_final_score:
                best_final_score = final_score
                best_comps_track = comps_track
                best_wiring = wiring

        # 5. On restaure le meilleur câblage dans l'arbre physique pour l'individu
        if best_wiring is not None:
            individual['wiring'] = best_wiring
            self.apply_wiring(best_wiring)
        else:
            self.apply_wiring({})

        if return_components:
            return best_final_score, best_comps_track
            
        return best_final_score


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
                    comp.value = CATALOG.snap_to_catalog(comp.value, CATALOG.get_comp_type(comp))
            self.optimize_catalog_values(ind)
            ind['is_optimized'] = False
            
        return (self.fitness(ind), ind)

    def _lamarckian_worker(self, child_ind):
        if random.random() < 0.40:
            self.optimize_values(child_ind, max_iter=8)
            child_ind['is_optimized'] = True
        return child_ind

    def optimize_catalog_values(self, individual):
        individual.pop('_cached_score', None)
        comps = [n for n in individual['tree'].get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        best_score = self.fitness(individual)
        improved = True
        
        while improved:
            improved = False
            for comp in comps:
                ctype = CATALOG.get_comp_type(comp)
                original_val = CATALOG.snap_to_catalog(comp.value, ctype)
                
                if ctype == 'C': arr = CATALOG.vals_c
                elif ctype == 'L': arr = CATALOG.vals_l
                else: arr = CATALOG.vals_r
                
                idx = np.abs(arr - original_val).argmin()
                best_comp_val = original_val
                
                for step in [-1, 1]:
                    new_idx = idx + step
                    if 0 <= new_idx < len(arr):
                        test_val = arr[new_idx]
                        comp.value = test_val
                        
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
        return individual

    def run(self, generations=50, pop_size=60, checkpoint_path=None):
        population = []
        
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
                    
                    w_drv = lambda: self.ways[0].driver.copy()
                    t_drv = lambda: self.ways[1].driver.copy()

                    w1 = SeriesNode(Inductor(1.0e-3), w_drv())
                    t1 = SeriesNode(Capacitor(4.7e-6), t_drv())
                    seeds.append(ParallelNode(w1, t1))

                    w2 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(10e-6), w_drv()))
                    t2 = SeriesNode(Capacitor(5.6e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w2, t2))

                    w3 = SeriesNode(Inductor(1.5e-3), ParallelNode(Capacitor(10e-6), SeriesNode(Inductor(0.5e-3), w_drv())))
                    t3 = SeriesNode(Capacitor(4.7e-6), ParallelNode(Inductor(0.3e-3), SeriesNode(Capacitor(10e-6), t_drv())))
                    seeds.append(ParallelNode(w3, t3))

                    notch = ParallelNode(Capacitor(15e-6), Inductor(0.1e-3))
                    w4 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(8.2e-6), SeriesNode(notch, w_drv())))
                    t4 = SeriesNode(Capacitor(5.6e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w4, t4))

                    lpad_t = SeriesNode(Resistor(3.3), ParallelNode(Resistor(10.0), t_drv()))
                    w5 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(10e-6), w_drv()))
                    t5 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.4e-3), lpad_t))
                    seeds.append(ParallelNode(w5, t5))

                    print(f"[+] Injection de {len(seeds)} templates fondamentaux dans la population.")
                    for s in seeds:
                        population.append({'tree': s, 'is_optimized': False})
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
                # NOUVEAU : Récupération du tuple (score, wiring) depuis _pool_fitness
                fitness_results = pool.map(_pool_fitness, population, chunksize=chunksize)
                
                scores = []
                for (fit, wiring), ind in zip(fitness_results, population):
                    if wiring is not None:
                        ind['wiring'] = wiring
                    scores.append((fit, ind))
                    
                scores.sort(key=lambda x: x[0])
                
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                
                top_10_count = max(1, pop_size // 10)
                top_inds = [s[1] for s in scores[:top_10_count]]
                
                gen_comps = []
                for ind in top_inds:
                    _, comps = self.fitness(ind, return_components=True)
                    gen_comps.append(comps)
                
                avg_comps = {k: np.mean([c[k] for c in gen_comps]) for k in gen_comps[0].keys()}
                self.loss_history.append(avg_comps)
                
                if gen < int(generations * 0.4):      
                    max_opt_iter = 5
                    snap_to_standard = False
                elif gen < int(generations * 0.8):    
                    max_opt_iter = 12
                    snap_to_standard = False
                else:                                 
                    max_opt_iter = 20
                    snap_to_standard = True
                    
                if gen == int(generations * 0.8):
                    print("Passage en PHASE 3 (Standardisation CATALOGUE)")
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
                    
                    if checkpoint_path:
                        save_tree = best_ind['tree'].copy()
                        for comp in save_tree.get_all_nodes():
                            if isinstance(comp, ComponentNode):
                                comp.value = CATALOG.snap_to_catalog(comp.value, CATALOG.get_comp_type(comp))
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
            self.optimize_catalog_values(best_ind)
            
        for comp in best_ind['tree'].get_all_nodes():
            if isinstance(comp, ComponentNode):
                comp.value = CATALOG.snap_to_catalog(comp.value, CATALOG.get_comp_type(comp))
            elif isinstance(comp, DriverNode):
                way = next(w for w in self.ways if w.label == comp.label)
                comp.model_name = way.driver.model_name

        return best_ind

    def plot_result(self, individual, filename_response="response.png", filename_filter="filter.png"):
        self.apply_wiring(individual.get('wiring', {})) # NOUVEAU
        
        root = individual['tree']
        res = self.evaluator.evaluate(root)
        
        plt.figure(figsize=(12, 7))
        p_sum = np.zeros_like(self.freqs, dtype=complex)
        
        for way in self.ways:
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_sum += p_real
            spl_real = 20 * np.log10(np.abs(p_real) + 1e-12)
            plt.semilogx(self.freqs, spl_real, label=way.driver.model_name, linewidth=2)

        spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
        plt.semilogx(self.freqs, spl_sum, label="System Sum", color='red', linewidth=3)
        plt.axhline(self.target_spl, color='green', linestyle='--', alpha=0.5, label="Target SPL")
            
        mask_range = (self.freqs >= 300) & (self.freqs <= 17000)
        if np.any(mask_range):
            spl_in_range = spl_sum[mask_range]
            spl_diff = np.max(spl_in_range) - np.min(spl_in_range)
            plt.text(0.02, 0.95, f"Ripple (300Hz-17kHz): {spl_diff:.1f} dB", 
                     transform=plt.gca().transAxes, 
                     fontsize=11, fontweight='bold', color='black',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
                     
        plt.title(f"System SPL Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.xlim(20, 20000)
        plt.ylim(self.target_spl - 30, self.target_spl + 10)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.savefig(filename_response)
        plt.close()

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
        self.apply_wiring(individual.get('wiring', {})) # NOUVEAU
        
        plt.figure(figsize=(12, 7))
        root = individual['tree']
        angles = ['0deg', '15deg', '30deg', '45deg']
        colors = ['red', 'orange', 'green', 'blue']
        alphas = [1.0, 0.8, 0.6, 0.5]
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
                        
            if not valid_angle: continue 
                
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
        plt.title("Off-Axis SPL Directivity")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.savefig(filename)
        plt.close()

    def draw_schematic(self, individual, filename="schematic.png"):
        # On récupère l'arbre et on lui injecte le câblage trouvé
        root = individual['tree']
        root.wiring = individual.get('wiring', {})
        
        # INJECTION DE LA QUANTITÉ pour SchematicRenderer
        for node in root.get_all_nodes():
            if isinstance(node, DriverNode):
                way = next((w for w in self.ways if w.label == node.label), None)
                if way:
                    node.count = getattr(way, 'count', 1)
                    
        renderer = SchematicRenderer(root)
        renderer.save(filename)
        
    def _get_off_axis_H(self, way, angle):
        off_axis_path = way.frd_path.replace('0deg', angle)
        if not os.path.exists(off_axis_path): return None
            
        try:
            frd_data = np.loadtxt(off_axis_path)
            frd_freqs = frd_data[:, 0]
            mag_db = frd_data[:, 1]
            ph_unwrapped = np.unwrap(np.deg2rad(frd_data[:, 2]))
            
            mag_interp = np.interp(self.freqs, frd_freqs, mag_db)
            ph_interp = np.interp(self.freqs, frd_freqs, ph_unwrapped)
            H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
            
            x_mm = getattr(way, 'x_offset', 0.0)
            y_mm = getattr(way, 'y_offset', 0.0)
            z_mm = getattr(way, 'z_offset', 0.0)
            listen_dist_mm = 2000.0
            
            dist_to_mic_mm = np.sqrt(x_mm**2 + y_mm**2 + (listen_dist_mm - z_mm)**2)
            path_diff_m = (dist_to_mic_mm - listen_dist_mm) / 1000.0
            delay_s = path_diff_m / 343.0
            
            phase_delay = np.exp(-1j * 2 * np.pi * self.freqs * delay_s)
            H_acoustic *= phase_delay
            
            # Application de l'effet wiring off-axis
            count = getattr(way, 'count', 1)
            w_type = getattr(way.driver, 'current_wiring', 'parallel')
            if count > 1 and w_type == 'parallel':
                H_acoustic *= count
                
            return H_acoustic
        except Exception as e:
            return None
        
    def plot_sonogram(self, individual, filename="sonogram.png"):
        self.apply_wiring(individual.get('wiring', {})) # NOUVEAU
        
        import matplotlib.ticker as ticker
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.interpolate import RectBivariateSpline
        import numpy as np
        
        root = individual['tree']
        test_angles = [0, 15, 30, 45]
        valid_data = {}
        original_H = {way.label: way.driver.H_acoustic.copy() for way in self.ways}
        
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
                
        for way in self.ways:
            for node in root.get_all_nodes():
                if isinstance(node, DriverNode) and node.label == way.label:
                    node.H_acoustic = original_H[way.label]
                    
        if len(valid_data) < 2: return

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

        spline = RectBivariateSpline(angles_raw, self.freqs, spl_matrix, kx=2, ky=3)
        angles_hd = np.linspace(angles_raw[0], angles_raw[-1], 500)
        freqs_hd = np.geomspace(100, 20000, 1000)
        spl_hd = spline(angles_hd, freqs_hd)

        vituix_colors = [
            (0.00, '#000000'), (0.14, '#000088'), (0.28, '#0000FF'), (0.42, '#00FFFF'),
            (0.57, '#00FF00'), (0.71, '#FFFF00'), (0.85, '#FF0000'), (1.00, '#550000')
        ]
        vituix_cmap = LinearSegmentedColormap.from_list('vituix_pro', vituix_colors, N=512)

        max_spl = self.target_spl + 6
        min_spl = max_spl - 40
        
        fig, ax = plt.subplots(figsize=(12, 6))
        X, Y = np.meshgrid(freqs_hd, angles_hd)
        c = ax.contourf(X, Y, spl_hd, levels=np.linspace(min_spl, max_spl, 200), cmap=vituix_cmap, extend='both', antialiased=True)
        levels_3db = np.arange(int(min_spl), int(max_spl) + 1, 3)
        ax.contour(X, Y, spl_hd, levels=levels_3db, colors='black', linewidths=0.5, alpha=0.7, antialiased=True)

        ax.set_xscale('log')
        ax.set_xlim(100, 20000)
        ax.set_ylim(angles_raw[0], angles_raw[-1])
        ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        def format_freq(x, pos):
            if x == 100: return '100Hz'
            elif x >= 1000: return f'{int(x/1000)}k'
            else: return f'{int(x)}'
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_freq))
        
        ax.set_yticks(angles_raw)
        ax.yaxis.tick_right() 
        ax.set_ylabel('deg', loc='top', rotation=0, labelpad=-20)
        ax.grid(True, which='major', color='white', alpha=0.3, linewidth=0.5)
        ax.grid(True, which='minor', color='white', alpha=0.1, linewidth=0.3)
        ax.set_title('Directivity (hor)', pad=10)

        plt.subplots_adjust(left=0.15, right=0.95) 
        cbar_ax = fig.add_axes([0.05, 0.15, 0.02, 0.7]) 
        cbar = fig.colorbar(c, cax=cbar_ax, ticks=np.arange(int(min_spl), int(max_spl), 8))
        cbar.ax.set_title('dB', pad=10)
        cbar.ax.yaxis.set_ticks_position('left')
        plt.savefig(filename, dpi=200, bbox_inches='tight') 
        plt.close()
        
    def plot_loss_history(self, filename="loss_history.png"):
        if not hasattr(self, 'loss_history') or not self.loss_history: return
        generations = np.arange(len(self.loss_history))
        keys = list(self.loss_history[0].keys())
        plt.figure(figsize=(12, 7))
        bottom = np.zeros(len(generations))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, key in enumerate(keys):
            values = np.array([gen[key] for gen in self.loss_history])
            plt.bar(generations, values, bottom=bottom, width=1.0, label=key.replace('_', ' '), color=colors[i % len(colors)], edgecolor='none')
            bottom += values
            
        plt.title("Evolution of Loss Components (Top 10% Average)", fontsize=14, fontweight='bold')
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Absolute Loss Score", fontsize=12)
        
        focus_idx = max(0, int(len(generations) * 0.15)) 
        if len(bottom) > focus_idx:
            max_y = np.max(bottom[focus_idx:]) * 1.5 
            plt.ylim(0, max_y)
        
        plt.xlim(0, len(generations) - 1)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _calc_node_impedance(self, node):
        name = node.__class__.__name__
        if name == "Resistor":
            return np.full_like(self.freqs, node.value, dtype=complex)
        elif name == "Capacitor":
            return 1.0 / (1j * 2 * np.pi * self.freqs * node.value + 1e-15)
        elif name == "Inductor":
            return 1j * 2 * np.pi * self.freqs * node.value
        elif name == "DriverNode":
            for way in self.ways:
                if way.label == node.label:
                    if hasattr(way.driver, 'Z_complex'): return way.driver.Z_complex
                    if hasattr(way.driver, 'Z'): return way.driver.Z
            return np.full_like(self.freqs, 8.0, dtype=complex)
        elif name == "SeriesNode":
            return self._calc_node_impedance(node.left) + self._calc_node_impedance(node.right)
        elif name == "ParallelNode":
            z1 = self._calc_node_impedance(node.left)
            z2 = self._calc_node_impedance(node.right)
            return 1.0 / (1.0 / (z1 + 1e-15) + 1.0 / (z2 + 1e-15))
        return np.full_like(self.freqs, 1e6, dtype=complex)
    
    def _get_max_power_dissipation(self, node, V_in):
        name = node.__class__.__name__
        if name == "Resistor":
            power_array = (np.abs(V_in)**2) / node.value
            return np.max(power_array)
        elif name in ["Capacitor", "Inductor", "DriverNode"]:
            return 0.0
        elif name == "ParallelNode":
            p_left = self._get_max_power_dissipation(node.left, V_in)
            p_right = self._get_max_power_dissipation(node.right, V_in)
            return max(p_left, p_right)
        elif name == "SeriesNode":
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
        self.apply_wiring(individual.get('wiring', {})) # NOUVEAU
        
        import matplotlib.ticker as ticker
        root = individual['tree']
        Z_in = self._calc_node_impedance(root)
        mag_Z = np.abs(Z_in)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color1 = '#0077BB' 
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Impedance (Ω)', color=color1, fontweight='bold')
        ax1.semilogx(self.freqs, mag_Z, color=color1, linewidth=2.5, label="Magnitude")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim(20, 20000)
        
        y_max = min(60, max(20, np.max(mag_Z) * 1.1))
        ax1.set_ylim(0, y_max)
        
        def format_freq(x, pos):
            if x == 100: return '100Hz'
            elif x >= 1000: return f'{int(x/1000)}k'
            else: return f'{int(x)}'
        ax1.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_freq))
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        ax1.legend(lines_1, labels_1, loc='upper right')
            
        plt.title(f"System Impedance")
        fig.tight_layout()  
        plt.savefig(filename)
        plt.close()
        
    def generate_parts_list(self, individual, filename="BOM_Parts_List.csv"):
        """Génère la nomenclature finale au format CSV et LaTeX."""
        comps = [n for n in individual['tree'].get_all_nodes() if isinstance(n, ComponentNode)]
        
        inventory = {}
        total_price = 0.0
        
        # ==========================================
        # 1. GÉNÉRATION DU CSV ET INVENTAIRE GLOBAL
        # ==========================================
        for comp in comps:
            ctype = CATALOG.get_comp_type(comp)
            val_cat = CATALOG.snap_to_catalog(comp.value, ctype)
            comp.value = val_cat 
            part_info = CATALOG.get_part_info(val_cat, ctype)
            part_num = part_info['PartNumber']
            
            if part_num not in inventory:
                inventory[part_num] = {
                    'Qty': 0, 'Description': part_info['Description'],
                    'Value': part_info['Value'], 'Type': ctype,
                    'Price': part_info['Price'], 'URL': part_info['URL']
                }
            inventory[part_num]['Qty'] += 1

        print("\n" + "="*60)
        print("🛒 CROSSOVER BILL OF MATERIALS (BOM)")
        print("="*60)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Part Number', 'Quantity', 'Value', 'Unit', 'Component', 'Description', 'Unit Price ($)', 'Line Total ($)', 'URL'])
            
            for part_num, data in inventory.items():
                qty = data['Qty']
                unit_price = data['Price'] if pd.notna(data['Price']) else 0.0
                line_total = qty * unit_price
                total_price += line_total
                
                unit_str = "μF" if data['Type'] == 'C' else "mH" if data['Type'] == 'L' else "Ω"
                comp_type_full = "Capacitor" if data['Type'] == 'C' else "Inductor" if data['Type'] == 'L' else "Resistor"
                
                writer.writerow([part_num, qty, data['Value'], unit_str, comp_type_full, data['Description'], round(unit_price, 2), round(line_total, 2), data['URL']])
                print(f"[{qty}x] {data['Value']}{unit_str} - {data['Description']} (Part #{part_num})")
                print(f"      Price: ${unit_price:.2f} each -> Total: ${line_total:.2f}")

            writer.writerow([])
            writer.writerow(['', '', '', '', '', 'TOTAL (1 Speaker):', f"${total_price:.2f}", '', ''])
            writer.writerow(['', '', '', '', '', 'TOTAL (Pair):', f"${total_price * 2:.2f}", '', ''])
            
        print("="*60)
        print(f"💰 TOTAL ESTIMATED COST: ${total_price:.2f} / Speaker")
        print(f"📄 Nomenclature (CSV) sauvegardée dans : {filename}")

        # ==========================================
        # 2. GÉNÉRATION DU FICHIER LATEX (OVERLEAF)
        # ==========================================
        # On remplace l'extension .csv par .tex
        latex_filename = filename.replace('.csv', '.tex') if filename.endswith('.csv') else filename + ".tex"
        
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[H]\n")
            f.write("    \\centering\n")
            f.write("    \\renewcommand{\\arraystretch}{1.5}\n")
            f.write("    \\begin{tabular}{@{}lllrl@{}}\n")
            f.write("        \\toprule\n")
            f.write("        \\textbf{ID} & \\textbf{Component} & \\textbf{Value} & \\textbf{Price (\\$/€)} & \\textbf{Buy Link} \\\\\n")
            f.write("        \\midrule\n")
            
            # On liste individuellement chaque composant pour créer les IDs uniques
            counts = {'C': 0, 'L': 0, 'R': 0}
            for comp in comps:
                ctype = CATALOG.get_comp_type(comp)
                counts[ctype] += 1
                comp_id = f"{ctype}{counts[ctype]}" # Produit C1, C2, L1, etc.
                
                val_cat = CATALOG.snap_to_catalog(comp.value, ctype)
                part_info = CATALOG.get_part_info(val_cat, ctype)
                
                # Formatage spécifique à la syntaxe LaTeX
                unit_str = "$\\mu$F" if ctype == 'C' else "mH" if ctype == 'L' else "$\\Omega$"
                comp_type_full = "Capacitor" if ctype == 'C' else "Inductor" if ctype == 'L' else "Resistor"
                price = part_info['Price'] if pd.notna(part_info['Price']) else 0.0
                
                # Protection des caractères spéciaux dans l'URL pour la compilation LaTeX
                url = str(part_info['URL']).replace('%', '\\%').replace('#', '\\#')
                
                f.write(f"        {comp_id} & {comp_type_full} & {part_info['Value']} {unit_str} & {price:.2f} & \\href{{{url}}}{{Link}} \\\\\n")
                
            f.write("        \\bottomrule\n")
            f.write("    \\end{tabular}\n")
            f.write("\\end{table}\n")
            
        print(f"📝 Tableau LaTeX sauvegardé dans : {latex_filename}\n")

if __name__ == "__main__":
    start_time = time.time()
    config = [
        # NOUVEAU : On peut maintenant passer count=2 à l'initialisation
        WayConfig("Woofer", r"crossovers\ER18RNX+27TDFC\SEAS_H1456-08_ER18RNX_SPL.frd", r"crossovers\ER18RNX+27TDFC\SEAS_H1456-08_ER18RNX_ZR.zma",
                  z_offset=0.00, y_offset=-100, x_offset=0.00, count=2),
        WayConfig("Tweeter", r"crossovers\ER18RNX+27TDFC\Tweeter_SPL.frd", r"crossovers\ER18RNX+27TDFC\Tweeter_ZR.zma")
    ]

    opt = CrossoverOptimizer(config)
    best = opt.run(generations=100, pop_size=120)
    best['tree'].display()
    end_time = time.time()
    print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")