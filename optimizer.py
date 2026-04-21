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

BOUNDS_R = (0.1, 50.0)
BOUNDS_C = (0.1e-6, 150e-6)
BOUNDS_L = (0.05e-3, 15e-3)

E24_SERIES = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 
                       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])

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
    def __init__(self, ways_configs, target_spl=83.0, fx_bounds=[(2000, 2000)]):
        self.freqs = np.geomspace(20, 20000, 400)
        self.target_spl = target_spl
        self.ways = ways_configs
        self.fx_bounds = fx_bounds
        self.evaluator = CircuitEvaluator(self.freqs)
        self.mutator = TreeMutator()
        
        # Préparation des drivers (Interpolation + Z-Offset)
        for way in self.ways:
            self._prepare_driver(way)

    def _prepare_driver(self, way):
        d = way.driver
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

    def _generate_targets(self, fx_list, spl):
        targets = {}
        pressure = 10 ** (spl / 20)
        
        for i, way in enumerate(self.ways):
            if way.target_type == 'LP':
                # Utilise fx_list[i] au lieu de fx_list[0]
                h = self._get_lr4_transfer(fx_list[i], 'LP') 
            elif way.target_type == 'HP':
                # Utilise fx_list[i]
                h = self._get_lr4_transfer(fx_list[i], 'HP')
            elif way.target_type == 'BP':
                h_low = self._get_lr4_transfer(fx_list[i-1], 'HP')
                h_high = self._get_lr4_transfer(fx_list[i], 'LP')
                h = h_low * h_high
            targets[way.label] = h * pressure
        return targets

    def fitness(self, individual):
        root = individual['tree']
        fx_list = individual['fx']
        
        try:
            if not isinstance(root, ParallelNode): return 1e9
            
            res = self.evaluator.evaluate(root)
            
            # SPL Adaptatif sur le woofer (première voie)
            p_ref = res.get(self.ways[0].label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            mask_ref = (self.freqs > 80) & (self.freqs < 400)
            avg_spl = 20 * np.log10(np.mean(np.abs(p_ref[mask_ref])) + 1e-12) if np.any(mask_ref) else self.target_spl
            dynamic_spl = np.clip(avg_spl, 75.0, 95.0)
            
            targets = self._generate_targets(fx_list, dynamic_spl)
            total_mse = 0.0

            # --- 1. MSE INDIVIDUELLE DES VOIES (Indépendant de la polarité) ---
            for way in self.ways:
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                spl_real = 20 * np.log10(np.abs(p_real) + 1e-12)
                spl_target = 20 * np.log10(np.abs(targets[way.label]) + 1e-12)
                
                # Pondération (Focus sur la zone de transition)
                weight = np.ones_like(self.freqs)
                for fx in fx_list:
                    weight[(self.freqs > fx/4) & (self.freqs < fx*4)] = 10.0
                
                total_mse += np.mean(((spl_real - spl_target)**2) * weight)

            # --- 2. ÉVALUATION DYNAMIQUE DES POLARITÉS ET DE LA PHASE ---
            best_score_sum = float('inf')
            best_polarities = [1.0] * len(self.ways)
            
            # On génère toutes les combinaisons possibles (+1 ou -1)
            pol_combinations = list(itertools.product([1.0, -1.0], repeat=len(self.ways)-1))
            
            sum_weight = np.zeros_like(self.freqs)
            for fx in fx_list:
                sum_weight[(self.freqs > fx/2) & (self.freqs < fx*2)] = 1.0

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
                
                # Pénalité de magnitude (Bosses 5x plus pénalisées)
                mse_sum = np.where(diff > 0, (diff**2) * 5.0, diff**2)
                score_sum = np.mean(mse_sum * sum_weight) * 3.0
                
                # --- NOUVEAU : PÉNALITÉ DE PHASE (PHASE TRACKING) ---
                phase_penalty = 0.0
                for j in range(len(self.ways) - 1):
                    p1 = p_ways[j]
                    p2 = p_ways[j+1]
                    
                    # Écart de phase en radians (ramené entre 0 et pi)
                    phase_diff_rad = np.abs(np.angle(p1 / (p2 + 1e-12)))
                    
                    # On pondère par le produit des magnitudes : la phase ne compte 
                    # QUE là où les deux HP jouent ensemble !
                    overlap_weight = np.abs(p1) * np.abs(p2)
                    overlap_weight /= (np.max(overlap_weight) + 1e-12) # Normalisation de 0 à 1
                    
                    # On tolère jusqu'à ~30 degrés d'écart (0.52 radians). Au-delà, pénalité !
                    excess_phase = np.maximum(0, phase_diff_rad - 0.52)
                    
                    # On ajoute au score global de cette combinaison de polarité
                    phase_penalty += np.mean(excess_phase * overlap_weight) * 100
                
                score_sum += phase_penalty
                # -----------------------------------------------------
                
                if score_sum < best_score_sum:
                    best_score_sum = score_sum
                    best_polarities = current_pols
            
            total_mse += best_score_sum
            
            # On stocke la meilleure polarité trouvée dans l'individu
            individual['best_polarities'] = best_polarities

            # --- 3. PÉNALITÉS PHYSIQUES ---
            penalty = 0.0
            Z_in = self.evaluator.get_impedance(root)
            min_Z = np.min(np.abs(Z_in))
            if min_Z < 3.2: penalty += 5000.0 * (3.2 - min_Z)**2
            
            last_way_v = res.get(self.ways[-1].label, {}).get("V_complex", np.zeros_like(self.freqs))
            v_low = np.abs(last_way_v)[self.freqs < (fx_list[-1] * 0.5)]
            penalty += np.sum(np.maximum(0, v_low - 0.1)**2) * 5000.0

            n_comps = len([n for n in root.get_all_nodes() if isinstance(n, ComponentNode)])
            comp_penalty = max(0, n_comps - 10) * 2.0

            return total_mse + penalty + comp_penalty
            
        except Exception: return 1e10

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
            
        res = minimize(obj, init, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        for i, v in enumerate(res.x): comps[i].value = 10**v
        return individual

    def run(self, generations=50, pop_size=60):
        population = []
        
        # Tentative de chargement du champion
        if os.path.exists("best_crossover.json"):
            try:
                with open("best_crossover.json", "r") as f: data = json.load(f)
                tree = Node.from_dict(data["tree"])
                # Réassignation des drivers
                for n in tree.get_all_nodes():
                    if isinstance(n, DriverNode):
                        way = next(w for w in self.ways if w.label == n.label)
                        n.H_acoustic, n.Z_complex = way.driver.H_acoustic, way.driver.Z_complex
                
                # Chargement de la nouvelle structure
                best_pols = data.get("best_polarities", [1.0] * len(self.ways))
                fxs = data.get("fx", [b[0] for b in self.fx_bounds])
                population.append({'tree': tree, 'fx': fxs, 'best_polarities': best_pols})
                print("[+] Champion chargé.")
            except Exception as e: print(f"Erreur chargement: {e}")
        
        else:
            try:
                # Branche Woofer : Série(Inductance), Parallèle(Driver, Condensateur)
                w_branch = SeriesNode(Inductor(1.5e-3), ParallelNode(self.ways[0].driver.copy(), Capacitor(10e-6)))
                # Branche Tweeter : Série(Condensateur), Parallèle(Driver, Inductance)
                t_branch = SeriesNode(Capacitor(4.7e-6), ParallelNode(self.ways[1].driver.copy(), Inductor(0.3e-3)))
                
                seed_tree = ParallelNode(w_branch, t_branch)
                
                # On place les Fx au milieu des plages autorisées
                fx_mid = [(b[0] + b[1]) / 2.0 for b in self.fx_bounds]
                
                population.append({'tree': seed_tree, 'fx': fx_mid})
                print("[+] Graine (Template 2ème ordre) injectée dans la population initiale.")
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
            
            fx_genes = [random.uniform(b[0], b[1]) for b in self.fx_bounds]
            population.append({'tree': root, 'fx': fx_genes})

        best_score = float('inf')
        best_ind = population[0]

        for gen in range(generations):
            scores = []
            for ind in population:
                scores.append((self.fitness(ind), ind))
            
            scores.sort(key=lambda x: x[0])
            
            # --- GESTION DES 3 PHASES D'ÉVOLUTION ---
            if gen < int(generations * 0.4):      # Phase 1: Exploration
                max_opt_iter = 2
                snap_to_standard = False
            elif gen < int(generations * 0.8):    # Phase 2: Consolidation
                max_opt_iter = 12
                snap_to_standard = False
            else:                                 # Phase 3: Polissage Réaliste
                max_opt_iter = 25
                snap_to_standard = True

            # Optimisation locale du Top 10%
            elite_count = max(2, pop_size // 10)
            for i in range(elite_count):
                # 1. Optimisation mathématique continue
                self.optimize_values(scores[i][1], max_iter=max_opt_iter)
                
                # 2. Aimantation sur les composants réels (Phase 3 uniquement)
                if snap_to_standard:
                    for comp in scores[i][1]['tree'].get_all_nodes():
                        if isinstance(comp, ComponentNode):
                            comp.value = snap_to_e24(comp.value)
                            
                # 3. Réévaluation du score avec les nouvelles valeurs
                scores[i] = (self.fitness(scores[i][1]), scores[i][1])
            
            scores.sort(key=lambda x: x[0])
            
            if scores[0][0] < best_score:
                best_score = scores[0][0]
                best_ind = scores[0][1]
                
                n_comps = len([n for n in best_ind['tree'].get_all_nodes() if isinstance(n, ComponentNode)])
                fx_str = " - ".join([f"{way.label}: {int(f)}Hz" for way, f in zip(self.ways, best_ind['fx'])])
                
                print(f"Gen {gen}: Record {best_score:.2f} | {fx_str} | Composants: {n_comps}")
                
                # Sauvegarde avec la nouvelle clé
                with open("best_crossover.json", "w") as f:
                    json.dump({
                        "tree": best_ind['tree'].to_dict(), 
                        "fx": best_ind['fx'],
                        "best_polarities": best_ind.get('best_polarities', [1.0] * len(self.ways))
                    }, f, indent=4)

            new_pop = [best_ind]
            for i in range(1, elite_count):
                new_pop.append(scores[i][1])
                
            while len(new_pop) < pop_size:
                parent = random.choice(scores[:pop_size//3])[1]
                child_tree = self.mutator.mutate(parent['tree'].copy())
                child_fx = [np.clip(f * random.uniform(0.95, 1.05), b[0], b[1]) 
                            for f, b in zip(parent['fx'], self.fx_bounds)]
                # L'individu généré ne contient plus de clé 'polarities'
                new_pop.append({'tree': child_tree, 'fx': child_fx})
            
            population = new_pop
            if gen % 5 == 0:
                print(f"--- Gen {gen}/{generations} - Best: {best_score:.2f} ---")

        # Polissage final
        self.optimize_values(best_ind, max_iter=150)
        self.plot_result(best_ind)
        return best_ind

    def plot_result(self, individual, filename="crossover_response.png"):
        root = individual['tree']
        fx_list = individual['fx']
        res = self.evaluator.evaluate(root)
        
        # Recalcul SPL pour le plot
        p_ref = res.get(self.ways[0].label, {}).get("P_acoustic", np.zeros_like(self.freqs))
        mask_ref = (self.freqs > 80) & (self.freqs < 400)
        dynamic_spl = 20 * np.log10(np.mean(np.abs(p_ref[mask_ref])) + 1e-12) if np.any(mask_ref) else self.target_spl
        
        targets = self._generate_targets(fx_list, dynamic_spl)
        
        plt.figure(figsize=(12, 8))
        p_sum = np.zeros_like(self.freqs, dtype=complex)
        best_pols = individual.get('best_polarities', [1.0] * len(self.ways))
        
        for i, way in enumerate(self.ways):
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            
            # Application de la polarité pour la somme totale
            p_sum += p_real * best_pols[i] 
            
            spl_real = 20 * np.log10(np.abs(p_real) + 1e-10)
            
            # Si le haut-parleur est inversé, on ajoute un petit (inv) dans la légende
            label_suffix = " (inv)" if best_pols[i] < 0 else ""
            plt.semilogx(self.freqs, spl_real, label=f"Réel {way.label}{label_suffix}", linewidth=2)
            
            spl_target = 20 * np.log10(np.abs(targets[way.label]) + 1e-10)
            plt.semilogx(self.freqs, spl_target, 'k:', alpha=0.3)

        plt.semilogx(self.freqs, 20 * np.log10(np.abs(p_sum) + 1e-10), label="Somme", color='red', linewidth=3)
        
        for fx in fx_list:
            plt.axvline(fx, color='grey', linestyle='--', alpha=0.5)
            
        plt.axhline(dynamic_spl, color='green', linestyle='--', alpha=0.5)
        plt.ylim(dynamic_spl - 40, dynamic_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.title(f"Réponse {len(self.ways)}-voies (Score: {self.fitness(individual):.2f})")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    # CONFIGURATION 2-VOIES PERSONNALISÉE
    config = [
        WayConfig("Woofer", "Driver_Data/RS225-8@0.frd", "Driver_Data/RS225-8.zma", target_type='LP', z_offset_m=0.03),
        WayConfig("Tweeter", "Driver_Data/SEAS_27TDFC_tweeter_SPL.frd", "Driver_Data/SEAS_27TDFC_tweeter_ZR.zma", target_type='HP', z_offset_m=0.0)
    ]
    
    # Plage de recherche Fx entre 1500Hz et 3000Hz
    opt = CrossoverOptimizer(config, fx_bounds=[(1500, 2200), (1500, 2500)])
    best = opt.run(generations=100, pop_size=120)
    best['tree'].display()
