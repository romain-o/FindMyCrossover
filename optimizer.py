import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from nodes import DriverNode, SeriesNode, ParallelNode, ShuntNode, Capacitor, Inductor, Resistor, Node, ComponentNode
from evaluator import CircuitEvaluator
from mutator import TreeMutator
from scipy.optimize import minimize

BOUNDS_R = (0.1, 50.0)
BOUNDS_C = (0.1e-6, 150e-6)
BOUNDS_L = (0.05e-3, 15e-3)

class CrossoverOptimizer:
    def __init__(self, low_paths, high_paths, target_spl=87.0, fx=2000.0):
        self.freqs = np.geomspace(20, 20000, 400)
        self.target_spl = target_spl
        self.fx = fx # Fréquence de croisement cible
        self.evaluator = CircuitEvaluator(self.freqs)
        self.mutator = TreeMutator()
        
        self.woofer = DriverNode("Woofer", low_paths[0], low_paths[1])
        self.tweeter = DriverNode("Tweeter", high_paths[0], high_paths[1])
        self._interpolate_driver(self.woofer)
        self._interpolate_driver(self.tweeter)
        
        # Génération des courbes cibles (Target Curves) Linkwitz-Riley 4ème ordre
        self._generate_target_curves()

    def _generate_target_curves(self, current_spl=None):
        """Génère les courbes cibles complexes idéales. SPL adaptatif si current_spl est fourni."""
        spl = current_spl if current_spl is not None else self.target_spl
        s = 1j * (self.freqs / self.fx)
        # Fonction de transfert LR4
        lr4_lp = 1 / ((s**2 + np.sqrt(2)*s + 1)**2)
        lr4_hp = (s**4) / ((s**2 + np.sqrt(2)*s + 1)**2)
        
        target_pressure = 10 ** (spl / 20)
        self.target_woofer = lr4_lp * target_pressure
        self.target_tweeter = lr4_hp * target_pressure
        return spl

    def _interpolate_driver(self, d):
        """Interpolation corrigée avec déroulement de la phase."""
        # SPL (.frd)
        mag_db = 20 * np.log10(np.abs(d.H_acoustic) + 1e-10)
        ph_unwrapped = np.unwrap(np.angle(d.H_acoustic)) # CORRECTION CRUCIALE
        
        mag_interp = np.interp(self.freqs, d.frd_freqs, mag_db)
        ph_interp = np.interp(self.freqs, d.frd_freqs, ph_unwrapped)
        d.H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
        
        # Impédance (.zma)
        z_mag = np.abs(d.Z_complex)
        z_ph_unwrapped = np.unwrap(np.angle(d.Z_complex))
        
        z_mag_interp = np.interp(self.freqs, d.zma_freqs, z_mag)
        z_ph_interp = np.interp(self.freqs, d.zma_freqs, z_ph_unwrapped)
        d.Z_complex = z_mag_interp * np.exp(1j * z_ph_interp)

    def fitness(self, root):
        try:
            if not isinstance(root, (ParallelNode, SeriesNode)): return 1e9
            if not self.check_terminal_drivers(root): return 1e8
            
            nodes = root.get_all_nodes()
            drivers = [n for n in nodes if isinstance(n, DriverNode)]
            if len(drivers) < 2: return 1e8
            
            tweeter = next((n for n in drivers if n.label == "Tweeter"), None)
            woofer = next((n for n in drivers if n.label == "Woofer"), None)
            
            # --- CALCUL DU SPL ADAPTATIF ---
            res_temp = self.evaluator.evaluate(root)
            p_w_raw = res_temp.get("Woofer", {}).get("P_acoustic", np.zeros_like(self.freqs))
            mask_ref = (self.freqs > 100) & (self.freqs < 800)
            if np.any(mask_ref):
                avg_spl = 20 * np.log10(np.mean(np.abs(p_w_raw[mask_ref])) + 1e-12)
                dynamic_spl = np.clip(avg_spl, 75.0, 95.0)
            else:
                dynamic_spl = self.target_spl

            self._generate_target_curves(dynamic_spl)

            def calc_mse(inv_polarity):
                tweeter.polarity_inverted = inv_polarity
                res = self.evaluator.evaluate(root)
                
                p_w = res.get("Woofer", {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_t = res.get("Tweeter", {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_sum = p_w + p_t
                
                spl_w = 20 * np.log10(np.abs(p_w) + 1e-12)
                spl_t = 20 * np.log10(np.abs(p_t) + 1e-12)
                spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
                
                target_spl_w = 20 * np.log10(np.abs(self.target_woofer) + 1e-12)
                target_spl_t = 20 * np.log10(np.abs(self.target_tweeter) + 1e-12)
                
                floor_spl = dynamic_spl - 40.0
                mse_w = np.mean((np.maximum(spl_w, floor_spl) - np.maximum(target_spl_w, floor_spl))**2)
                mse_t = np.mean((np.maximum(spl_t, floor_spl) - np.maximum(target_spl_t, floor_spl))**2)
                
                mask_sum = (self.freqs > self.fx/2) & (self.freqs < self.fx*2)
                mse_sum = np.mean((spl_sum[mask_sum] - dynamic_spl)**2)
                
                return mse_w + mse_t + (mse_sum * 1.5), res, dynamic_spl

            mse_n, res_n, spl_n = calc_mse(False)
            mse_i, res_i, spl_i = calc_mse(True)
            
            best_mse = min(mse_n, mse_i)
            tweeter.polarity_inverted = (mse_i < mse_n)
            best_res = res_i if mse_i < mse_n else res_n
            final_dynamic_spl = spl_i if mse_i < mse_n else spl_n

            penalty = 0.0
            Z_in = self.evaluator.get_impedance(root)
            min_Z = np.min(np.abs(Z_in))
            if min_Z < 3.0: penalty += 2000.0 * (3.0 - min_Z)**2
                
            tw_v = best_res.get("Tweeter", {}).get("V_complex", np.zeros_like(self.freqs))
            v_low = np.abs(tw_v)[self.freqs < (self.fx * 0.5)]
            penalty += np.sum(np.maximum(0, v_low - 0.15)**2) * 3000.0

            n_comps = len([n for n in nodes if isinstance(n, (Resistor, Capacitor, Inductor))])
            comp_penalty = max(0, n_comps - 6) * 3.0 

            if final_dynamic_spl < 80.0:
                penalty += (80.0 - final_dynamic_spl) * 50.0

            return best_mse + penalty + comp_penalty
            
        except Exception:
            return 1e10

    def check_terminal_drivers(self, node):
        """Vérifie que les drivers ne sont pas utilisés comme composants de passage."""
        if isinstance(node, SeriesNode):
            if len(self.mutator._get_driver_labels(node.left)) > 0: return False
            return self.check_terminal_drivers(node.left) and self.check_terminal_drivers(node.right)
        elif isinstance(node, ParallelNode): 
            return self.check_terminal_drivers(node.left) and self.check_terminal_drivers(node.right)
        elif isinstance(node, ShuntNode): 
            return self.check_terminal_drivers(node.component)
        return True

    def optimize_values(self, root, max_iter=5):
        """Recherche locale. max_iter très réduit par défaut pour ne pas ralentir le GP."""
        comps = [n for n in root.get_all_nodes() if isinstance(n, (Resistor, Capacitor, Inductor))]
        if not comps: return root
        
        init = [np.log10(np.clip(c.value, 1e-12, 1e2)) for c in comps]
        bounds = [(np.log10(BOUNDS_R[0]), np.log10(BOUNDS_R[1])) if isinstance(c, Resistor) else 
                  (np.log10(BOUNDS_C[0]), np.log10(BOUNDS_C[1])) if isinstance(c, Capacitor) else 
                  (np.log10(BOUNDS_L[0]), np.log10(BOUNDS_L[1])) for c in comps]
                  
        def obj(x_log):
            for i, v in enumerate(x_log): comps[i].value = 10**v
            return self.fitness(root)
            
        res = minimize(obj, init, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-4})
        for i, v in enumerate(res.x): comps[i].value = 10**v
        return root

    def run(self, generations=50, pop_size=60):
        population = []
        if os.path.exists("best_crossover.json"):
            try:
                with open("best_crossover.json", "r") as f: cp = Node.from_dict(json.load(f))
                for n in cp.get_all_nodes():
                    if isinstance(n, DriverNode):
                        d = self.woofer if n.label == "Woofer" else self.tweeter
                        n.H_acoustic, n.Z_complex = d.H_acoustic, d.Z_complex
                population.append(self.mutator.simplify(cp))
                print("[+] Champion chargé.")
            except: pass
        
        while len(population) < pop_size:
            bw = self.mutator.generate_random_tree(self.woofer.copy(), max_depth=2)
            bt = self.mutator.generate_random_tree(self.tweeter.copy(), max_depth=2)
            population.append(ParallelNode(bw, bt))

        best_score = float('inf')
        best_tree = population[0]
        
        print(f"Optimisation Cible LR4 à {self.fx}Hz...")

        for gen in range(generations):
            # Évaluation initiale de la population
            scores = [(self.fitness(ind), ind) for ind in population]
            scores.sort(key=lambda x: x[0])
            
            # On n'applique la recherche locale (L-BFGS-B) que sur le Top 10% !
            # Cela fait gagner un temps CPU massif.
            elite_count = max(2, pop_size // 10)
            for i in range(elite_count):
                self.optimize_values(scores[i][1], max_iter=5)
            
            # Réévaluation du Top 10% après optimisation
            for i in range(elite_count):
                scores[i] = (self.fitness(scores[i][1]), scores[i][1])
            scores.sort(key=lambda x: x[0])
            
            if scores[0][0] < best_score:
                best_score = scores[0][0]
                best_tree = scores[0][1].copy()
                print(f"Gen {gen}: Record ! Score: {best_score:.4f} | Composants: {len([n for n in best_tree.get_all_nodes() if isinstance(n, ComponentNode)])}")
                with open("best_crossover.json", "w") as f: json.dump(best_tree.to_dict(), f, indent=4)

            new_pop = [best_tree.copy()]
            
            # Élite inchangée passe à la génération suivante (Élitisme)
            for i in range(1, elite_count):
                new_pop.append(scores[i][1].copy())
                
            # Remplissage par mutation
            while len(new_pop) < pop_size:
                # Sélection par tournoi ou biaisée vers l'élite
                parent = random.choice(scores[:pop_size//2])[1]
                child = self.mutator.mutate(parent)
                if self.check_terminal_drivers(child):
                    new_pop.append(child)
                    
            population = new_pop
            if gen % 5 == 0: 
                print(f"--- Gen {gen}/{generations} - Best: {best_score:.4f} ---")
                
        # Polissage final intense sur le grand gagnant
        print("Lancement de l'optimisation locale finale profonde...")
        self.optimize_values(best_tree, max_iter=150)
        final_score = self.fitness(best_tree)
        print(f"Score final après polissage : {final_score:.4f}")
        
        return best_tree

    def plot_result(self, root, filename="crossover_response.png"):
        res = self.evaluator.evaluate(root)
        
        # Recalcul du SPL dynamique pour le plot
        p_w_raw = res.get("Woofer", {}).get("P_acoustic", np.zeros_like(self.freqs))
        mask_ref = (self.freqs > 100) & (self.freqs < 800)
        dynamic_spl = 20 * np.log10(np.mean(np.abs(p_w_raw[mask_ref])) + 1e-12) if np.any(mask_ref) else self.target_spl
        dynamic_spl = np.clip(dynamic_spl, 75.0, 95.0)
        
        # Régénération des cibles pour l'affichage
        self._generate_target_curves(dynamic_spl)
        
        plt.figure(figsize=(12, 8))
        
        # Courbes cibles
        spl_target_w = 20 * np.log10(np.abs(self.target_woofer) + 1e-10)
        spl_target_t = 20 * np.log10(np.abs(self.target_tweeter) + 1e-10)
        plt.semilogx(self.freqs, spl_target_w, 'k:', alpha=0.3, label=f"Target LP (LR4 @ {dynamic_spl:.1f}dB)")
        plt.semilogx(self.freqs, spl_target_t, 'k:', alpha=0.3, label="Target HP (LR4)")

        p_tot = np.zeros(len(self.freqs), dtype=complex)
        for label, data in res.items():
            spl = 20 * np.log10(np.abs(data["P_acoustic"]) + 1e-10)
            plt.semilogx(self.freqs, spl, label=f"Réel {label}", linewidth=2)
            p_tot += data["P_acoustic"]
            
        spl_t = 20 * np.log10(np.abs(p_tot) + 1e-10)
        plt.semilogx(self.freqs, spl_t, label="Somme Réelle", color='red', linewidth=3)
        
        plt.axhline(dynamic_spl, color='green', linestyle='--', alpha=0.4, label="Niveau de référence")
        plt.axvline(self.fx, color='grey', linestyle='-.', label=f"Fx = {self.fx} Hz")
        
        plt.ylim(dynamic_spl - 40, dynamic_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(loc='lower left')
        plt.title(f"Réponse Crossover (Score: {self.fitness(root):.2f} | SPL: {dynamic_spl:.1f} dB)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("SPL (dB)")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    w_f = ("Driver_Data/RS225-8@0.frd", "Driver_Data/RS225-8.zma")
    t_f = ("Driver_Data/SEAS_27TDFC_tweeter_SPL.frd", "Driver_Data/SEAS_27TDFC_tweeter_ZR.zma")
    opt = CrossoverOptimizer(w_f, t_f, fx=2000.0)
    best = opt.run(generations=60, pop_size=60)
    best.display()
    opt.plot_result(best)