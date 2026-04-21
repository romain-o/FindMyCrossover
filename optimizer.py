import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from nodes import DriverNode, SeriesNode, ParallelNode, ShuntNode, Capacitor, Inductor, Resistor, Node
from evaluator import CircuitEvaluator
from mutator import TreeMutator
from scipy.optimize import minimize

BOUNDS_R = (0.1, 50.0); BOUNDS_C = (0.1e-6, 150e-6); BOUNDS_L = (0.05e-3, 15e-3)

class CrossoverOptimizer:
    def __init__(self, low_paths, high_paths, target_spl=87.0):
        self.freqs = np.geomspace(20, 20000, 400); self.target_spl = target_spl
        self.evaluator = CircuitEvaluator(self.freqs); self.mutator = TreeMutator()
        self.woofer = DriverNode("Woofer", low_paths[0], low_paths[1])
        self.tweeter = DriverNode("Tweeter", high_paths[0], high_paths[1])
        self._interpolate_driver(self.woofer); self._interpolate_driver(self.tweeter)

    def _interpolate_driver(self, d):
        mag = 20 * np.log10(np.abs(d.H_acoustic) + 1e-10); ph = np.angle(d.H_acoustic)
        d.H_acoustic = (10 ** (np.interp(self.freqs, d.frd_freqs, mag) / 20)) * np.exp(1j * np.interp(self.freqs, d.frd_freqs, ph))
        z_mag = np.abs(d.Z_complex); z_ph = np.angle(d.Z_complex)
        d.Z_complex = np.interp(self.freqs, d.zma_freqs, z_mag) * np.exp(1j * np.interp(self.freqs, d.zma_freqs, z_ph))

    def fitness(self, root):
        try:
            if not isinstance(root, (ParallelNode, SeriesNode)): return 1e9
            tweeter = next(n for n in root.get_all_nodes() if isinstance(n, DriverNode) and n.label == "Tweeter")
            if not self.check_terminal_drivers(root): return 1e8

            mask = (self.freqs >= 100) & (self.freqs <= 18000)
            def calc_score(inv):
                tweeter.polarity_inverted = inv; res = self.evaluator.evaluate(root)
                p_tot = sum(d["P_acoustic"] for d in res.values()); spl = 20 * np.log10(np.abs(p_tot) + 1e-10)
                spl_u = spl[mask]
                
                # --- LOGIQUE CHAMPION ---
                avg = np.mean(spl_u) # Niveau flottant
                flat = np.mean((spl_u - avg)**4) * 40.0 # Flatness puissance 4
                smooth = np.mean(np.diff(spl_u)**2) * 250.0 # Smoothness (Gradient)
                
                # Pénalité de gain (seulement si on tombe trop bas par rapport à la sensibilité naturelle)
                gain_pen = (86.0 - avg)**2 * 20.0 if avg < 86.0 else 0
                
                # Anti-Cancellation
                p_w = res.get("Woofer", {}).get("P_acoustic", np.zeros_like(p_tot))[mask]
                p_t = res.get("Tweeter", {}).get("P_acoustic", np.zeros_like(p_tot))[mask]
                spl_max = np.maximum(20*np.log10(np.abs(p_w)+1e-10), 20*np.log10(np.abs(p_t)+1e-10))
                canc = np.sum(np.maximum(0, spl_max - spl_u)**2) * 30.0
                
                return flat + smooth + gain_pen + canc
            
            s_n = calc_score(False); s_i = calc_score(True)
            best_s = min(s_n, s_i); tweeter.polarity_inverted = (s_i < s_n)
            
            Z_in = self.evaluator.get_impedance(root); min_Z = np.min(np.abs(Z_in))
            z_pen = ((2.0 - min_Z)**2) * 100000.0 if min_Z < 2.0 else 0
            
            comps = [n for n in root.get_all_nodes() if isinstance(n, (Resistor, Capacitor, Inductor))]
            return best_s + z_pen + len(comps) * 0.5
        except: return 1e10

    def check_terminal_drivers(self, node):
        if isinstance(node, SeriesNode):
            if len(self.mutator._get_driver_labels(node.left)) > 0: return False
            return self.check_terminal_drivers(node.left) and self.check_terminal_drivers(node.right)
        elif isinstance(node, ParallelNode): return self.check_terminal_drivers(node.left) and self.check_terminal_drivers(node.right)
        elif isinstance(node, ShuntNode): return self.check_terminal_drivers(node.component)
        return True

    def optimize_values(self, root, max_iter=25):
        comps = [n for n in root.get_all_nodes() if isinstance(n, (Resistor, Capacitor, Inductor))]
        if not comps: return root
        init = [np.log10(np.clip(c.value, 1e-12, 1e2)) for c in comps]
        bounds = [(np.log10(BOUNDS_R[0]), np.log10(BOUNDS_R[1])) if isinstance(c, Resistor) else 
                  (np.log10(BOUNDS_C[0]), np.log10(BOUNDS_C[1])) if isinstance(c, Capacitor) else 
                  (np.log10(BOUNDS_L[0]), np.log10(BOUNDS_L[1])) for c in comps]
        def obj(x_log):
            for i, v in enumerate(x_log): comps[i].value = 10**v
            return self.fitness(root)
        res = minimize(obj, init, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-6})
        for i, v in enumerate(res.x): comps[i].value = 10**v
        return root

    def run(self, generations=100, pop_size=40):
        if os.path.exists("best_crossover.json"): os.remove("best_crossover.json")
        population = []
        # STARTER KIT DIVERSIFIÉ
        while len(population) < 5:
            l = Inductor(2.2e-3); c_s = Capacitor(15e-6) # 2ème ordre woofer
            c = Capacitor(6.8e-6); l_s = Inductor(0.47e-3); r_s = Resistor(2.2) # 3ème ordre tweeter
            w = SeriesNode(l, ParallelNode(ShuntNode(c_s), self.woofer.copy()))
            t = SeriesNode(c, ParallelNode(ShuntNode(SeriesNode(l_s, r_s)), self.tweeter.copy()))
            population.append(ParallelNode(w, t))
        while len(population) < pop_size:
            bw = self.mutator.generate_random_tree(self.woofer.copy(), max_depth=2)
            bt = self.mutator.generate_random_tree(self.tweeter.copy(), max_depth=2)
            root = ParallelNode(bw, bt)
            if self.check_terminal_drivers(root): population.append(root)

        best_score = float('inf'); best_tree = population[0]
        print("Recherche du Candidat Champion (Niveau Flottant + Smoothness)...")

        for gen in range(generations):
            for ind in population: self.optimize_values(ind, max_iter=15)
            scores = sorted([(self.fitness(ind), ind) for ind in population], key=lambda x: x[0])
            
            if scores[0][0] < best_score:
                best_score = scores[0][0]; best_tree = scores[0][1].copy()
                self.optimize_values(best_tree, max_iter=100); best_score = self.fitness(best_tree)
                print(f"Gen {gen}: Nouveau Record ! {best_score:.2f}")
                with open("best_crossover.json", "w") as f: json.dump(best_tree.to_dict(), f, indent=4)

            new_pop = [best_tree.copy()]
            while len(new_pop) < pop_size:
                p = random.sample(scores[:12], 1)[0][1]; child = self.mutator.mutate(p)
                if self.check_terminal_drivers(child): new_pop.append(child)
            population = new_pop
            if gen % 5 == 0: print(f"Gen {gen}/{generations} - Best: {best_score:.2f}")
        return best_tree

    def plot_result(self, root, filename="crossover_response.png"):
        res = self.evaluator.evaluate(root); plt.figure(figsize=(10, 6))
        p_tot = np.zeros(len(self.freqs), dtype=complex)
        for label, data in res.items():
            spl = 20 * np.log10(np.abs(data["P_acoustic"]) + 1e-10)
            plt.semilogx(self.freqs, spl, label=label, alpha=0.6); p_tot += data["P_acoustic"]
        spl_tot = 20 * np.log10(np.abs(p_tot) + 1e-10)
        avg = np.mean(spl_tot[(self.freqs >= 100) & (self.freqs <= 18000)])
        plt.semilogx(self.freqs, spl_tot, label=f"Total (Moy: {avg:.1f}dB)", color='black', linewidth=2)
        plt.axhline(avg, color='green', linestyle='--', label="Moyenne")
        plt.ylim(avg-25, avg+10); plt.grid(True, which="both", ls="-", alpha=0.5); plt.legend(); plt.savefig(filename); plt.close()

if __name__ == "__main__":
    w_f = ("Driver_Data/RS225-8@0.frd", "Driver_Data/RS225-8.zma")
    t_f = ("Driver_Data/SEAS_27TDFC_tweeter_SPL.frd", "Driver_Data/SEAS_27TDFC_tweeter_ZR.zma")
    opt = CrossoverOptimizer(w_f, t_f); best = opt.run(generations=60, pop_size=40)
    best.display(); opt.plot_result(best)
