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
            
            def calc(inv):
                tweeter.polarity_inverted = inv; res = self.evaluator.evaluate(root)
                p_tot = sum(d["P_acoustic"] for d in res.values()); spl = 20 * np.log10(np.abs(p_tot) + 1e-10)
                spl_u = spl[mask]; avg = np.mean(spl_u)
                
                # FITNESS RECORD (4.74)
                flat = np.mean((spl_u - avg)**4) * 30.0
                smooth = np.mean(np.diff(spl_u)**2) * 50.0
                gain_pen = (84.0 - avg)**2 * 10.0 if avg < 84.0 else 0
                
                p_w = res.get("Woofer", {}).get("P_acoustic", np.zeros_like(p_tot))[mask]
                p_t = res.get("Tweeter", {}).get("P_acoustic", np.zeros_like(p_tot))[mask]
                spl_max = np.maximum(20*np.log10(np.abs(p_w)+1e-10), 20*np.log10(np.abs(p_t)+1e-10))
                canc = np.sum(np.maximum(0, spl_max - spl_u)**2) * 20.0
                return flat + smooth + gain_pen + canc
                
            s_n = calc(False); s_i = calc(True); b_s = min(s_n, s_i); tweeter.polarity_inverted = (s_i < s_n)
            Z_in = self.evaluator.get_impedance(root); min_Z = np.min(np.abs(Z_in))
            z_pen = ((2.0 - min_Z)**2) * 100000.0 if min_Z < 2.0 else 0
            comps = [n for n in root.get_all_nodes() if isinstance(n, (Resistor, Capacitor, Inductor))]
            return b_s + z_pen + len(comps) * 0.2
        except: return 1e10

    def check_terminal_drivers(self, node):
        if isinstance(node, SeriesNode):
            if len(self.mutator._get_driver_labels(node.left)) > 0: return False
            return self.check_terminal_drivers(node.left) and self.check_terminal_drivers(node.right)
        elif isinstance(node, ParallelNode): return self.check_terminal_drivers(node.left) and self.check_terminal_drivers(node.right)
        elif isinstance(node, ShuntNode): return self.check_terminal_drivers(node.component)
        return True

    def optimize_values(self, root, max_iter=35): # Maturation plus longue pour caler les Notch
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

    def run(self, generations=40, pop_size=50):
        population = []
        if os.path.exists("best_crossover.json"):
            try:
                with open("best_crossover.json", "r") as f: cp = Node.from_dict(json.load(f))
                for n in cp.get_all_nodes():
                    if isinstance(n, DriverNode):
                        d = self.woofer if n.label == "Woofer" else self.tweeter
                        n.H_acoustic, n.Z_complex = d.H_acoustic, d.Z_complex
                population.append(self.mutator.simplify(cp)); print("[+] Champion chargé et simplifié.")
            except: pass
        
        while len(population) < pop_size:
            bw = self.mutator.generate_random_tree(self.woofer.copy(), max_depth=1)
            bt = self.mutator.generate_random_tree(self.tweeter.copy(), max_depth=1)
            population.append(ParallelNode(bw, bt))

        best_score = float('inf'); best_tree = population[0]
        print("Optimisation (Mode Lamarckien Profond - Focus Notch)...")

        for gen in range(generations):
            # Maturation de toute la population
            for ind in population: self.optimize_values(ind, max_iter=20)
            scores = sorted([(self.fitness(ind), ind) for ind in population], key=lambda x: x[0])
            
            if scores[0][0] < best_score:
                best_score = scores[0][0]; best_tree = scores[0][1].copy()
                self.optimize_values(best_tree, max_iter=120)
                best_score = self.fitness(best_tree)
                print(f"Gen {gen}: Record ! {best_score:.2f}")
                with open("best_crossover.json", "w") as f: json.dump(best_tree.to_dict(), f, indent=4)

            new_pop = [best_tree.copy()]
            while len(new_pop) < pop_size:
                p = random.sample(scores[:12], 1)[0][1]; child = self.mutator.mutate(p)
                # Mutation Lamarckienne immédiate (Secret du record)
                if self.check_terminal_drivers(child):
                    self.optimize_values(child, max_iter=30)
                    new_pop.append(child)
            population = new_pop
            if gen % 5 == 0: print(f"Gen {gen}/{generations} - Best: {best_score:.2f}")
        return best_tree

    def plot_result(self, root, filename="crossover_response.png"):
        res = self.evaluator.evaluate(root); plt.figure(figsize=(10, 6))
        p_tot = np.zeros(len(self.freqs), dtype=complex)
        for label, data in res.items():
            spl = 20 * np.log10(np.abs(data["P_acoustic"]) + 1e-10)
            plt.semilogx(self.freqs, spl, label=label, alpha=0.6); p_tot += data["P_acoustic"]
        spl_t = 20 * np.log10(np.abs(p_tot) + 1e-10); avg = np.mean(spl_t[(self.freqs >= 100) & (self.freqs <= 18000)])
        plt.semilogx(self.freqs, spl_t, label="Total", color='black', linewidth=2)
        plt.axhline(avg, color='green', linestyle='--', label="Moyenne"); plt.ylim(avg-25, avg+10)
        plt.grid(True, which="both", ls="-", alpha=0.5); plt.legend(); plt.savefig(filename); plt.close()

if __name__ == "__main__":
    w_f = ("Driver_Data/RS225-8@0.frd", "Driver_Data/RS225-8.zma")
    t_f = ("Driver_Data/SEAS_27TDFC_tweeter_SPL.frd", "Driver_Data/SEAS_27TDFC_tweeter_ZR.zma")
    opt = CrossoverOptimizer(w_f, t_f); best = opt.run(generations=30, pop_size=40)
    best.display(); best.save_as_html("circuit_viz.html"); opt.plot_result(best)
