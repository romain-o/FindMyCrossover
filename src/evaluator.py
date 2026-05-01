from src.nodes import *
import numpy as np

class CircuitEvaluator:
    def __init__(self, freqs):
        self.freqs = np.array(freqs)
        self.w = 2 * np.pi * self.freqs
        self.jw = 1j * self.w
        self.inv_jw = 1 / (self.jw + 1e-20)
        self.ones = np.ones_like(self.freqs)
        self._cache = {}

    def get_impedance(self, node, clear_cache=False):
        if clear_cache:
            self._cache = {}
            
        if node.id in self._cache:
            return self._cache[node.id]

        if isinstance(node, Resistor):
            res = node.value * self.ones
        elif isinstance(node, Capacitor):
            esr = 0.01
            res = (self.inv_jw / node.value) + esr
        elif isinstance(node, Inductor):
            dcr = node.value * 400 
            res = (self.jw * node.value) + dcr
        elif isinstance(node, DriverNode):
            res = node.Z_complex
        elif isinstance(node, ShuntNode):
            res = self.get_impedance(node.component)
        elif isinstance(node, SeriesNode):
            res = self.get_impedance(node.left) + self.get_impedance(node.right)
        elif isinstance(node, ParallelNode):
            z_left = self.get_impedance(node.left)
            z_right = self.get_impedance(node.right)
            res = (z_left * z_right) / (z_left + z_right + 1e-15)
        else:
            raise ValueError(f"Type de nœud non supporté: {type(node)}")
        
        self._cache[node.id] = res
        return res

    def evaluate(self, node, v_in=None):
        """
        Parcourt l'arbre de manière récursive (top-down) avec la règle du diviseur 
        de tension pour trouver la tension et la réponse de chaque DriverNode.
        """
        # On vide le cache au début de chaque évaluation complète
        if v_in is None:
            self._cache = {}
            v_in = np.ones(len(self.freqs), dtype=complex)

        responses = {}

        if isinstance(node, DriverNode):
            p_acoustic = v_in * node.H_acoustic
            if node.polarity_inverted:
                p_acoustic = -p_acoustic
            responses[node.label] = {
                "V_complex": v_in,
                "P_acoustic": p_acoustic
            }
        elif isinstance(node, SeriesNode):
            z_left = self.get_impedance(node.left)
            z_right = self.get_impedance(node.right)
            z_total = z_left + z_right + 1e-15

            # Diviseur de tension classique Z_i / Z_total
            v_left = v_in * (z_left / z_total)
            v_right = v_in * (z_right / z_total)

            responses.update(self.evaluate(node.left, v_left))
            responses.update(self.evaluate(node.right, v_right))
        elif isinstance(node, ParallelNode):
            # En parallèle, la tension est la même sur les deux branches
            responses.update(self.evaluate(node.left, v_in))
            responses.update(self.evaluate(node.right, v_in))
        elif isinstance(node, ShuntNode):
            responses.update(self.evaluate(node.component, v_in))

        return responses