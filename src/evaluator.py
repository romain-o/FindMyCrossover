from src.nodes import Resistor, Capacitor, Inductor, DriverNode, SeriesNode, ParallelNode, ShuntNode
import numpy as np

class CircuitEvaluator:
    def __init__(self, freqs):
        self.freqs = freqs  # Déjà un array numpy défini dans l'optimizer
        self.jw = 1j * 2 * np.pi * self.freqs
        self.inv_jw = 1.0 / (self.jw + 1e-20)

    def compile_tree(self, root):
        """
        Aplatit l'arbre en listes d'instructions séquentielles (Execution Plan).
        Élimine totalement la récursion Python pour des performances maximales.
        """
        z_plan = []
        v_plan = []
        driver_nodes = []
        
        # 1. Parcours Postfix (Bottom-Up) pour planifier le calcul d'impédance
        def build_z(node):
            t = type(node)
            if t in (SeriesNode, ParallelNode):
                build_z(node.left)
                build_z(node.right)
                z_plan.append((t, node, node.left, node.right))
            elif t is ShuntNode:
                build_z(node.component)
                z_plan.append((t, node, node.component, None))
            else:
                z_plan.append((t, node, None, None))
                if t is DriverNode:
                    driver_nodes.append(node)
        
        build_z(root)
        
        # 2. Parcours Prefix (Top-Down) pour planifier la distribution des tensions
        def build_v(node):
            t = type(node)
            if t in (SeriesNode, ParallelNode):
                v_plan.append((t, node, node.left, node.right))
                build_v(node.left)
                build_v(node.right)
            elif t is ShuntNode:
                v_plan.append((t, node, node.component, None))
                build_v(node.component)
                
        build_v(root)
        
        # On sauvegarde le plan d'exécution directement dans le nœud racine
        root._z_plan = z_plan
        root._v_plan = v_plan
        root._driver_nodes = driver_nodes

    def get_impedance(self, node, clear_cache=False):
        """Rétrocompatibilité : Retourne l'impédance si calculée."""
        if not hasattr(node, '_Z') or clear_cache:
            self.evaluate(node)
        return node._Z

    def evaluate(self, root, v_in=1.0):
        """
        Exécution du plan compilé. Boucle plate 100% vectorisée Numpy.
        """
        # Compilation "JIT" si c'est la première fois qu'on évalue cet arbre
        if not hasattr(root, '_z_plan'):
            self.compile_tree(root)
            
        # --- PASSE 1 : Calcul des Impédances (Z) ---
        # L'utilisation de pointeurs mémoire directs (l, r) évite les lookups
        for t, n, l, r in root._z_plan:
            if t is Resistor:
                n._Z = float(n.value)
            elif t is Capacitor:
                n._Z = (self.inv_jw / n.value) + 0.01
            elif t is Inductor:
                n._Z = (self.jw * n.value) + (n.value * 400.0)
            elif t is DriverNode:
                n._Z = n.Z_complex
            elif t is SeriesNode:
                n._Z = l._Z + r._Z
            elif t is ParallelNode:
                n._Z = (l._Z * r._Z) / (l._Z + r._Z + 1e-15)
            elif t is ShuntNode:
                n._Z = l._Z
                
        # --- PASSE 2 : Distribution des Tensions (V) ---
        root._V = v_in
        for t, n, l, r in root._v_plan:
            if t is SeriesNode:
                ztot = n._Z + 1e-15
                l._V = n._V * (l._Z / ztot)
                r._V = n._V * (r._Z / ztot)
            elif t is ParallelNode:
                l._V = n._V
                r._V = n._V
            elif t is ShuntNode:
                l._V = n._V
                
        # --- PASSE 3 : Récolte des données acoustiques ---
        responses = {}
        for d in root._driver_nodes:
            # Si la tension est restée un scalaire (branchement direct), on la force en tableau
            v_out = d._V
            if np.isscalar(v_out):
                v_out = np.full_like(self.freqs, v_out, dtype=complex)
                
            responses[d.label] = {
                "V_complex": v_out,
                # La multiplication d'un scalaire avec H_acoustic gère déjà la conversion
                "P_acoustic": d._V * d.H_acoustic 
            }
            
        return responses