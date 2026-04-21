import uuid
import random
import copy
import numpy as np
import tempfile
import webbrowser
import os

class Node:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
    
    def copy(self):
        """Version personnalisée de copie pour éviter RecursionError et cycles."""
        # On utilise une approche de reconstruction pour être sûr de casser les cycles
        if isinstance(self, SeriesNode):
            new_node = SeriesNode(self.left.copy(), self.right.copy())
        elif isinstance(self, ParallelNode):
            new_node = ParallelNode(self.left.copy(), self.right.copy())
        elif isinstance(self, ShuntNode):
            new_node = ShuntNode(self.component.copy())
        elif isinstance(self, Resistor):
            new_node = Resistor(self.value)
        elif isinstance(self, Capacitor):
            new_node = Capacitor(self.value)
        elif isinstance(self, Inductor):
            new_node = Inductor(self.value)
        elif isinstance(self, DriverNode):
            # Les données lourdes (H_acoustic, Z_complex) sont partagées (non modifiées)
            new_node = DriverNode(self.label, "dummy", "dummy")
            # On court-circuite le load_data pour la copie
            new_node.frd_freqs = self.frd_freqs
            new_node.zma_freqs = self.zma_freqs
            new_node.H_acoustic = self.H_acoustic
            new_node.Z_complex = self.Z_complex
        else:
            raise ValueError(f"Type inconnu pour la copie: {type(self)}")
        
        new_node.id = str(uuid.uuid4())[:8] # Toujours un nouvel ID pour la copie
        return new_node

    def get_all_nodes(self):
        """Utile pour choisir un point de mutation aléatoire"""
        nodes = [self]
        # Utilisation de hasattr pour éviter les problèmes d'ordre de déclaration
        if hasattr(self, 'left') and hasattr(self, 'right'):
            nodes.extend(self.left.get_all_nodes())
            nodes.extend(self.right.get_all_nodes())
        elif hasattr(self, 'component'):
            nodes.extend(self.component.get_all_nodes())
        return nodes

    @staticmethod
    def from_dict(data):
        """Reconstruit un arbre à partir d'un dictionnaire (JSON)."""
        node_type = data["type"]
        
        if node_type == "SeriesNode":
            return SeriesNode(Node.from_dict(data["left"]), Node.from_dict(data["right"]))
        elif node_type == "ParallelNode":
            return ParallelNode(Node.from_dict(data["left"]), Node.from_dict(data["right"]))
        elif node_type == "ShuntNode":
            return ShuntNode(Node.from_dict(data["component"]))
        elif node_type == "Resistor":
            return Resistor(data["value"])
        elif node_type == "Capacitor":
            return Capacitor(data["value"])
        elif node_type == "Inductor":
            return Inductor(data["value"])
        elif node_type == "DriverNode":
            node = DriverNode(data["label"], "dummy", "dummy")
            node.polarity_inverted = data.get("polarity_inverted", False)
            return node
        
        raise ValueError(f"Type inconnu: {node_type}")

    def to_dict(self):
        """Convertit l'arbre en dictionnaire pour sérialisation JSON."""
        data = {
            "type": self.__class__.__name__,
            "id": self.id
        }
        
        if hasattr(self, 'value'):
            data["value"] = self.value
            data["unit"] = self.unit
        
        if hasattr(self, 'label'):
            data["label"] = self.label
            if hasattr(self, 'polarity_inverted'):
                data["polarity_inverted"] = bool(self.polarity_inverted)
            
        if hasattr(self, 'left') and hasattr(self, 'right'):
            data["left"] = self.left.to_dict()
            data["right"] = self.right.to_dict()
        elif hasattr(self, 'component'):
            data["component"] = self.component.to_dict()
            
        return data

    def display(self, prefix="", is_last=True, is_root=True):
        """Affiche l'arbre de composants de manière intuitive dans la console."""
        connector = "" if is_root else ("\\-- " if is_last else "|-- ")
        
        if hasattr(self, 'value') and hasattr(self, 'unit'):
            # Arrondi à 3 chiffres significatifs pour la lisibilité
            val_str = f"{self.value:.3e}"
            node_str = f"{self.__class__.__name__} ({val_str} {self.unit})"
        elif hasattr(self, 'label'):
            pol_str = " (INV)" if getattr(self, 'polarity_inverted', False) else ""
            node_str = f"Driver '{self.label}'{pol_str}"
        else:
            node_str = self.__class__.__name__
            
        # Affichage avec l'ID pour faciliter le débogage si besoin
        print(f"{prefix}{connector}{node_str}")
        
        # Préparation du préfixe pour les enfants
        new_prefix = prefix + ("" if is_root else ("    " if is_last else "|   "))
        
        # Appels récursifs
        if hasattr(self, 'left') and hasattr(self, 'right'):
            self.left.display(new_prefix, is_last=False, is_root=False)
            self.right.display(new_prefix, is_last=True, is_root=False)
        elif hasattr(self, 'component'):
            self.component.display(new_prefix, is_last=True, is_root=False)

    def _generate_mermaid(self):
        lines = ["graph TD"]
        
        def traverse(node, parent_id=None):
            node_id = f"n_{node.id.replace('-', '_')}"
            
            # Formatage du nom et de la valeur du noeud courant
            if hasattr(node, 'value') and hasattr(node, 'unit'):
                val_str = f"{node.value:.2e}"
                label = f"{node.__class__.__name__}<br>{val_str} {node.unit}"
                shape_start, shape_end = "(", ")" # Composants avec des bords arrondis
            elif hasattr(node, 'label'):
                label = f"Driver<br>{node.label}"
                shape_start, shape_end = "[/", "/]" # Haut parleur forme specifique
            else:
                label = node.__class__.__name__
                shape_start, shape_end = "[", "]" # Operateur carre
                
            lines.append(f'  {node_id}{shape_start}"{label}"{shape_end}')
            
            if parent_id:
                lines.append(f'  {parent_id} --> {node_id}')
                
            # Appels récursifs
            if hasattr(node, 'left') and hasattr(node, 'right'):
                traverse(node.left, node_id)
                traverse(node.right, node_id)
            elif hasattr(node, 'component'):
                traverse(node.component, node_id)

        traverse(self)
        return "\n".join(lines)

    def get_bom(self):
        """Retourne la liste plate de tous les composants physiques (BOM)."""
        bom = []
        nodes = self.get_all_nodes()
        for n in nodes:
            if isinstance(n, (Resistor, Capacitor, Inductor)):
                bom.append({
                    "type": n.__class__.__name__,
                    "value": n.value,
                    "unit": n.unit,
                    "id": n.id
                })
            elif isinstance(n, DriverNode):
                 bom.append({
                    "type": "Driver",
                    "label": n.label,
                    "id": n.id
                })
        return bom

    def _generate_circuit_mermaid(self):
        lines = ["graph LR", "  IN((IN+))", "  GND((GND))"]
        node_counter = 0

        def get_new_node():
            nonlocal node_counter
            node_counter += 1
            return f"N{node_counter}"

        def traverse(node, in_node, out_node):
            if isinstance(node, ComponentNode):
                val_str = f"{node.value:.2e}"
                label = f"{node.__class__.__name__} ({val_str} {node.unit})"
                lines.append(f"  {in_node} -- \"{label}\" --- {out_node}")
            elif isinstance(node, DriverNode):
                label = f"Driver {node.label}"
                lines.append(f"  {in_node} -- \"{label}\" --- {out_node}")
            elif isinstance(node, SeriesNode):
                mid_node = get_new_node()
                lines.append(f"  {mid_node}(( ))")
                traverse(node.left, in_node, mid_node)
                traverse(node.right, mid_node, out_node)
            elif isinstance(node, ParallelNode):
                traverse(node.left, in_node, out_node)
                traverse(node.right, in_node, out_node)
            elif isinstance(node, ShuntNode):
                # Un shunt va du signal à la masse
                traverse(node.component, in_node, "GND")
            else:
                lines.append(f"  {in_node} -- \"{node.__class__.__name__}\" --- {out_node}")

        traverse(self, "IN", "GND")
        
        lines.append("  classDef gnd fill:#000,stroke:#000,color:#fff;")
        lines.append("  class GND gnd;")
        
        return "\n".join(lines)

    def save_as_html(self, filename="circuit_viz.html"):
        """Génère un fichier HTML autonome pour visualiser le circuit."""
        mermaid_ast = self._generate_mermaid()
        mermaid_circuit = self._generate_circuit_mermaid()
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Visualisation du Filtre</title>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    <style>
        body {{ font-family: sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; background-color: #f4f4f9; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 20px; width: 100%; justify-content: center; }}
        .panel {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); min-width: 45%; }}
        h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>Structure du Filtre Optimisé</h1>
    <div class="container">
        <div class="panel">
            <h2>Arbre Logique</h2>
            <div class="mermaid">{mermaid_ast}</div>
        </div>
        <div class="panel">
            <h2>Schéma Électrique</h2>
            <div class="mermaid">{mermaid_circuit}</div>
        </div>
    </div>
</body>
</html>"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filename
    
class OperatorNode(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

class SeriesNode(OperatorNode):
    def __str__(self):
        return f"Series({self.left}, {self.right})"

class ParallelNode(OperatorNode):
    def __str__(self):
        return f"Parallel({self.left}, {self.right})"
    
class ComponentNode(Node):
    def __init__(self, value, unit):
        super().__init__()
        self.value = value
        self.unit = unit

    def mutate_value(self):
        """Modifie légèrement la valeur (ex: +/- 10%)"""
        self.value *= random.uniform(0.9, 1.1)

    def __str__(self):
        return f"{self.__class__.__name__}({self.value:.2e}{self.unit})"

class Resistor(ComponentNode):
    def __init__(self, value=10): super().__init__(value, "Ohm")

class Capacitor(ComponentNode):
    def __init__(self, value=1e-6): super().__init__(value, "F")

class Inductor(ComponentNode):
    def __init__(self, value=1e-3): super().__init__(value, "H")
    
class ShuntNode(Node):
    """Connecte un composant entre le signal et la masse (GND)"""
    def __init__(self, component):
        super().__init__()
        self.component = component

    def __str__(self):
        return f"Shunt({self.component})"

class DriverNode(Node):
    def __init__(self, label, frd_path, zma_path):
        super().__init__()
        self.label = label
        self.polarity_inverted = False
        
        if frd_path == "dummy":
            # On laisse les attributs être remplis par la méthode copy()
            self.frd_freqs = None
            self.zma_freqs = None
            self.Z_complex = None
            self.H_acoustic = None
            return

        # 1. Chargement des données brutes
        self.zma_freqs, z_mag, z_phase = self.load_data(zma_path)
        self.frd_freqs, spl_db, spl_phase = self.load_data(frd_path)
        
        # 2. Conversion de l'impédance en tableau de complexes
        self.Z_complex = z_mag * np.exp(1j * np.radians(z_phase))
        
        # 3. Conversion de la réponse acoustique en tableau de complexes
        self.H_acoustic = (10 ** (spl_db / 20)) * np.exp(1j * np.radians(spl_phase))

    def load_data(self, filepath):
        """Lit un fichier texte à 3 colonnes et retourne des arrays numpy, en ignorant les en-têtes."""
        # On tente de charger uniquement les 3 premières colonnes (Freq, Mag, Phase)
        # skip_header=1 aide souvent, mais genfromtxt avec invalid_raise=False est plus souple
        data = np.genfromtxt(filepath, usecols=(0, 1, 2), invalid_raise=False)
        
        # Si le résultat est 1D ou vide, il y a un problème de parsing
        if data.ndim == 1:
            # On réessaye avec une méthode plus manuelle si genfromtxt échoue à voir les colonnes
            with open(filepath, 'r') as f:
                lines = f.readlines()
            parsed_data = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        parsed_data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
            data = np.array(parsed_data)

        # Nettoyage final des NaN
        if data.size > 0:
            data = data[~np.isnan(data).any(axis=1)]
            
        return data[:, 0], data[:, 1], data[:, 2]

    def __str__(self):
        return f"Driver({self.label}, FRD/ZMA chargés)"
    
