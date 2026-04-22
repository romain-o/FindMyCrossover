import random
from nodes import (
    Node, OperatorNode, SeriesNode, ParallelNode, ComponentNode,
    Resistor, Capacitor, Inductor, ShuntNode, DriverNode
)

class TreeMutator:
    """
    Classe responsable des mutations d'un arbre représentant un filtre passif.
    Chaque mutation prend en entrée un noeud racine, le duplique, le modifie, 
    et retourne le nouvel arbre.
    """
    
    def __init__(self, 
                 prob_value_mut=0.4, 
                 prob_type_mut=0.2, 
                 prob_topology_mut=0.2, 
                 prob_add_node=0.1,
                 prob_remove_node=0.1):
        
        self.probs = [
            (self.mutate_value, prob_value_mut),
            (self.mutate_component_type, prob_type_mut),
            (self.mutate_topology, prob_topology_mut),
            (self.add_node, prob_add_node),
            (self.remove_node, prob_remove_node)
        ]
        
    def mutate(self, root: Node) -> Node:
        """
        Applique une mutation aléatoire sur une copie de l'arbre et simplifie le résultat.
        """
        new_tree = root.copy()
        
        # Tirage au sort de la mutation basée sur les probabilités
        mutations, weights = zip(*self.probs)
        chosen_mutation = random.choices(mutations, weights=weights, k=1)[0]
        
        new_tree = chosen_mutation(new_tree)
        return self.simplify(new_tree)

    def _get_all_nodes(self, node: Node):
        """Récupère tous les nœuds de l'arbre de manière itérative (évite RecursionError)."""
        nodes = []
        stack = [node]
        visited = set()
        
        while stack:
            curr = stack.pop()
            if curr.id in visited:
                continue
            visited.add(curr.id)
            nodes.append(curr)
            
            if isinstance(curr, OperatorNode):
                stack.append(curr.right)
                stack.append(curr.left)
            elif isinstance(curr, ShuntNode):
                stack.append(curr.component)
        return nodes

    def _get_parent_map(self, node: Node):
        """Construit un dictionnaire map[node_id] = (parent_node, attribut_enfant)"""
        parent_map = {node.id: (None, None)}
        stack = [node]
        
        while stack:
            curr = stack.pop()
            if isinstance(curr, OperatorNode):
                parent_map[curr.left.id] = (curr, 'left')
                parent_map[curr.right.id] = (curr, 'right')
                stack.append(curr.right)
                stack.append(curr.left)
            elif isinstance(curr, ShuntNode):
                parent_map[curr.component.id] = (curr, 'component')
                stack.append(curr.component)
        return parent_map

    def _replace_node(self, tree: Node, old_node: Node, new_node: Node) -> Node:
        """Remplace old_node par new_node dans l'arbre. Retourne la nouvelle racine."""
        if tree.id == old_node.id:
            return new_node
            
        # Securité anti-cycle : new_node ne doit pas être un ancêtre de parent
        parent_map = self._get_parent_map(tree)
        parent, attr = parent_map.get(old_node.id, (None, None))
        
        if parent is not None and attr is not None:
            # Vérification supplémentaire pour éviter les boucles infinies
            # Si new_node est déjà présent dans l'arbre (sauf à la place de old_node), 
            # on prend une copie profonde pour casser toute référence circulaire
            setattr(parent, attr, new_node.copy())
            
        return tree

    def mutate_value(self, tree: Node) -> Node:
        """Modifie la valeur numérique d'un composant existant."""
        nodes = self._get_all_nodes(tree)
        components = [n for n in nodes if isinstance(n, ComponentNode)]
        
        if components:
            target = random.choice(components)
            target.mutate_value()
            
        return tree

    def mutate_component_type(self, tree: Node) -> Node:
        """Change un type de composant (R <-> C <-> L)."""
        nodes = self._get_all_nodes(tree)
        components = [n for n in nodes if isinstance(n, ComponentNode)]
        
        if components:
            target = random.choice(components)
            choices = [Resistor, Capacitor, Inductor]
            choices.remove(type(target))
            new_class = random.choice(choices)
            
            # Initialisation avec une valeur cohérente par défaut pour le nouveau type
            new_node = new_class()
            new_node.id = target.id # On conserve l'id
            
            tree = self._replace_node(tree, target, new_node)
            
        return tree

    def mutate_topology(self, tree: Node) -> Node:
        """Change un noeud Série en Parallèle ou inversement."""
        nodes = self._get_all_nodes(tree)
        operators = [n for n in nodes if isinstance(n, OperatorNode)]
        
        if operators:
            target = random.choice(operators)
            if isinstance(target, SeriesNode):
                new_node = ParallelNode(target.left, target.right)
            else:
                new_node = SeriesNode(target.left, target.right)
                
            new_node.id = target.id
            tree = self._replace_node(tree, target, new_node)
            
        return tree

    def _generate_random_component(self):
        """Génère un composant passif aléatoire."""
        comp_class = random.choice([Resistor, Capacitor, Inductor])
        return comp_class()

    def generate_random_tree(self, terminal_node: Node, max_depth: int = 3) -> Node:
        """
        Génère une structure d'arbre aléatoire.
        Le terminal_node (le HP) est TOUJOURS poussé vers la droite (fin de chaîne).
        """
        if max_depth <= 0 or random.random() < 0.3:
            return terminal_node
            
        new_component = self._generate_random_component()
        op = random.choice(["series", "parallel", "shunt"])
        
        if op == "series":
            # Racine(Composant, Sous-Arbre-avec-HP) -> HP est bien à la fin
            return SeriesNode(new_component, self.generate_random_tree(terminal_node, max_depth - 1))
        elif op == "parallel":
            # En parallèle, l'ordre importe peu pour la topologie IN/GND
            return ParallelNode(self.generate_random_tree(terminal_node, max_depth - 1), new_component)
        else: # shunt
            return ParallelNode(self.generate_random_tree(terminal_node, max_depth - 1), new_component)

    def _generate_notch(self):
        """Génère un filtre bouchon (LCR parallèle)."""
        r = Resistor(random.uniform(1, 47))
        l = Inductor(random.uniform(0.05e-3, 0.5e-3))
        c = Capacitor(random.uniform(1e-6, 10e-6))
        # Topologie : R + (L // C)
        return SeriesNode(r, ParallelNode(l, c))
    
    def _generate_zobel(self):
        """Génère un réseau RC série (Zobel)."""
        r = Resistor(random.uniform(5, 15))
        c = Capacitor(random.uniform(5e-6, 47e-6))
        return SeriesNode(r, c)
    
    def _generate_lpad(self, target):
        """Génère un atténuateur L-Pad autour de la cible."""
        r_series = Resistor(random.uniform(0.5, 10))
        r_parallel = Resistor(random.uniform(1, 47))
        return SeriesNode(r_series, ParallelNode(r_parallel, target))

    def add_node(self, tree: Node) -> Node:
        """Insère un composant ou une macro évoluée dans l'arbre."""
        nodes = self._get_all_nodes(tree)
        target = random.choice(nodes)
        target_has_driver = len(self._get_driver_labels(target)) > 0

        choice = random.random()

        # --- STRATÉGIE 1 : Insertion de Macros sur une voie avec HP ---
        if target_has_driver and choice < 0.45:
            sub_choice = random.random()
            
            if sub_choice < 0.35: # Macro Zobel (en parallèle du HP)
                new_node = ParallelNode(target, self._generate_zobel())
                
            elif sub_choice < 0.70: # Macro Notch (en série avec le flux)
                new_node = SeriesNode(self._generate_notch(), target)
                
            else: # Macro L-Pad (atténuation propre)
                new_node = self._generate_lpad(target)

        # --- STRATÉGIE 2 : Mutation atomique classique (Composants seuls) ---
        else:
            new_component = self._generate_random_component()
            # On conserve votre logique de placement pour ne pas casser le flux
            if target_has_driver:
                # 70% de chance de mettre en série pour créer une pente
                new_node = SeriesNode(new_component, target) if random.random() < 0.7 else ParallelNode(target, new_component)
            else:
                op = random.choice([SeriesNode, ParallelNode])
                new_node = op(target, new_component)

        tree = self._replace_node(tree, target, new_node)
        return self.simplify(tree)

    def _get_driver_labels(self, node: Node) -> set:
        """Retourne l'ensemble des labels des drivers présents dans un sous-arbre."""
        return {n.label for n in self._get_all_nodes(node) if isinstance(n, DriverNode)}

    def remove_node(self, tree: Node) -> Node:
        """Supprime un noeud opérateur en veillant à ne pas perdre de driver unique."""
        nodes = self._get_all_nodes(tree)
        operators = [n for n in nodes if isinstance(n, OperatorNode)]
        
        if operators:
            target = random.choice(operators)
            
            # On vérifie quels drivers sont dans chaque branche
            drivers_left = self._get_driver_labels(target.left)
            drivers_right = self._get_driver_labels(target.right)
            
            # Si une branche contient un driver que l'autre n'a pas, on doit garder cette branche
            if drivers_left and not drivers_right:
                child_to_keep = target.left
            elif drivers_right and not drivers_left:
                child_to_keep = target.right
            elif not drivers_left and not drivers_right:
                child_to_keep = random.choice([target.left, target.right])
            else:
                # Les deux branches ont des drivers (ex: crossover série), 
                # supprimer l'opérateur est risqué, on annule.
                return tree
                
            tree = self._replace_node(tree, target, child_to_keep)
        return tree

    def simplify(self, node: Node) -> Node:
        """
        Simplifie l'arbre : Supprime les ShuntNodes devenus inutiles (Numpy) 
        et fusionne TOUS les composants identiques d'une même branche (série ou parallèle),
        même s'ils sont séparés par d'autres sous-nœuds.
        """
        # 1. Élimination du ShuntNode
        if isinstance(node, ShuntNode):
            return self.simplify(node.component)

        if isinstance(node, SeriesNode):
            node.left = self.simplify(node.left)
            node.right = self.simplify(node.right)

            # Fonction pour aplatir toute la branche série
            def flatten_series(n):
                if n is None: return []
                if isinstance(n, SeriesNode):
                    return flatten_series(n.left) + flatten_series(n.right)
                return [n]
            
            elements = flatten_series(node)
            
            # Trier et grouper les composants
            resistors = [e for e in elements if isinstance(e, Resistor)]
            inductors = [e for e in elements if isinstance(e, Inductor)]
            capacitors = [e for e in elements if isinstance(e, Capacitor)]
            others = [e for e in elements if not isinstance(e, (Resistor, Inductor, Capacitor))]
            
            new_elements = []
            
            # Fusion mathématique en SÉRIE
            if resistors:
                new_elements.append(Resistor(sum(r.value for r in resistors)))
            if inductors:
                new_elements.append(Inductor(sum(l.value for l in inductors)))
            if capacitors:
                inv_c = sum(1.0 / (c.value + 1e-15) for c in capacitors)
                new_elements.append(Capacitor(1.0 / inv_c))
                
            new_elements.extend(others)
            
            # Sécurité anti-nœud vide
            if not new_elements:
                return None
            if len(new_elements) == 1:
                return new_elements[0]
                
            # Reconstruire un bel arbre (penché vers la droite pour Schemdraw)
            root = new_elements[-1]
            for e in reversed(new_elements[:-1]):
                root = SeriesNode(e, root)
            return root

        elif isinstance(node, ParallelNode):
            node.left = self.simplify(node.left)
            node.right = self.simplify(node.right)

            # Fonction pour aplatir toute la branche parallèle
            def flatten_parallel(n):
                if n is None: return []
                if isinstance(n, ParallelNode):
                    return flatten_parallel(n.left) + flatten_parallel(n.right)
                return [n]
                
            elements = flatten_parallel(node)
            
            resistors = [e for e in elements if isinstance(e, Resistor)]
            inductors = [e for e in elements if isinstance(e, Inductor)]
            capacitors = [e for e in elements if isinstance(e, Capacitor)]
            others = [e for e in elements if not isinstance(e, (Resistor, Inductor, Capacitor))]
            
            new_elements = []
            
            # Fusion mathématique en PARALLÈLE
            if resistors:
                inv_r = sum(1.0 / (r.value + 1e-15) for r in resistors)
                new_elements.append(Resistor(1.0 / inv_r))
            if inductors:
                inv_l = sum(1.0 / (l.value + 1e-15) for l in inductors)
                new_elements.append(Inductor(1.0 / inv_l))
            if capacitors:
                new_elements.append(Capacitor(sum(c.value for c in capacitors)))
                
            new_elements.extend(others)
            
            if not new_elements:
                return None
            if len(new_elements) == 1:
                return new_elements[0]
                
            # Reconstruire l'arbre
            root = new_elements[-1]
            for e in reversed(new_elements[:-1]):
                root = ParallelNode(e, root)
            return root

        return node

    def crossover(self, parent1: Node, parent2: Node) -> Node:
        """
        Combine deux arbres et simplifie le résultat.
        """
        child = parent1.copy()
        nodes_child = self._get_all_nodes(child)
        
        target = random.choice(nodes_child)
        target_drivers = self._get_driver_labels(target)
        
        nodes_p2 = self._get_all_nodes(parent2)
        candidates = [n for n in nodes_p2 if self._get_driver_labels(n) == target_drivers]
        
        if not candidates:
            return self.simplify(child)
            
        replacement = random.choice(candidates).copy()
        child = self._replace_node(child, target, replacement)
        
        return self.simplify(child)
