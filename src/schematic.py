import schemdraw
import schemdraw.elements as elm
from src.nodes import DriverNode, SeriesNode, ParallelNode, ComponentNode, Resistor, Capacitor, Inductor

class SchematicRenderer:
    def __init__(self, tree):
        self.tree = tree
        self.comp_counts = {'C': 0, 'L': 0, 'R': 0}
        self.current_min_y = float('inf')
        self.current_max_x = float('-inf')

    def _track_y(self, d):
        """Enregistre le point le plus bas atteint (Marge Verticale)"""
        if hasattr(d, 'here'):
            self.current_min_y = min(self.current_min_y, d.here.y)

    def _track_x(self, d):
        """Enregistre le point le plus à droite atteint (Marge Horizontale)"""
        if hasattr(d, 'here'):
            self.current_max_x = max(self.current_max_x, d.here.x)

    def _has_driver(self, node):
        if isinstance(node, DriverNode): return True
        if hasattr(node, 'left') and hasattr(node, 'right'):
            return self._has_driver(node.left) or self._has_driver(node.right)
        return False

    def _draw_component(self, d, node, direction):
        # 1. Haut-Parleurs
        if isinstance(node, DriverNode):
            try: comp = elm.Speaker()
            except AttributeError: comp = elm.Resistor()
            
            label_text = getattr(node, 'model_name', node.label)
            
            if direction == 'right':
                d += comp.right().label(label_text, loc='top')
                self._track_x(d)
                d += elm.Line().down().length(1.5)
                d += elm.Ground()
                self._track_y(d)
            else: # direction == 'down'
                d += comp.down().label(label_text, loc='bottom')
                self._track_y(d)
                self._track_x(d)
            return

        # 2. Composants Passifs (Avec Nomenclature)
        if isinstance(node, Resistor):
            self.comp_counts['R'] += 1
            name = f"R{self.comp_counts['R']}"
            val = f"{node.value:.1f} Ω"
            comp = elm.Resistor()
        elif isinstance(node, Capacitor):
            self.comp_counts['C'] += 1
            name = f"C{self.comp_counts['C']}"
            val = f"{node.value*1e6:.1f} µF"
            comp = elm.Capacitor()
        elif isinstance(node, Inductor):
            self.comp_counts['L'] += 1
            name = f"L{self.comp_counts['L']}"
            val = f"{node.value*1000:.2f} mH"
            comp = elm.Inductor2()

        label_str = f"{name}\n{val}"

        if direction == 'right':
            d += comp.right().label(label_str, loc='top')
            self._track_x(d)
        else: # direction == 'down'
            d += comp.down().label(label_str, loc='bottom') # bottom force l'affichage à gauche
            self._track_y(d)
            self._track_x(d)

    def _draw_branch(self, d, node, direction='right'):
        if isinstance(node, (ComponentNode, DriverNode)):
            self._draw_component(d, node, direction)
            
        elif isinstance(node, SeriesNode):
            self._draw_branch(d, node.left, direction)
            self._draw_branch(d, node.right, direction)
            
        elif isinstance(node, ParallelNode):
            has_drv_left = self._has_driver(node.left)
            has_drv_right = self._has_driver(node.right)
            
            if has_drv_left and has_drv_right:
                d += elm.Line().right().length(0.5)
                d += elm.Dot()
                split_inner = d.here
                
                d.push()
                d += elm.Line().up().length(2.0)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, node.left, 'right')
                self._track_y(d)
                d.pop()
                
                safe_y = min(split_inner.y - 2.5, self.current_min_y - 2.5)
                
                d.push()
                d += elm.Line().down().length(split_inner.y - safe_y)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, node.right, 'right')
                d.pop()

            # --- LE CALCUL MAGIQUE HORIZONTAL (Shunt) ---
            elif has_drv_left or has_drv_right:
                main = node.left if has_drv_left else node.right
                shunt = node.right if has_drv_left else node.left
                
                pre_shunt_x = d.here.x
                d.push()
                d += elm.Dot()
                
                # On isole le tracking X pour mesurer la largeur exacte de la branche
                prev_max_x = self.current_max_x
                self.current_max_x = pre_shunt_x
                
                self._draw_branch(d, shunt, 'down')
                d += elm.Ground()
                self._track_y(d)
                self._track_x(d)
                
                shunt_max_x = self.current_max_x
                self.current_max_x = max(prev_max_x, shunt_max_x) # Restauration du max global
                d.pop()
                
                # Le composant descend, son texte prend environ 1.8 unités à droite.
                # Si le shunt s'est étendu (sous-circuit), on prend sa vraie largeur + 0.5.
                safe_x = max(pre_shunt_x + 2.2, shunt_max_x + 1)
                advance = safe_x - pre_shunt_x
                
                if advance > 0:
                    d += elm.Line().right().length(advance)
                    
                self._draw_branch(d, main, direction)
                
            # --- Filtre Bouchon (Notch) : Rendu Compact ---
            else:
                if direction == 'right':
                    d += elm.Line().right().length(0.2)
                    
                    d.push()
                    d += elm.Line().up().length(1.5)
                    self._draw_branch(d, node.left, direction)
                    top_end = d.here
                    d.pop()
                    
                    d.push()
                    d += elm.Line().down().length(1.5)
                    self._draw_branch(d, node.right, direction)
                    bot_end = d.here
                    self._track_y(d) 
                    d.pop()
                    
                    max_x = max(top_end.x, bot_end.x)
                    
                    top_final = top_end
                    if max_x > top_end.x:
                        d += elm.Line().at(top_end).right().length(max_x - top_end.x)
                        top_final = d.here
                        
                    bot_final = bot_end
                    if max_x > bot_end.x:
                        d += elm.Line().at(bot_end).right().length(max_x - bot_end.x)
                        bot_final = d.here
                        
                    d += elm.Line().at(top_final).to(bot_final)
                    
                    mid_y = (top_final.y + bot_final.y) / 2
                    d += elm.Line().at((max_x, mid_y)).right().length(0.2)
                    self._track_x(d)
                    
                elif direction == 'down':
                    d += elm.Line().down().length(0.2)
                    
                    d.push()
                    d += elm.Line().left().length(1.5) # Rendu très compact horizontalement
                    self._draw_branch(d, node.left, direction)
                    left_end = d.here
                    self._track_y(d)
                    self._track_x(d)
                    d.pop()
                    
                    d.push()
                    d += elm.Line().right().length(1.5)
                    self._draw_branch(d, node.right, direction)
                    right_end = d.here
                    self._track_y(d)
                    self._track_x(d)
                    d.pop()
                    
                    min_y = min(left_end.y, right_end.y)
                    
                    left_final = left_end
                    if min_y < left_end.y:
                        d += elm.Line().at(left_end).down().length(left_end.y - min_y)
                        left_final = d.here
                        
                    right_final = right_end
                    if min_y < right_end.y:
                        d += elm.Line().at(right_end).down().length(right_end.y - min_y)
                        right_final = d.here
                        
                    d += elm.Line().at(left_final).to(right_final)
                    
                    mid_x = (left_final.x + right_final.x) / 2
                    d += elm.Line().at((mid_x, min_y)).down().length(0.2)
                    self._track_y(d)

    def save(self, filename="crossover_schematic.png"):
        self.comp_counts = {'C': 0, 'L': 0, 'R': 0}
        self.current_min_y = float('inf')
        self.current_max_x = float('-inf')

        with schemdraw.Drawing(file=filename, show=False, dpi=300) as d:
            d.config(fontsize=10) # 10 est la police idéale pour les schémas compacts
            
            start_pos = d.here 
            d += elm.Dot().label('IN+', loc='left')
            
            d.push()
            d += elm.Dot().at((start_pos.x, start_pos.y - 1.5)).label('IN-', loc='left')
            d += elm.Ground()
            d.pop()
            
            d += elm.Line().right().length(0.5)
            
            if isinstance(self.tree, ParallelNode) and self._has_driver(self.tree.left) and self._has_driver(self.tree.right):
                d += elm.Dot()
                split_dot = d.here
                
                # VOIE 1 : WOOFER
                d.push()
                d += elm.Line().up().length(1.5) # Départ très serré
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, self.tree.left, 'right')
                self._track_y(d) 
                d.pop()
                
                # CALCUL DE LA MARGE VERTICALE SÉCURISÉE (2.5 unités d'air garanti)
                safe_y = min(split_dot.y - 2.5, self.current_min_y - 2.5)
                drop_length = split_dot.y - safe_y
                
                # VOIE 2 : TWEETER
                d.push()
                d += elm.Line().down().length(drop_length)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, self.tree.right, 'right')
                d.pop()
            else:
                self._draw_branch(d, self.tree, 'right')
                
        print(f"[+] Schéma électronique généré : {filename}")