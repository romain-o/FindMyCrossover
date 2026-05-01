import schemdraw
import schemdraw.elements as elm
from src.nodes import DriverNode, SeriesNode, ParallelNode, ComponentNode, Resistor, Capacitor, Inductor

class SchematicRenderer:
    def __init__(self, tree):
        self.tree = tree

    def _has_driver(self, node):
        if isinstance(node, DriverNode): return True
        if hasattr(node, 'left') and hasattr(node, 'right'):
            return self._has_driver(node.left) or self._has_driver(node.right)
        return False

    def _draw_component(self, d, node, direction):
        if isinstance(node, Resistor):
            comp, label = elm.Resistor(), f"{node.value:.1f} Ω"
        elif isinstance(node, Capacitor):
            comp, label = elm.Capacitor(), f"{node.value*1e6:.1f} µF"
        elif isinstance(node, Inductor):
            comp, label = elm.Inductor2(), f"{node.value*1000:.2f} mH"
        elif isinstance(node, DriverNode):
            try:
                comp = elm.Speaker()
            except AttributeError:
                comp = elm.Resistor() # Fallback visuel
            
            # Utilise le vrai nom du HP (ex: RS225-8) s'il existe, sinon "Woofer"
            label = getattr(node, 'model_name', node.label)
            
        if direction == 'right':
            d += comp.right().label(label)
            if isinstance(node, DriverNode):
                # Retour de masse propre après le haut-parleur (orienté vers la droite)
                d += elm.Line().down().length(1.5)
                d += elm.Ground()
        else: # direction == 'down'
            d += comp.down().label(label, loc='bottom')
            if isinstance(node, DriverNode):
                d += elm.Ground()

    def _draw_branch(self, d, node, direction='right'):
        if isinstance(node, (ComponentNode, DriverNode)):
            self._draw_component(d, node, direction)
            
        elif isinstance(node, SeriesNode):
            self._draw_branch(d, node.left, direction)
            self._draw_branch(d, node.right, direction)
            
        elif isinstance(node, ParallelNode):
            has_drv_left = self._has_driver(node.left)
            has_drv_right = self._has_driver(node.right)
            
            if has_drv_left or has_drv_right:
                # TOPOLOGIE SHUNT (Dérivation vers la masse)
                main = node.left if has_drv_left else node.right
                shunt = node.right if has_drv_left else node.left
                
                d.push()
                d += elm.Dot()
                self._draw_branch(d, shunt, 'down')
                d += elm.Ground()
                d.pop()
                
                # Ecartement augmenté pour éviter la superposition des textes !
                d += elm.Line().right().length(2.5)
                self._draw_branch(d, main, direction)
                
            else:
                # TOPOLOGIE BOUCHON (Tank/Notch filter)
                d += elm.Line().right().length(0.5)
                
                d.push()
                d += elm.Line().up().length(1.5)
                self._draw_branch(d, node.left, direction)
                top_end = d.here
                d.pop()
                
                d.push()
                d += elm.Line().down().length(1.5)
                self._draw_branch(d, node.right, direction)
                bot_end = d.here
                d.pop()
                
                max_x = max(top_end.x, bot_end.x)
                
                if max_x > top_end.x:
                    d += elm.Line().at(top_end).right().length(max_x - top_end.x)
                    top_final = d.here
                else:
                    top_final = top_end
                
                if max_x > bot_end.x:
                    d += elm.Line().at(bot_end).right().length(max_x - bot_end.x)
                    bot_final = d.here
                else:
                    bot_final = bot_end
                
                d += elm.Line().at(top_final).to(bot_final)
                
                mid_y = (top_final.y + bot_final.y) / 2
                d += elm.Line().at((max_x, mid_y)).right().length(0.5)

    def save(self, filename="crossover_schematic.png"):
        with schemdraw.Drawing(file=filename, show=False) as d:
            d.config(fontsize=10)
            
            # --- BLOC ENTRÉE CORRIGÉ (Plus de court-circuit !) ---
            # On mémorise la position de départ (IN+)
            start_pos = d.here 
            d += elm.Dot().label('IN+', loc='left')
            
            d.push()
            # On dessine IN- et sa masse 1.5 unités plus bas, SANS tirer de fil depuis IN+
            d += elm.Dot().at((start_pos.x, start_pos.y - 1.5)).label('IN-', loc='left')
            d += elm.Ground()
            d.pop()
            
            d += elm.Line().right().length(1)
            
            if isinstance(self.tree, ParallelNode) and self._has_driver(self.tree.left) and self._has_driver(self.tree.right):
                d += elm.Dot()
                
                # Voie 1 (Haut)
                d.push()
                d += elm.Line().up().length(2.5)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, self.tree.left, 'right')
                d.pop()
                
                # Voie 2 (Bas)
                d.push()
                d += elm.Line().down().length(2.5)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, self.tree.right, 'right')
                d.pop()
            else:
                self._draw_branch(d, self.tree, 'right')
                
        print(f"[+] Schéma électronique généré : {filename}")