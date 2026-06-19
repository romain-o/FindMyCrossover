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
            d += elm.Line().right().length(0.5)
            SpeakerType = elm.Speaker if hasattr(elm, 'Speaker') else elm.Resistor
            label_text = getattr(node, 'model_name', node.label)
            
            wiring = getattr(self.tree, 'wiring', {}).get(node.label, 'parallel')
            count = getattr(node, 'count', 1)
            
            # --- 1 SEUL HAUT-PARLEUR (Standard) ---
            if count == 1:
                if direction == 'right':
                    d += SpeakerType().right().label(label_text, loc='top')
                    self._track_x(d)
                    d += elm.Line().down().length(1.5)
                    d += elm.Ground()
                    self._track_y(d)
                else: 
                    d += SpeakerType().down().label(label_text, loc='bottom')
                    self._track_y(d)
                    self._track_x(d)
                    
            # --- 2 HAUT-PARLEURS EN SÉRIE ---
            elif count == 2 and wiring == 'series':
                if direction == 'right':
                    d += SpeakerType().right().label(f"{label_text} (1)", loc='top')
                    self._track_x(d)
                    d += SpeakerType().right().label(f"{label_text} (2)", loc='top')
                    self._track_x(d)
                    d += elm.Line().down().length(1.5)
                    d += elm.Ground()
                    self._track_y(d)
                else:
                    d += SpeakerType().down().label(f"{label_text} (1)", loc='bottom')
                    self._track_y(d)
                    d += SpeakerType().down().label(f"{label_text} (2)", loc='bottom')
                    self._track_y(d)
                    self._track_x(d)

            # --- 2 HAUT-PARLEURS EN PARALLÈLE ---
            elif count == 2 and wiring == 'parallel':
                if direction == 'right':
                    d += elm.Line().right().length(0.5)
                    d += elm.Dot()
                    
                    d.push()
                    d += elm.Line().up().length(1.5)
                    d += SpeakerType().right().label(f"{label_text} (A)", loc='top')
                    self._track_x(d)
                    d += elm.Line().down().length(1.5)
                    d += elm.Ground()
                    d.pop()
                    
                    d.push()
                    d += elm.Line().down().length(1.5)
                    d += SpeakerType().right().label(f"{label_text} (B)", loc='bottom')
                    self._track_x(d)
                    d += elm.Line().down().length(1.5)
                    d += elm.Ground()
                    self._track_y(d)
                    d.pop()
                else:
                    d += SpeakerType().down().label(f"2x {label_text} (Parallèle)", loc='bottom')
                    self._track_y(d)
                    self._track_x(d)
            return

        # 2. Composants Passifs
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
        else: 
            d += comp.down().label(label_str, loc='bottom')
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
            
            # --- SPLIT INTERNE DES VOIES (ex: Mid / Tweeter) ---
            if has_drv_left and has_drv_right:
                d += elm.Line().right().length(0.5)
                d += elm.Dot()
                split_inner = d.here
                
                # Voie supérieure
                d.push()
                d += elm.Line().up().length(1.0)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, node.left, 'right')
                self._track_y(d)
                d.pop()
                
                # Descente dynamique vers la voie inférieure
                safe_y = min(split_inner.y - 2.5, self.current_min_y - 3.5)
                
                d.push()
                d += elm.Line().down().length(split_inner.y - safe_y)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, node.right, 'right')
                d.pop()

            # --- DÉRIVATION SHUNT (Un seul driver en aval) ---
            elif has_drv_left or has_drv_right:
                main = node.left if has_drv_left else node.right
                shunt = node.right if has_drv_left else node.left
                
                pre_shunt_x = d.here.x
                d.push()
                d += elm.Dot()
                
                prev_max_x = self.current_max_x
                self.current_max_x = pre_shunt_x
                
                self._draw_branch(d, shunt, 'down')
                d += elm.Ground()
                self._track_y(d)
                self._track_x(d)
                
                shunt_max_x = self.current_max_x
                self.current_max_x = max(prev_max_x, shunt_max_x)
                d.pop()
                
                safe_x = max(pre_shunt_x + 1.5, shunt_max_x + 1)
                advance = safe_x - pre_shunt_x
                
                if advance > 0:
                    d += elm.Line().right().length(advance)
                    
                self._draw_branch(d, main, direction)
                
            # --- COMPOSANTS EN DÉRIVATION (Filtre Bouchon / Parallèle) ---
            else:
                if direction == 'right':
                    left_pad = d.here
                    d += elm.Line().at(left_pad).right().length(0.5)
                    self._track_x(d)
                    d += elm.Dot()
                    start_dot = d.here
                    
                    # Le premier composant reste sur la ligne de la branche principale
                    self._draw_branch(d, node.left, direction)
                    end_dot = d.here
                    
                    # Le second composant passe en dessous
                    d.push()
                    d += elm.Line().at(start_dot).down().length(2.5)
                    self._draw_branch(d, node.right, direction)
                    bot_end = d.here
                    self._track_y(d)
                    
                    max_x = max(end_dot.x, bot_end.x)
                    
                    main_final = end_dot
                    if max_x > end_dot.x:
                        d += elm.Line().at(end_dot).right().length(max_x - end_dot.x)
                        main_final = d.here
                        
                    bot_final = bot_end
                    if max_x > bot_end.x:
                        d += elm.Line().at(bot_end).right().length(max_x - bot_end.x)
                        bot_final = d.here
                        
                    d += elm.Line().at(bot_final).to(main_final)
                    d.pop()
                    
                    # --- NOUVEAU : ESPACEMENT HORIZONTAL ---
                    d += elm.Dot().at(main_final)
                    d += elm.Line().at(main_final).right().length(1)
                    self._track_x(d)
                    
                elif direction == 'down':
                    d += elm.Dot()
                    start_dot = d.here
                    
                    # Le premier composant reste sur la ligne de descente
                    self._draw_branch(d, node.left, direction)
                    end_dot = d.here
                    self._track_y(d)
                    
                    # Le second s'enroule par la droite
                    d.push()
                    d += elm.Line().at(start_dot).right().length(2.0)
                    self._draw_branch(d, node.right, direction)
                    right_end = d.here
                    self._track_x(d)
                    
                    min_y = min(end_dot.y, right_end.y)
                    
                    main_final = end_dot
                    if min_y < end_dot.y:
                        d += elm.Line().at(end_dot).down().length(end_dot.y - min_y)
                        main_final = d.here
                        
                    right_final = right_end
                    if min_y < right_end.y:
                        d += elm.Line().at(right_end).down().length(right_end.y - min_y)
                        right_final = d.here
                        
                    d += elm.Line().at(right_final).to(main_final)
                    d.pop()
                    
                    # --- NOUVEAU : ESPACEMENT VERTICAL ---
                    d += elm.Dot().at(main_final)
                    d += elm.Line().at(main_final).down().length(1.25)
                    self._track_y(d)

    def save(self, filename="crossover_schematic.png"):
        self.comp_counts = {'C': 0, 'L': 0, 'R': 0}
        self.current_min_y = float('inf')
        self.current_max_x = float('-inf')

        with schemdraw.Drawing(file=filename, show=False, dpi=300) as d:
            d.config(fontsize=10)
            
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
                d += elm.Line().up().length(1.5)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, self.tree.left, 'right')
                self._track_y(d) 
                d.pop()
                
                # CALCUL DE LA MARGE VERTICALE GLOBALE
                safe_y = min(split_dot.y - 2.5, self.current_min_y - 3.5)
                drop_length = split_dot.y - safe_y
                
                # VOIES 2/3 : DESCENTE VERS MID ET TWEETER
                d.push()
                d += elm.Line().down().length(drop_length)
                d += elm.Line().right().length(0.5)
                self._draw_branch(d, self.tree.right, 'right')
                d.pop()
            else:
                self._draw_branch(d, self.tree, 'right')
                
        print(f"[+] Schéma électronique généré : {filename}")