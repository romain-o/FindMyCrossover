import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from src.nodes import Node, SeriesNode, ParallelNode, ShuntNode, Resistor, Capacitor, Inductor, DriverNode, ComponentNode

class VituixCloneGenerator:
    """Générateur XML strict basé sur l'empreinte réelle de VituixCAD."""
    def __init__(self, filename, target_spl):
        self.filename = filename
        self.target_spl = str(target_spl)
        self.root = ET.Element("SPEAKER")
        self.crossover_node = None
        self._init_global_params()

    def _init_global_params(self):
        params = {
            "Description": "Created by FindMyCrossover", "ReferenceAngle": "0", "SPLmax": "110", "DualPlane": "True",
            "KeywordHor": "hor", "KeywordVer": "ver", "AngleMultiplier": "1",
            "XMin": "20", "XMax": "20000", "Interpolate": "True",
            "UserAnglesHor": "", "UserAnglesVer": "", "IntensitySphere": "True",
            "IntensityCylinder": "False", "IncludeHor": "True", "IncludeVer": "True",
            "HalfSpace": "False", "Corner": "False", "LiswinDI": "True",
            "CTA2034Aweights": "True", "AngleStep": "10", "FrontWall": "False",
            "FrontWallZ": "1000", "LeftWall": "False", "LeftWallX": "-1000",
            "Ceiling": "False", "CeilingY": "1500", "Floor": "False",
            "FloorY": "-1000", "Toein": "25", "AbsorpWall": "2", "AbsorpCeil": "2",
            "AbsorpFloor": "2", "ReferDistance": "2000", "PlaneRotation": "0",
            "DrvOffsetX": "0", "DrvOffsetY": "0"
        }
        for k, v in params.items(): ET.SubElement(self.root, k).text = v
        for tag in ["AxialTarget", "PowerTarget"]:
            node = ET.SubElement(self.root, tag)
            targets = {"FreqMin": "20.0", "FreqMax": "20000.0", "SPL": self.target_spl, "Tilt": "0.0", "DrvN": "1", "Invert": "False", "FreeLF": "False", "FreeHF": "False"}
            for k, v in targets.items(): ET.SubElement(node, k).text = v

    def add_drivers(self, drivers_config):
        for idx, cfg in enumerate(drivers_config):
            d_node = ET.SubElement(self.root, "DRIVER", di=str(idx))
            ET.SubElement(d_node, "Model").text = cfg['name']
            ET.SubElement(d_node, "SPL").text = "80"; ET.SubElement(d_node, "Z").text = "8"
            ET.SubElement(d_node, "ExtendedData").text = "False"; ET.SubElement(d_node, "ResponseDirectory").text = ""
            ET.SubElement(d_node, "ResponseScale").text = "1"; ET.SubElement(d_node, "ResponseDelay").text = "0"
            ET.SubElement(d_node, "ResponseInvert").text = "False"; ET.SubElement(d_node, "ResponseMute").text = "False"
            ET.SubElement(d_node, "MinimumPhase").text = "False"; ET.SubElement(d_node, "ResponseSmooth").text = "None"
            ET.SubElement(d_node, "ImpedanceFile").text = cfg['z']
            ET.SubElement(d_node, "ImpedanceScale").text = "1"
            resp = ET.SubElement(d_node, "RESPONSE", ri="0")
            ET.SubElement(resp, "FileName").text = cfg['f']; ET.SubElement(resp, "Hor").text = "0"; ET.SubElement(resp, "Ver").text = "0"

    def generate_schematic(self, netlist):
        ET.SubElement(self.root, "Variant").text = "0"
        self.crossover_node = ET.SubElement(self.root, "CROSSOVER")
        ET.SubElement(self.crossover_node, "DSP").text = "Analog"; ET.SubElement(self.crossover_node, "SampleRate").text = "96000"
        ET.SubElement(self.crossover_node, "DSPSettings"); ET.SubElement(self.crossover_node, "DSPTemplate")
        for i, item in enumerate(netlist): 
            self._add_part_from_netlist(i, item)

    def _add_part_from_netlist(self, index, item):
        part = ET.SubElement(self.crossover_node, "PART", xi=str(index))
        layout = item['Layout']
        ET.SubElement(part, "Type").text = item['Type']
        ET.SubElement(part, "CenX").text = layout['CenX']
        ET.SubElement(part, "CenY").text = layout['CenY']
        
        if item['Type'] not in ['Wire', 'Ground']:
            ET.SubElement(part, "Open").text = "False"
            if item['Type'] != 'Generator':
                ET.SubElement(part, "Shorted").text = "False"
            ET.SubElement(part, "Rotated").text = layout['Rotated']
            if item['Component'] != "Unknown": 
                ET.SubElement(part, "PartID").text = item['Component']
        else:
            ET.SubElement(part, "Open").text = "False"
            if item['Type'] == 'Ground':
                ET.SubElement(part, "Rotated").text = "False"

        ET.SubElement(part, "GUID")
        
        if item['Type'] == 'Driver':
            ET.SubElement(part, "Model").text = item.get('Model', 'Unknown Driver')
            ET.SubElement(part, "Muted").text = "False"
            ET.SubElement(part, "Hidden").text = "False"
            ET.SubElement(part, "Inverted").text = str(item.get('Inverted', False))
            
            for t in ["DriverTarget", "FilterTarget"]:
                dt = ET.SubElement(part, t)
                ET.SubElement(dt, "FreqMin").text = "20.0"
                ET.SubElement(dt, "FreqMax").text = "20000.0"
                ET.SubElement(dt, "SPL").text = self.target_spl if t=="DriverTarget" else "0.0"
                ET.SubElement(dt, "Tilt").text = "0.0"
                ET.SubElement(dt, "DrvN").text = "1"
                ET.SubElement(dt, "Invert").text = "False"
                ET.SubElement(dt, "FreeLF").text = "False"
                ET.SubElement(dt, "FreeHF").text = "False"

        for idx, (k, v, unit) in enumerate(self._get_params(item)):
            p = ET.SubElement(part, "PARAM", pi=str(idx))
            ET.SubElement(p, "Name").text = k
            ET.SubElement(p, "Value").text = str(v)
            ET.SubElement(p, "Unit").text = unit
            ET.SubElement(p, "Optimize").text = "False"
            ET.SubElement(p, "Expression")
            ET.SubElement(p, "Min").text = "0.001"
            ET.SubElement(p, "Max").text = "100000"
            ET.SubElement(p, "OptiBlock").text = "False"

        for idx, w_data in enumerate(layout['Wires']):
            w = ET.SubElement(part, "WIRE", wi=str(idx))
            ET.SubElement(w, "X").text = str(w_data[0])
            ET.SubElement(w, "Y").text = str(w_data[1])

    def _get_params(self, item):
        val = item['Value_Main']; ctype = item['Type']
        if ctype == 'Generator': return [('Eg', val, 'V'), ('Tg', 0, 'us'), ('Rg', 0.001, 'Ω')]
        if ctype == 'Resistor': return [('R', val, 'Ω'), ('Pow', 5, 'W')]
        if ctype == 'Capacitor': return [('C', val*1e6, 'uF'), ('ESR', 0.01, 'Ω')]
        if ctype == 'Inductor': return [('L', val*1000, 'mH'), ('DCR', 0.28, 'Ω'), ('Wire', 1.4, 'mm'), ('Rpar', 1000000, 'Ω'), ('Cpar', 0.0001, 'uF')]
        if ctype == 'Driver': return [('X', 0, 'mm'), ('Y', 0, 'mm'), ('Z', 0, 'mm'), ('R', 0, 'deg'), ('T', 0, 'deg')]
        return []

    def save(self):
        xml_bytes = ET.tostring(self.root, encoding='utf-8')
        parsed = minidom.parseString(xml_bytes)
        pretty_xml = parsed.toprettyxml(indent="  ")
        lines = pretty_xml.split('\n')[1:] 
        header = '<?xml version="1.0" encoding="utf-8"?>\n<!--VituixCAD PROJECT-->\n<!--Version 2-->\n'
        with open(self.filename, "w", encoding="utf-8") as f: f.write(header + '\n'.join(lines))
        print(f"✅ Schéma VituixCAD parfaitement dessiné : {self.filename}")


class VituixAdapter:
    def __init__(self, filename="optim_crossover.vxp", target_spl=85.0):
        self.generator = VituixCloneGenerator(filename, target_spl)
        self.netlist = []
        self.id_counters = {'R': 1, 'C': 1, 'L': 1, 'D': 1}
        
    def _get_id(self, prefix):
        current_id = f"{prefix}{self.id_counters[prefix]}"
        self.id_counters[prefix] += 1
        return current_id

    def _has_driver(self, node):
        if isinstance(node, DriverNode): return True
        if hasattr(node, 'left') and hasattr(node, 'right'):
            return self._has_driver(node.left) or self._has_driver(node.right)
        if isinstance(node, ShuntNode) or hasattr(node, 'component'):
            return self._has_driver(node.component)
        return False

    def _add_trace(self, x1, y1, x2, y2):
        if x1 == x2 and y1 == y2: return
        self.netlist.append({
            'Type': 'Wire', 'Component': 'Unknown',
            'Layout': {'CenX': str((x1+x2)//2), 'CenY': str((y1+y2)//2), 'Rotated': 'False', 'Wires': [(x1, y1), (x2, y2)]},
            'Value_Main': 0
        })

    def _add_ground(self, wx, wy):
        """Place l'icône de masse exactement connectée à la broche (wx, wy)"""
        self.netlist.append({
            'Type': 'Ground', 'Component': 'Unknown',
            'Layout': {'CenX': str(wx), 'CenY': str(wy + 1), 'Rotated': 'False', 'Wires': [(wx, wy)]},
            'Value_Main': 0
        })

    def _add_component(self, comp, cx, cy, rot, w1, w2):
        ctype = comp.__class__.__name__
        prefix = 'R' if ctype=='Resistor' else 'C' if ctype=='Capacitor' else 'L'
        self.netlist.append({
            'Type': ctype, 'Component': self._get_id(prefix),
            'Layout': {'CenX': str(cx), 'CenY': str(cy), 'Rotated': 'True' if rot else 'False', 'Wires': [w1, w2]},
            'Value_Main': comp.value
        })

    def _build_circuit(self, node, x_in, y_in, direction, ways_configs, polarities):
        """
        direction = 'H' (Série) -> Avance de gauche à droite.
        direction = 'V' (Shunt/Masse) -> Tombe de haut en bas.
        Retourne : x_out, y_out, encombrement_y_max
        """
        if isinstance(node, ComponentNode):
            if direction == 'H':
                x_out, y_out = x_in + 6, y_in
                self._add_component(node, cx=x_in+3, cy=y_in, rot=False, w1=(x_in, y_in), w2=(x_out, y_out))
                return x_out, y_out, y_in + 3
            else: # Direction 'V'
                x_out, y_out = x_in, y_in + 6
                self._add_component(node, cx=x_in, cy=y_in+3, rot=True, w1=(x_in, y_in), w2=(x_out, y_out))
                return x_out, y_out, y_out + 1

        elif isinstance(node, SeriesNode):
            x_mid, y_mid, max_L = self._build_circuit(node.left, x_in, y_in, direction, ways_configs, polarities)
            x_out, y_out, max_R = self._build_circuit(node.right, x_mid, y_mid, direction, ways_configs, polarities)
            return x_out, y_out, max(max_L, max_R)

        elif isinstance(node, ShuntNode):
            x_tap = x_in + 2
            self._add_trace(x_in, y_in, x_tap, y_in)

            # La dérivation plonge vers la MASSE en VERTICAL ('V')
            xb, yb, max_S = self._build_circuit(node.component, x_tap, y_in, 'V', ways_configs, polarities)
            self._add_ground(xb, yb)

            # La ligne en série continue
            x_out = x_tap + 2
            self._add_trace(x_tap, y_in, x_out, y_in)
            return x_out, y_in, max_S + 1

        elif isinstance(node, ParallelNode):
            has_L = self._has_driver(node.left)
            has_R = self._has_driver(node.right)

            if has_L and has_R:
                # Crossover 2-voies : On déporte avec un fil horizontal pour éviter le court-circuit du drop !
                x_drop = x_in + 2
                x_start = x_drop + 2
                self._add_trace(x_in, y_in, x_drop, y_in)

                # Chemin 1 (Haut)
                self._add_trace(x_drop, y_in, x_start, y_in)
                x_L, y_L, max_L = self._build_circuit(node.left, x_start, y_in, 'H', ways_configs, polarities)

                # Chemin 2 (Bas)
                y_start_R = max_L + 6
                self._add_trace(x_drop, y_in, x_drop, y_start_R)
                self._add_trace(x_drop, y_start_R, x_start, y_start_R)
                x_R, y_R, max_R = self._build_circuit(node.right, x_start, y_start_R, 'H', ways_configs, polarities)

                return max(x_L, x_R), y_in, max_R

            elif has_L != has_R:
                # C'est un bouchon asymétrique (Shunt embarqué)
                main_node = node.left if has_L else node.right
                shunt_node = node.right if has_L else node.left

                x_tap = x_in + 2
                self._add_trace(x_in, y_in, x_tap, y_in)

                # La branche à la masse (tombe à la verticale depuis x_tap)
                xb, yb, max_shunt = self._build_circuit(shunt_node, x_tap, y_in, 'V', ways_configs, polarities)
                self._add_ground(xb, yb)

                # La branche principale s'écarte de la descente (sécurité)
                x_start_main = x_tap + 2
                self._add_trace(x_tap, y_in, x_start_main, y_in)
                x_main, y_main, max_main = self._build_circuit(main_node, x_start_main, y_in, 'H', ways_configs, polarities)

                return x_main, y_main, max(max_main, max_shunt + 1)

            else:
                # Filtre Notch Classique
                if direction == 'H':
                    x_drop = x_in + 2
                    x_start = x_drop + 2
                    self._add_trace(x_in, y_in, x_drop, y_in)

                    self._add_trace(x_drop, y_in, x_start, y_in)
                    x_L, y_L, max_L = self._build_circuit(node.left, x_start, y_in, 'H', ways_configs, polarities)
                    
                    y_start_R = y_in + 6
                    self._add_trace(x_drop, y_in, x_drop, y_start_R)
                    self._add_trace(x_drop, y_start_R, x_start, y_start_R)
                    x_R, y_R, max_R = self._build_circuit(node.right, x_start, y_start_R, 'H', ways_configs, polarities)

                    x_join_drop = max(x_L, x_R) + 2
                    x_join_end = x_join_drop + 2
                    
                    self._add_trace(x_L, y_L, x_join_drop, y_L)
                    self._add_trace(x_R, y_R, x_join_drop, y_R)
                    self._add_trace(x_join_drop, y_L, x_join_drop, y_R)
                    self._add_trace(x_join_drop, y_L, x_join_end, y_L)

                    return x_join_end, y_in, max(max_L, max_R)
                else: # Direction = 'V' (Notch intégré à un Shunt)
                    y_split = y_in + 2
                    self._add_trace(x_in, y_in, x_in, y_split)

                    x_L_start, x_R_start = x_in - 3, x_in + 3
                    self._add_trace(x_in, y_split, x_L_start, y_split)
                    self._add_trace(x_in, y_split, x_R_start, y_split)

                    x_L, y_L, max_L = self._build_circuit(node.left, x_L_start, y_split, 'V', ways_configs, polarities)
                    x_R, y_R, max_R = self._build_circuit(node.right, x_R_start, y_split, 'V', ways_configs, polarities)

                    y_join = max(y_L, y_R) + 2
                    self._add_trace(x_L, y_L, x_L, y_join)
                    self._add_trace(x_R, y_R, x_R, y_join)
                    self._add_trace(x_L, y_join, x_R, y_join)

                    y_end = y_join + 2
                    self._add_trace(x_in, y_join, x_in, y_end)
                    return x_in, y_end, max(max_L, max_R)

        elif isinstance(node, DriverNode):
            # Le Haut-Parleur VituixCAD
            model_name = node.label
            is_inv = False
            for idx, w in enumerate(ways_configs):
                if w.label == node.label:
                    model_name = getattr(w.driver, 'model_name', model_name)
                    if polarities and idx < len(polarities) and polarities[idx] < 0:
                        is_inv = True
                    break

            cx, cy = x_in + 1, y_in + 3
            y_bot = y_in + 6
            self.netlist.append({
                'Type': 'Driver', 'Component': self._get_id('D'),
                'Layout': {'CenX': str(cx), 'CenY': str(cy), 'Rotated': 'False', 'Wires': [(x_in, y_in), (x_in, y_bot)]},
                'Value_Main': 0, 'Model': model_name, 'Inverted': is_inv
            })

            # Masse obligatoire fermant la boucle du Driver
            self._add_ground(x_in, y_bot)

            return x_in + 6, y_bot, y_bot + 2

    def export(self, best_ind, ways_configs):
        # 1. Montage des modèles
        drivers_config = []
        for way in ways_configs:
            drivers_config.append({
                'name': way.driver.model_name if hasattr(way.driver, 'model_name') else way.label,
                'f': os.path.abspath(way.frd_path) if way.frd_path else "",
                'z': os.path.abspath(way.zma_path) if way.zma_path else ""
            })
        self.generator.add_drivers(drivers_config)
        
        # 2. Source Générateur Audio avec sa masse (pôle négatif)
        self.netlist.append({
            'Type': 'Generator', 'Component': 'G1',
            'Layout': {'CenX': '3', 'CenY': '9', 'Rotated': 'False', 'Wires': [(3, 6), (3, 12)]},
            'Value_Main': 2.83
        })
        self._add_ground(3, 12)

        # 3. Lancement du Moteur de Dessin
        polarities = best_ind.get('best_polarities', [])
        self._build_circuit(best_ind['tree'], x_in=3, y_in=6, direction='H', ways_configs=ways_configs, polarities=polarities)
        
        # 4. Compilation finale
        self.generator.generate_schematic(self.netlist)
        self.generator.save()