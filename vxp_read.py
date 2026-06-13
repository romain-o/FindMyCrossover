import os
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET

class UnionFind:
    """Structure de données pour regrouper les coordonnées reliées par des fils."""
    def __init__(self):
        self.parent = {}
    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def parse_vituix(vxp_path, name):
    # 1. Chargement des métadonnées pour faire correspondre le Modèle au Label (Woofer/Tweeter)
    metadata_file = os.path.join("crossovers", name, f"{name}_metadata.json")
    model_to_label = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            if meta.get("Woofer"):
                model_to_label[meta["Woofer"].replace("2x ", "")] = "Woofer"
            if meta.get("Tweeter"):
                model_to_label[meta["Tweeter"]] = "Tweeter"
            if meta.get("Midrange"):
                model_to_label[meta["Midrange"]] = "Midrange"

    tree = ET.parse(vxp_path)
    root = tree.getroot()

    # Fallback si pas de metadata : on devine selon l'ordre d'apparition
    if not model_to_label:
        drivers = root.findall('DRIVER')
        unique_models = []
        for d in drivers:
            m = d.find('Model').text
            if m not in unique_models: unique_models.append(m)
        if len(unique_models) >= 1: model_to_label[unique_models[0]] = "Woofer"
        if len(unique_models) >= 2: model_to_label[unique_models[-1]] = "Tweeter"
        if len(unique_models) == 3: model_to_label[unique_models[1]] = "Midrange"

    crossover = root.find('CROSSOVER')
    if crossover is None:
        print("[-] Erreur : Aucun filtre CROSSOVER trouvé dans le fichier VituixCAD.")
        return None
        
    parts = crossover.findall('PART')

    # 2. Construction des nœuds électriques (Union-Find sur les fils)
    uf = UnionFind()
    for part in parts:
        ptype = part.find('Type').text
        wires = part.findall('WIRE')
        if not wires: continue
        coords = [(int(w.find('X').text), int(w.find('Y').text)) for w in wires]

        if ptype == 'Wire':
            uf.union(coords[0], coords[1])
        elif ptype == 'Ground':
            uf.union(coords[0], 'GROUND')

    ground = uf.find('GROUND')
    in_plus = None
    
    # Trouver l'entrée positive depuis le Générateur
    for part in parts:
        if part.find('Type').text == 'Generator':
            wires = part.findall('WIRE')
            coords = [(int(w.find('X').text), int(w.find('Y').text)) for w in wires]
            if uf.find(coords[0]) == ground:
                in_plus = uf.find(coords[1])
            else:
                in_plus = uf.find(coords[0])
            break

    # 3. Extraction des composants physiques
    edges = []
    edge_id = 0
    for part in parts:
        ptype = part.find('Type').text
        if ptype in ['Resistor', 'Capacitor', 'Inductor', 'Driver']:
            wires = part.findall('WIRE')
            c1 = (int(wires[0].find('X').text), int(wires[0].find('Y').text))
            c2 = (int(wires[1].find('X').text), int(wires[1].find('Y').text))
            n1 = uf.find(c1)
            n2 = uf.find(c2)

            if n1 == n2:
                continue # Ignore les courts-circuits

            comp_dict = None
            if ptype == 'Resistor':
                val = float(next(p.find('Value').text for p in part.findall('PARAM') if p.find('Name').text == 'R'))
                comp_dict = {"type": "Resistor", "value": val}
            elif ptype == 'Capacitor':
                val = float(next(p.find('Value').text for p in part.findall('PARAM') if p.find('Name').text == 'C'))
                comp_dict = {"type": "Capacitor", "value": val * 1e-6} # Convertit uF en Farads
            elif ptype == 'Inductor':
                val = float(next(p.find('Value').text for p in part.findall('PARAM') if p.find('Name').text == 'L'))
                comp_dict = {"type": "Inductor", "value": val * 1e-3} # Convertit mH en Henrys
            elif ptype == 'Driver':
                model = part.find('Model').text
                label = model_to_label.get(model, "Unknown")
                comp_dict = {"type": "DriverNode", "label": label}

            if comp_dict:
                edges.append({'id': edge_id, 'comp': comp_dict, 'n1': n1, 'n2': n2})
                edge_id += 1

    # 4. Réduction Mathématique (Graphe 2D -> Arbre Série/Parallèle)
    changed = True
    while changed and len(edges) > 1:
        changed = False
        # Réduction Parallèle
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                e1, e2 = edges[i], edges[j]
                if (e1['n1'] == e2['n1'] and e1['n2'] == e2['n2']) or \
                   (e1['n1'] == e2['n2'] and e1['n2'] == e2['n1']):
                    new_edge = {
                        'id': f"P_{e1['id']}_{e2['id']}",
                        'comp': {'type': 'ParallelNode', 'left': e1['comp'], 'right': e2['comp']},
                        'n1': e1['n1'], 'n2': e1['n2']
                    }
                    edges.remove(e2)
                    edges.remove(e1)
                    edges.append(new_edge)
                    changed = True
                    break
            if changed: break
        if changed: continue

        # Réduction Série
        node_degrees = {}
        for e in edges:
            node_degrees[e['n1']] = node_degrees.get(e['n1'], 0) + 1
            node_degrees[e['n2']] = node_degrees.get(e['n2'], 0) + 1

        for n, deg in node_degrees.items():
            if deg == 2 and n != in_plus and n != ground:
                adj_edges = [e for e in edges if e['n1'] == n or e['n2'] == n]
                if len(adj_edges) == 2:
                    e1, e2 = adj_edges[0], adj_edges[1]
                    n_other1 = e1['n2'] if e1['n1'] == n else e1['n1']
                    n_other2 = e2['n2'] if e2['n1'] == n else e2['n1']

                    new_edge = {
                        'id': f"S_{e1['id']}_{e2['id']}",
                        'comp': {'type': 'SeriesNode', 'left': e1['comp'], 'right': e2['comp']},
                        'n1': n_other1, 'n2': n_other2
                    }
                    edges.remove(e2)
                    edges.remove(e1)
                    edges.append(new_edge)
                    changed = True
                    break
        if changed: continue

    if not edges:
        print("[-] Erreur: Aucun composant trouvé ou schéma invalide.")
        return None

    raw_ast = edges[0]['comp']
    if len(edges) > 1:
        print(f"[!] Attention : {len(edges)} branches n'ont pas pu être réduites parfaitement. Le schéma Vituix a peut-être un croisement complexe.")
        # Sécurité : on les force en parallèle pour ne pas crasher
        for e in edges[1:]:
            raw_ast = {'type': 'ParallelNode', 'left': raw_ast, 'right': e['comp']}

    # 5. Rétro-ingénierie du Multi-Driver (Annule le dessin visuel pour retrouver count=2)
    wiring_dict = {}
    def collapse_drivers(node):
        if node['type'] == 'DriverNode':
            return node, node['label']
        if node['type'] in ['SeriesNode', 'ParallelNode']:
            left, l_label = collapse_drivers(node['left'])
            right, r_label = collapse_drivers(node['right'])
            node['left'] = left
            node['right'] = right
            
            # Si on trouve deux woofers connectés entre eux, on les fusionne en un seul !
            if l_label and r_label and l_label == r_label:
                wiring_dict[l_label] = 'series' if node['type'] == 'SeriesNode' else 'parallel'
                return left, l_label 
            else:
                return node, None
        return node, None

    final_ast, _ = collapse_drivers(raw_ast)
    return {"tree": final_ast, "wiring": wiring_dict}

def main():
    parser = argparse.ArgumentParser(description="Importer un schéma VituixCAD vers Python")
    parser.add_argument("--vxp", required=True, help="Chemin du fichier VituixCAD (.vxp)")
    parser.add_argument("--name", required=True, help="Nom exact du projet (ex: 2x_12SW-4HE_X_AMT2-4)")
    args = parser.parse_args()

    print(f"\n[*] Lecture et analyse du schéma VituixCAD : {args.vxp}")
    data = parse_vituix(args.vxp, args.name)

    if not data:
        print("[-] Echec de l'extraction.")
        return

    out_dir = os.path.join("crossovers", args.name)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_file = os.path.join(out_dir, "checkpoint_evolution.json")

    print(f"[+] Traduction réussie ! Sauvegarde de l'arbre dans : {checkpoint_file}")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print("[*] Déclenchement de plot.py pour mettre à jour les graphiques...")
    subprocess.run(["python", "plot.py", "--name", args.name])

if __name__ == "__main__":
    main()