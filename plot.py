import json
import os
import argparse
from src.optimizer import CrossoverOptimizer, WayConfig
from src.latex_gen import LatexReportGenerator
from src.nodes import Node, ComponentNode
from src.catalog_manager import CatalogManager
from src.vituix_exporter import VituixAdapter

def get_driver_paths(data_dir, driver_name, category):
    """Reconstruit le chemin vers le FRD et le ZMA."""
    if category == 'W':
        cat_folder = "Woofers"
    elif category == 'M':
        cat_folder = "Midranges"
    else:
        cat_folder = "Tweeters"
        
    frd_0deg = os.path.join(data_dir, cat_folder, "FRD", "0deg", f"{driver_name}_0deg.frd")
    zma = os.path.join(data_dir, cat_folder, "ZMA", f"{driver_name}.zma")
    return frd_0deg, zma

def main():
    parser = argparse.ArgumentParser(description="Générer les graphiques depuis une sauvegarde JSON")
    parser.add_argument("--name", required=True, help="Nom exact du projet (ex: 2x_BC-6NDL38_X_BMS-4526ND)")
    parser.add_argument("--data_dir", default="data", help="Dossier racine des données")
    args = parser.parse_args()

    out_dir = os.path.join("crossovers", args.name)
    checkpoint_file = os.path.join(out_dir, "checkpoint_evolution.json")
    metadata_file = os.path.join(out_dir, f"{args.name}_metadata.json")

    if not os.path.exists(checkpoint_file):
        print(f"[-] Erreur : Le fichier {checkpoint_file} est introuvable.")
        return

    # ==========================================
    # 1. RÉCUPÉRATION DES HAUT-PARLEURS & POSITIONS
    # ==========================================
    woofer_name, midrange_name, tweeter_name = None, None, None
    
    # Listes de positions par défaut
    positions_woofer = [(0.0, -0.100, 0.0)]
    positions_mid = [(0.0, -0.050, 0.0)]
    positions_twt = [(0.0, 0.0, 0.0)]
    
    # On lit le metadata.json créé par run.py
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
            # Noms des haut-parleurs
            woofer_name = meta.get("Woofer")
            if woofer_name and woofer_name.startswith("2x "):
                woofer_name = woofer_name.replace("2x ", "")
                
            tweeter_name = meta.get("Tweeter")
            midrange_name = meta.get("Midrange")
            
            # --- NOUVELLE LOGIQUE DE LECTURE DES POSITIONS ---
            if "positions" in meta:
                pos_data = meta["positions"]
                if pos_data.get("woofer"):
                    positions_woofer = [tuple(p) for p in pos_data["woofer"]]
                if pos_data.get("midrange"):
                    positions_mid = [tuple(p) for p in pos_data["midrange"]]
                if pos_data.get("tweeter"):
                    positions_twt = [tuple(p) for p in pos_data["tweeter"]]
            else:
                # Fallback : Compatibilité avec vos anciennes sauvegardes (wx, wy, etc.)
                wx = meta.get("wx", 0.0)
                wy = meta.get("wy", -0.100)
                wz = meta.get("wz", 0.0)
                wx2 = meta.get("wx2", 0.0)
                wy2 = meta.get("wy2", -0.200)
                wz2 = meta.get("wz2", 0.0)
                mx = meta.get("mx", 0.0)
                my = meta.get("my", -0.050)
                mz = meta.get("mz", 0.0)
                tx = meta.get("tx", 0.0)
                ty = meta.get("ty", 0.0)
                tz = meta.get("tz", 0.0)
                
                positions_woofer = [(wx, wy, wz)]
                if meta.get("woofer_count", 1) == 2 or ("2x_" in args.name):
                    positions_woofer.append((wx2, wy2, wz2))
                positions_mid = [(mx, my, mz)]
                positions_twt = [(tx, ty, tz)]
            
    # Fallback si le metadata.json a été supprimé (on devine via le nom du dossier)
    if not woofer_name or not tweeter_name:
        parts = args.name.replace("2x_", "").split("_X_")
        if len(parts) >= 3:
            woofer_name = parts[0]
            midrange_name = parts[1]
            tweeter_name = parts[2]
        elif len(parts) == 2:
            woofer_name = parts[0]
            tweeter_name = parts[1]
        else:
            print("[-] Impossible d'identifier les haut-parleurs. Vérifiez le nom ou metadata.json.")
            return

    w_count = len(positions_woofer)

    print(f"\n[+] Chargement du projet : {args.name}")
    print(f"    - Woofer  : {w_count}x {woofer_name}")
    if midrange_name:
        print(f"    - Médium  : {midrange_name}")
    print(f"    - Tweeter : {tweeter_name}")

    w_frd, w_zma = get_driver_paths(args.data_dir, woofer_name, 'W')
    t_frd, t_zma = get_driver_paths(args.data_dir, tweeter_name, 'T')

    # Configuration des voies avec les listes de tuples extraites
    config = [
        WayConfig("Woofer", w_frd, w_zma, positions=positions_woofer)
    ]
    
    if midrange_name:
        m_frd, m_zma = get_driver_paths(args.data_dir, midrange_name, 'M')
        config.append(WayConfig("Midrange", m_frd, m_zma, positions=positions_mid))
        
    config.append(WayConfig("Tweeter", t_frd, t_zma, positions=positions_twt))

    # ==========================================
    # 2. RECONSTRUCTION DE L'ENVIRONNEMENT
    # ==========================================
    opt = CrossoverOptimizer(config)

    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
        
    tree = Node.from_dict(data["tree"])
    wiring = data.get("wiring", {}) 
    
    # Liaison des DriverNodes "virtuels" aux vraies courbes de l'Optimizer
    for n in tree.get_all_nodes():
        if type(n).__name__ == "DriverNode":
            way = next(w for w in opt.ways if w.label == n.label)
            n.H_acoustic = way.driver.H_acoustic
            n.Z_complex = way.driver.Z_complex
            n.model_name = way.driver.model_name
            n.H_base = way.driver.H_base
            n.Z_base = way.driver.Z_base

    individual = {
        'tree': tree,
        'wiring': wiring
    }

    # Calcul correct du SPL cible (Correction du bug de retour de apply_wiring)
    opt.apply_wiring(wiring)
    individual['target_spl'] = opt.target_spl

    # ==========================================
    # 3. GÉNÉRATION DES GRAPHIQUES
    # ==========================================
    print("[+] Génération des visuels en cours...")
    
    opt.plot_result(individual, 
                    filename_response=os.path.join(out_dir, f"{args.name}_Reponse_SPL.png"), 
                    filename_filter=os.path.join(out_dir, f"{args.name}_Transfert_Elec.png"))
    
    opt.plot_directivity(individual, filename=os.path.join(out_dir, f"{args.name}_Directivity.png"))
    opt.plot_sonogram(individual, filename=os.path.join(out_dir, f"{args.name}_Directivity_Heatmap.png"))
    opt.plot_impedance(individual, filename=os.path.join(out_dir, f"{args.name}_Impedance.png"))
    opt.draw_schematic(individual, filename=os.path.join(out_dir, f"{args.name}_Schema.png"))
    opt.generate_parts_list(individual, filename=os.path.join(out_dir, f"{args.name}_Parts_List.csv"))
    opt.plot_geometry(filename=os.path.join(out_dir, f"{args.name}_Geometry.png"))
    
    part_list_tex = os.path.join(out_dir, f"{args.name}_Parts_List.tex")
    logo_abs_path = "C:/Geekosphere/FindMyCrossover/utils/logo.png"
    w_label = f"{w_count}x {woofer_name}" if w_count > 1 else woofer_name
    
    report_gen = LatexReportGenerator(args.name, out_dir, w_label, tweeter_name, logo_abs_path)
    report_gen.generate(part_list_tex)

    # 3. Export VituixCAD (.vxp)
    exporter = VituixAdapter(filename=os.path.join(out_dir, f"{args.name}_VituixCAD.vxp"), target_spl=opt.target_spl)
    exporter.export(best_ind=individual, ways_configs=config)
       
    print(f"[+] ✅ Terminé ! Tous les graphiques du dossier '{args.name}' ont été rafraîchis aux valeurs réelles.")

if __name__ == "__main__":
    main()