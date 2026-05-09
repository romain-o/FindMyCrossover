import json
import os
import argparse
from src.optimizer import CrossoverOptimizer, WayConfig
from src.latex_gen import LatexReportGenerator
from src.nodes import Node

def get_driver_paths(data_dir, driver_name, category):
    """Reconstruit le chemin vers le FRD et le ZMA."""
    cat_folder = "Woofers" if category == 'W' else "Tweeters"
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
    # 1. RÉCUPÉRATION DES HAUT-PARLEURS
    # ==========================================
    woofer_name, tweeter_name = None, None
    
    # On lit le metadata.json créé par run.py
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            woofer_name = meta.get("Woofer")
            tweeter_name = meta.get("Tweeter")
            wx = meta.get("wx", 0.0)
            wy = meta.get("wy", -0.100)
            wz = meta.get("wz", 0.0)
            
    # Fallback si le metadata.json a été supprimé (on devine via le nom du dossier)
    if not woofer_name or not tweeter_name:
        parts = args.name.split("_X_")
        if len(parts) >= 2:
            woofer_name = parts[0].replace("2x_", "")
            tweeter_name = parts[1]
        else:
            print("[-] Impossible d'identifier les haut-parleurs. Vérifiez le nom ou metadata.json.")
            return

    # Détection automatique du nombre de woofers
    w_count = 2 if "2x_" in args.name else 1

    print(f"\n[+] Chargement du projet : {args.name}")
    print(f"    - Woofer  : {w_count}x {woofer_name}")
    print(f"    - Tweeter : {tweeter_name}")

    w_frd, w_zma = get_driver_paths(args.data_dir, woofer_name, 'W')
    t_frd, t_zma = get_driver_paths(args.data_dir, tweeter_name, 'T')

    # Configuration des voies avec la géométrie par défaut
    config = [
        WayConfig("Woofer", w_frd, w_zma, count=w_count, z_offset=0, y_offset=-0.100, x_offset=0),
        WayConfig("Tweeter", t_frd, t_zma, z_offset=0, y_offset=0, x_offset=0)
    ]

    # ==========================================
    # 2. RECONSTRUCTION DE L'ENVIRONNEMENT
    # ==========================================
    # On initialise l'optimizer UNIQUEMENT pour charger les outils (evaluator, freqs, ways)
    opt = CrossoverOptimizer(config)

    # Chargement de l'arbre sauvegardé
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
        
    tree = Node.from_dict(data["tree"])
    wiring = data.get("wiring", {}) 
    
    # ÉTAPE CRUCIALE : Relier les nœuds "texte" du JSON aux vraies courbes de l'Optimizer
    for n in tree.get_all_nodes():
        if type(n).__name__ == "DriverNode":
            way = next(w for w in opt.ways if w.label == n.label)
            n.H_acoustic = way.driver.H_acoustic
            n.Z_complex = way.driver.Z_complex
            n.model_name = way.driver.model_name
            # Nécessaire pour la simulation série/parallèle
            n.H_base = way.driver.H_base
            n.Z_base = way.driver.Z_base

    individual = {
        'tree': tree,
        'wiring': wiring
    }

    # On recalcule le Target SPL correct selon le câblage (série ou parallèle)
    individual['target_spl'] = opt.apply_wiring(wiring)

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
    
    part_list_tex = os.path.join(out_dir, f"{args.name}_Parts_List.tex")
    logo_abs_path = "C:/Geekosphere/FindMyCrossover/utils/logo.png"
    # On formate joliment les noms de HP
    w_label = f"{w_count}x {woofer_name}" if w_count > 1 else woofer_name
    
    report_gen = LatexReportGenerator(args.name, out_dir, w_label, tweeter_name, logo_abs_path)
    report_gen.generate(part_list_tex)
    
    # Note: plot_loss_history() n'est pas appelé car l'historique d'entraînement
    # n'est pas conservé dans le checkpoint (il n'existe que pendant l'optimisation).

    print(f"[+] ✅ Terminé ! Tous les graphiques du dossier '{args.name}' ont été rafraîchis.")

if __name__ == "__main__":
    main()