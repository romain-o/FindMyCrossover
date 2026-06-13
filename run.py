import json
import time
from src.optimizer import CrossoverOptimizer, WayConfig, WEIGHTS as DEFAULT_WEIGHTS
from src.vituix_exporter import VituixAdapter
from src.latex_gen import LatexReportGenerator 
import argparse
import os
import json

def get_driver_paths(data_dir, driver_name, category):
    """Reconstruit le chemin vers le FRD (0deg) et le ZMA depuis le nom."""
    if category == 'W':
        cat_folder = "Woofers"
    elif category == 'M':
        cat_folder = "Midranges"
    else:
        cat_folder = "Tweeters"
        
    frd_0deg = os.path.join(data_dir, cat_folder, "FRD", "0deg", f"{driver_name}_0deg.frd")
    zma = os.path.join(data_dir, cat_folder, "ZMA", f"{driver_name}.zma")
    return frd_0deg, zma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimiseur de Crossover CLI")
    parser.add_argument("--woofer", required=True, help="Nom exact du Woofer (ex: RS150-8)")
    parser.add_argument("--woofer_count", type=int, default=1, help="Nombre de woofers (1 ou 2)")
    parser.add_argument("--midrange", default="",    help="Nom du Médium (optionnel, active le mode 3-voies)")
    parser.add_argument("--tweeter", required=True, help="Nom exact du Tweeter (ex: RST28F-4)")
    parser.add_argument("--data_dir", default="data", help="Dossier racine des données")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gen", type=int, default=150, help="Nombre de générations pour l'optimisation")
    parser.add_argument("--pop", type=int, default=150, help="Taille de la population pour l'optimisation")
    
    parser.add_argument("--fc", type=float, nargs="+", default=None,
                            help="Fréquence(s) de coupure cible. 1 valeur (2-voies) ou 2 valeurs (3-voies). 0 = auto.")
        
    parser.add_argument("--wx", type=float, default=0.0, help="Woofer X offset (gauche/droite en mètres)")
    parser.add_argument("--wy", type=float, default=-0.100, help="Woofer Y offset (haut/bas en mètres)")
    parser.add_argument("--wz", type=float, default=0.0, help="Woofer Z offset (profondeur en mètres)")
    
    parser.add_argument("--wx2", type=float, default=0.0, help="Second Woofer X offset (pour 2 woofers, sinon ignoré)")
    parser.add_argument("--wy2", type=float, default=0.100, help="Second Woofer Y offset (pour 2 woofers, sinon ignoré)")
    parser.add_argument("--wz2", type=float, default=0.0, help="Second Woofer Z offset (pour 2 woofers, sinon ignoré)")
    
    parser.add_argument("--mx", type=float, default=0.0, help="Midrange X offset")
    parser.add_argument("--my", type=float, default=-0.050, help="Midrange Y offset")
    parser.add_argument("--mz", type=float, default=0.0, help="Midrange Z offset")
    
    parser.add_argument("--tx", type=float, default=0.0, help="Tweeter X offset")
    parser.add_argument("--ty", type=float, default=0.0, help="Tweeter Y offset")
    parser.add_argument("--tz", type=float, default=0.0, help="Tweeter Z offset")
    args = parser.parse_args()

    start_time = time.time()
    three_way = bool(args.midrange)
    print(f"\n[{args.name}] Mode : {'3-voies' if three_way else '2-voies'}")
    print(f"\n[{args.name}] Début de la conception du filtre...")

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Construction de target_fc ────────────────────────────────────────────
    if args.fc is None:
        target_fc = None
    elif len(args.fc) == 1:
        target_fc = args.fc[0]               # float  → 2-voies (0.0 = auto)
    else:
        target_fc = tuple(args.fc[:2])       # tuple  → 3-voies (fc_bas, fc_haut)
    
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            app_config = json.load(f)
    except FileNotFoundError:
        print("[!] Fichier config.json introuvable, utilisation des paramètres par défaut.")
        app_config = {}

    w_frd, w_zma = get_driver_paths(args.data_dir, args.woofer, 'W')
    t_frd, t_zma = get_driver_paths(args.data_dir, args.tweeter, 'T')

    positions_woofer = [(args.wx, args.wy, args.wz)]
    if args.woofer_count == 2:
        positions_woofer.append((args.wx2, args.wy2, args.wz2))

    config = [WayConfig("Woofer", w_frd, w_zma, positions=positions_woofer)]

    if three_way:
        m_frd, m_zma = get_driver_paths(args.data_dir, args.midrange, 'M')
        config.append(WayConfig("Midrange", m_frd, m_zma, positions=[(args.mx, args.my, args.mz)]))
    config.append(WayConfig("Tweeter", t_frd, t_zma, positions=[(args.tx, args.ty, args.tz)]))

    # Construction des chemins de sauvegarde
    checkpoint_file = os.path.join(args.out_dir, "checkpoint_evolution.json")
    graph_response_file = os.path.join(args.out_dir, f"{args.name}_Reponse_SPL.png")
    graph_filter_file = os.path.join(args.out_dir, f"{args.name}_Transfert_Elec.png")
    schema_file = os.path.join(args.out_dir, f"{args.name}_Schema.png")
    vituix_file = os.path.join(args.out_dir, f"{args.name}_VituixCAD.vxp")
    graph_directivity_file = os.path.join(args.out_dir, f"{args.name}_Directivity.png")
    dir_heatmap_file = os.path.join(args.out_dir, f"{args.name}_Directivity_Heatmap.png")
    graph_impedance_file = os.path.join(args.out_dir, f"{args.name}_Impedance.png")
    loss_history_file = os.path.join(args.out_dir, f"{args.name}_Loss_History.png")
    part_list_file = os.path.join(args.out_dir, f"{args.name}_Parts_List.csv")
    geometry_file = os.path.join(args.out_dir, f"{args.name}_Geometry.png")
    
    logo_abs_path = "C:/Geekosphere/FindMyCrossover/utils/logo.png"
    # Lancement de l'optimiseur (En passant le checkpoint file)
    opt = CrossoverOptimizer(config, target_fc=args.fc, app_config=app_config)
    best = opt.run(generations=args.gen, pop_size=args.pop, checkpoint_path=checkpoint_file)

    # ==========================================
    # SAUVEGARDE DE TOUS LES FICHIERS
    # ==========================================
    print(f"\n[{args.name}] Génération des fichiers de sortie...")
    
    # 1. Graphiques (SPL + Électrique)
    opt.plot_result(best, filename_response=graph_response_file, filename_filter=graph_filter_file)
    opt.plot_directivity(best, filename=graph_directivity_file)
    opt.plot_sonogram(best, filename=dir_heatmap_file)
    opt.plot_loss_history(filename=loss_history_file)
    opt.plot_impedance(best, filename=graph_impedance_file)
    # 2. Schéma visuel (PNG)
    opt.draw_schematic(best, filename=schema_file)
    opt.generate_parts_list(best, filename=part_list_file)
    opt.plot_geometry(filename=geometry_file)
    
    # Génération LATEX
    part_list_tex = part_list_file.replace(".csv", ".tex")
    # On gère l'affichage correct si c'est un "2x_"
    w_label = f"2x {args.woofer}" if args.woofer_count == 2 else args.woofer
    report_gen = LatexReportGenerator(args.name, args.out_dir, w_label, args.tweeter, logo_abs_path)
    report_gen.generate(part_list_tex)
    # --------------------------------------------------------
    
    # 3. Export VituixCAD (.vxp)
    exporter = VituixAdapter(filename=vituix_file, target_spl=opt.target_spl)
    exporter.export(best_ind=best, ways_configs=config)
    
    metadata_file = os.path.join(args.out_dir, f"{args.name}_metadata.json")
    metadata = {
        "Woofer": w_label,
        "Tweeter": args.tweeter,
        "Midrange":     args.midrange if three_way else None,
        "target_fc":    list(target_fc) if isinstance(target_fc, tuple) else target_fc,
        "app_config":   app_config,
        "positions": {
            "woofer": positions_woofer,
            "midrange": [(args.mx, args.my, args.mz)] if three_way else None,
            "tweeter": [(args.tx, args.ty, args.tz)]
        }
    }
            
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"[{args.name}] ✅ Terminé en {end_time - start_time:.2f} secondes. Fichiers dans : {args.out_dir}")