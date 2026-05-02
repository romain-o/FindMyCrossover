import time
from src.optimizer import CrossoverOptimizer, WayConfig
from src.vituix_exporter import VituixAdapter
import argparse
import os

def get_driver_paths(data_dir, driver_name, category):
    """Reconstruit le chemin vers le FRD (0deg) et le ZMA depuis le nom."""
    cat_folder = "Woofers" if category == 'W' else "Tweeters"
    frd_0deg = os.path.join(data_dir, cat_folder, "FRD", "0deg", f"{driver_name}_0deg.frd")
    zma = os.path.join(data_dir, cat_folder, "ZMA", f"{driver_name}.zma")
    return frd_0deg, zma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimiseur de Crossover CLI")
    parser.add_argument("--woofer", required=True, help="Nom exact du Woofer (ex: RS150-8)")
    parser.add_argument("--tweeter", required=True, help="Nom exact du Tweeter (ex: RST28F-4)")
    parser.add_argument("--data_dir", default="data", help="Dossier racine des données")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gen", type=int, default=150, help="Nombre de générations pour l'optimisation")
    parser.add_argument("--pop", type=int, default=150, help="Taille de la population pour l'optimisation")
    
    parser.add_argument("--fc", type=float, default=0.0, help="Fréquence de coupure cible (0 = Auto)")
    
    args = parser.parse_args()

    start_time = time.time()
    print(f"\n[{args.name}] Début de la conception du filtre...")

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(args.out_dir, exist_ok=True)

    w_frd, w_zma = get_driver_paths(args.data_dir, args.woofer, 'W')
    t_frd, t_zma = get_driver_paths(args.data_dir, args.tweeter, 'T')

    # Configuration des voies avec les chemins dynamiques
    config = [
        WayConfig("Woofer", w_frd, w_zma, 
                  z_offset=0, y_offset=-0.100, x_offset=0),
        WayConfig("Tweeter", t_frd, t_zma, 
                  z_offset=0, y_offset=0, x_offset=0) # Tweeter aligné avec le micro
    ]

    # Construction des chemins de sauvegarde
    checkpoint_file = os.path.join(args.out_dir, "checkpoint_evolution.json")
    graph_response_file = os.path.join(args.out_dir, f"{args.name}_Reponse_SPL.png")
    graph_filter_file = os.path.join(args.out_dir, f"{args.name}_Transfert_Elec.png")
    schema_file = os.path.join(args.out_dir, f"{args.name}_Schema.png")
    vituix_file = os.path.join(args.out_dir, f"{args.name}_VituixCAD.vxp")
    graph_directivity_file = os.path.join(args.out_dir, f"{args.name}_Directivity.png")
    dir_heatmap_file = os.path.join(args.out_dir, f"{args.name}_Directivity_Heatmap.png")
    # Lancement de l'optimiseur (En passant le checkpoint file)
    opt = CrossoverOptimizer(config, target_fc=args.fc)
    best = opt.run(generations=args.gen, pop_size=args.pop, checkpoint_path=checkpoint_file)

    # ==========================================
    # SAUVEGARDE DE TOUS LES FICHIERS
    # ==========================================
    print(f"\n[{args.name}] Génération des fichiers de sortie...")
    
    # 1. Graphiques (SPL + Électrique)
    opt.plot_result(best, filename_response=graph_response_file, filename_filter=graph_filter_file)
    opt.plot_directivity(best, filename=graph_directivity_file)
    opt.plot_sonogram(best, filename=dir_heatmap_file)
    # 2. Schéma visuel (PNG)
    opt.draw_schematic(best, filename=schema_file)
    
    # 3. Export VituixCAD (.vxp)
    exporter = VituixAdapter(filename=vituix_file, target_spl=opt.target_spl)
    exporter.export(best_ind=best, ways_configs=config)

    end_time = time.time()
    print(f"[{args.name}] ✅ Terminé en {end_time - start_time:.2f} secondes. Fichiers dans : {args.out_dir}")