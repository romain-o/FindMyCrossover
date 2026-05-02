import time
from src.optimizer import CrossoverOptimizer, WayConfig
from src.vituix_exporter import VituixAdapter
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimiseur de Crossover CLI")
    parser.add_argument("--woofer_frd", required=True)
    parser.add_argument("--woofer_zma", required=True)
    parser.add_argument("--tweeter_frd", required=True)
    parser.add_argument("--tweeter_zma", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gen", type=int, default=150, help="Nombre de générations pour l'optimisation")
    parser.add_argument("--pop", type=int, default=150, help="Taille de la population pour l'optimisation")
    args = parser.parse_args()

    start_time = time.time()
    print(f"\n[{args.name}] Début de la conception du filtre...")

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(args.out_dir, exist_ok=True)

    # Configuration des voies avec les chemins dynamiques
    config = [
        WayConfig("Woofer", args.woofer_frd, args.woofer_zma, 
                  z_offset=0, y_offset=-0.100, x_offset=0),
        WayConfig("Tweeter", args.tweeter_frd, args.tweeter_zma, 
                  z_offset=0, y_offset=0, x_offset=0) # Tweeter aligné avec le micro
    ]

    # Construction des chemins de sauvegarde
    checkpoint_file = os.path.join(args.out_dir, "checkpoint_evolution.json")
    graph_response_file = os.path.join(args.out_dir, f"{args.name}_Reponse_SPL.png")
    graph_filter_file = os.path.join(args.out_dir, f"{args.name}_Transfert_Elec.png")
    schema_file = os.path.join(args.out_dir, f"{args.name}_Schema.png")
    vituix_file = os.path.join(args.out_dir, f"{args.name}_VituixCAD.vxp")
    graph_directivity_file = os.path.join(args.out_dir, f"{args.name}_Directivity.png")
    # Lancement de l'optimiseur (En passant le checkpoint file)
    opt = CrossoverOptimizer(config)
    best = opt.run(generations=args.gen, pop_size=args.pop, checkpoint_path=checkpoint_file)

    # ==========================================
    # SAUVEGARDE DE TOUS LES FICHIERS
    # ==========================================
    print(f"\n[{args.name}] Génération des fichiers de sortie...")
    
    # 1. Graphiques (SPL + Électrique)
    opt.plot_result(best, filename_response=graph_response_file, filename_filter=graph_filter_file)
    opt.plot_directivity(best, filename=graph_directivity_file)
    # 2. Schéma visuel (PNG)
    opt.draw_schematic(best, filename=schema_file)
    
    # 3. Export VituixCAD (.vxp)
    exporter = VituixAdapter(filename=vituix_file, target_spl=opt.target_spl)
    exporter.export(best_ind=best, ways_configs=config)

    end_time = time.time()
    print(f"[{args.name}] ✅ Terminé en {end_time - start_time:.2f} secondes. Fichiers dans : {args.out_dir}")