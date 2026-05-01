import time
from src.optimizer import CrossoverOptimizer, WayConfig
from src.vituix_exporter import VituixAdapter
import argparse
import os

if __name__ == "__main__":
    # Configuration des arguments pour la ligne de commande
    parser = argparse.ArgumentParser(description="Optimiseur de Crossover")
    parser.add_argument("--woofer_frd", required=True, help="Chemin du FRD Woofer")
    parser.add_argument("--woofer_zma", required=True, help="Chemin du ZMA Woofer")
    parser.add_argument("--tweeter_frd", required=True, help="Chemin du FRD Tweeter")
    parser.add_argument("--tweeter_zma", required=True, help="Chemin du ZMA Tweeter")
    parser.add_argument("--out_dir", required=True, help="Dossier de sauvegarde des résultats")
    parser.add_argument("--name", required=True, help="Nom de base pour les fichiers de sortie")
    
    args = parser.parse_args()

    print(f"\n--- Démarrage Optimisation : {args.name} ---")

    # 1. Configuration des voies avec les arguments
    config = [
        WayConfig("Woofer", args.woofer_frd, args.woofer_zma, z_offset_m=0.03),
        WayConfig("Tweeter", args.tweeter_frd, args.tweeter_zma, z_offset_m=0.0)
    ]

    # 2. Lancement de l'optimiseur
    opt = CrossoverOptimizer(config, crossover_freq=2000, target_spl=85.0)
    best = opt.run(generations=150, population_size=150) # Ajustez selon votre puissance de calcul

    print(f"\n[*] Optimisation terminée. Score final: {best['score']:.2f}")

    # 3. Sauvegarde dans le dossier de sortie
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Chemins de sortie
    graph_path = os.path.join(args.out_dir, f"{args.name}_Graph.png")
    schema_path = os.path.join(args.out_dir, f"{args.name}_Schema.png")
    vituix_path = os.path.join(args.out_dir, f"{args.name}_VituixCAD.vxp")
    
    # Export des graphiques et schémas
    opt.plot_result(best['tree'], filename=graph_path)
    opt.draw_schematic(best['tree'], filename=schema_path)
    
    # Export VituixCAD
    exporter = VituixAdapter(filename=vituix_path, target_spl=opt.target_spl)
    exporter.export(best_tree=best['tree'], ways_configs=config)

    print(f"[+] Succès : Fichiers enregistrés dans {args.out_dir}")