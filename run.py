import json
import time
from src.optimizer import CrossoverOptimizer, WayConfig, WEIGHTS as DEFAULT_WEIGHTS
from src.vituix_exporter import VituixAdapter
import argparse
import os


def get_driver_paths(data_dir, driver_name, category):
    """Reconstruit les chemins FRD (0deg) et ZMA depuis le nom et la catégorie."""
    folders = {'W': 'Woofers', 'M': 'Midranges', 'T': 'Tweeters'}
    cat_folder = folders.get(category, 'Woofers')
    frd_0deg = os.path.join(data_dir, cat_folder, "FRD", "0deg", f"{driver_name}_0deg.frd")
    zma      = os.path.join(data_dir, cat_folder, "ZMA", f"{driver_name}.zma")
    return frd_0deg, zma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimiseur de Crossover CLI")

    # --- Haut-parleurs ---
    parser.add_argument("--woofer",   required=True, help="Nom exact du Woofer (ex: RS150-8)")
    parser.add_argument("--tweeter",  required=True, help="Nom exact du Tweeter (ex: RST28F-4)")
    parser.add_argument("--midrange", default="",    help="Nom du Médium (optionnel, active le mode 3-voies)")

    # --- Chemins ---
    parser.add_argument("--data_dir", default="data", help="Dossier racine des données")
    parser.add_argument("--out_dir",  required=True)
    parser.add_argument("--name",     required=True)

    # --- Moteur génétique ---
    parser.add_argument("--gen", type=int, default=150, help="Nombre de générations")
    parser.add_argument("--pop", type=int, default=150, help="Taille de la population")

    # --- Fréquence(s) de coupure ---
    # 2-voies : --fc 2500
    # 3-voies : --fc 500 3000
    parser.add_argument("--fc", type=float, nargs="+", default=None,
                        help="Fréquence(s) de coupure cible. 1 valeur (2-voies) ou 2 valeurs (3-voies). 0 = auto.")

    # --- Poids de la fitness (JSON) ---
    parser.add_argument("--weights", type=str, default="",
                        help='JSON des poids fitness (ex: \'{"n_comps": 10, "crossover": 4.0}\')')

    args = parser.parse_args()

    start_time = time.time()
    three_way = bool(args.midrange)
    print(f"\n[{args.name}] Mode : {'3-voies' if three_way else '2-voies'}")
    print(f"[{args.name}] Début de la conception du filtre...")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Construction de target_fc ────────────────────────────────────────────
    if args.fc is None:
        target_fc = None
    elif len(args.fc) == 1:
        target_fc = args.fc[0]               # float  → 2-voies (0.0 = auto)
    else:
        target_fc = tuple(args.fc[:2])       # tuple  → 3-voies (fc_bas, fc_haut)

    # ── Fusion des poids ─────────────────────────────────────────────────────
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        try:
            custom_w = json.loads(args.weights)
            # n_comps doit rester un entier
            if 'n_comps' in custom_w:
                custom_w['n_comps'] = int(custom_w['n_comps'])
            weights.update(custom_w)
            print(f"[+] Poids personnalisés appliqués : {custom_w}")
        except json.JSONDecodeError as e:
            print(f"[-] Erreur parsing --weights JSON : {e}. Poids par défaut utilisés.")

    # ── Construction de la config des voies ──────────────────────────────────
    w_frd, w_zma = get_driver_paths(args.data_dir, args.woofer,  'W')
    t_frd, t_zma = get_driver_paths(args.data_dir, args.tweeter, 'T')

    config = [WayConfig("Woofer", w_frd, w_zma, y_offset=-0.100)]

    if three_way:
        m_frd, m_zma = get_driver_paths(args.data_dir, args.midrange, 'M')
        config.append(WayConfig("Midrange", m_frd, m_zma, y_offset=-0.050))

    config.append(WayConfig("Tweeter", t_frd, t_zma, y_offset=0.0))

    # ── Chemins de sortie ────────────────────────────────────────────────────
    checkpoint_file       = os.path.join(args.out_dir, "checkpoint_evolution.json")
    graph_response_file   = os.path.join(args.out_dir, f"{args.name}_Reponse_SPL.png")
    graph_filter_file     = os.path.join(args.out_dir, f"{args.name}_Transfert_Elec.png")
    schema_file           = os.path.join(args.out_dir, f"{args.name}_Schema.png")
    vituix_file           = os.path.join(args.out_dir, f"{args.name}_VituixCAD.vxp")
    graph_directivity_file = os.path.join(args.out_dir, f"{args.name}_Directivity.png")
    dir_heatmap_file      = os.path.join(args.out_dir, f"{args.name}_Directivity_Heatmap.png")
    graph_impedance_file  = os.path.join(args.out_dir, f"{args.name}_Impedance.png")
    loss_history_file     = os.path.join(args.out_dir, f"{args.name}_Loss_History.png")
    part_list_file        = os.path.join(args.out_dir, f"{args.name}_Parts_List.csv")
    metadata_file         = os.path.join(args.out_dir, f"{args.name}_metadata.json")

    # ── Optimisation ─────────────────────────────────────────────────────────
    opt  = CrossoverOptimizer(config, target_fc=target_fc, weights=weights)
    best = opt.run(generations=args.gen, pop_size=args.pop, checkpoint_path=checkpoint_file)

    # ── Génération des fichiers de sortie ────────────────────────────────────
    print(f"\n[{args.name}] Génération des fichiers de sortie...")

    opt.plot_result(best, filename_response=graph_response_file, filename_filter=graph_filter_file)
    opt.plot_directivity(best, filename=graph_directivity_file)
    opt.plot_sonogram(best, filename=dir_heatmap_file)
    opt.plot_loss_history(filename=loss_history_file)
    opt.plot_impedance(best, filename=graph_impedance_file)
    opt.draw_schematic(best, filename=schema_file)
    opt.export_bom_csv(best, filename=part_list_file)

    exporter = VituixAdapter(filename=vituix_file, target_spl=opt.target_spl)
    exporter.export(best_ind=best, ways_configs=config)

    metadata = {
        "Project_Name": args.name,
        "Woofer":       args.woofer,
        "Midrange":     args.midrange if three_way else None,
        "Tweeter":      args.tweeter,
        "target_fc":    list(target_fc) if isinstance(target_fc, tuple) else target_fc,
        "weights":      weights,
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"[{args.name}] ✅ Terminé en {end_time - start_time:.2f} s. Fichiers dans : {args.out_dir}")
