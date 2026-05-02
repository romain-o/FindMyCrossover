import pandas as pd
import os
import shutil
import subprocess
import json
from pathlib import Path

# --- CONFIGURATION ---
N_CROSSOVERS_A_TRAITER = 2  # Combien de crossovers voulez-vous générer en une fois ?
DOSSIER_DATA = r"data" # Le dossier propre que nous avons créé
DOSSIER_SORTIE = r"crossovers" # Là où iront les PDF/PNG/VXP
FICHIER_CSV = r"data\W_T_pairs.csv"

def get_driver_files(driver_name, category):
    """Trouve les bons chemins FRD (0deg) et ZMA pour un driver."""
    cat_folder = "Woofers" if category == 'W' else "Tweeters"
    frd_path = os.path.join(DOSSIER_DATA, cat_folder, "FRD", "0deg", f"{driver_name}_0deg.frd")
    zma_path = os.path.join(DOSSIER_DATA, cat_folder, "ZMA", f"{driver_name}.zma")
    
    return frd_path, zma_path

def process_batch():
    # 1. Vérifier si les dossiers existent
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    # 2. Charger la liste des paires triées par priorité
    try:
        df = pd.read_csv(FICHIER_CSV)
    except FileNotFoundError:
        print(f"[-] Erreur : Impossible de trouver le fichier '{FICHIER_CSV}'")
        return

    crossovers_traites = 0

    print(f"[*] Démarrage du Traitement par Lots (Objectif : {N_CROSSOVERS_A_TRAITER} crossovers)")
    print("-" * 50)

    # 3. Parcourir la liste du haut vers le bas
    for index, row in df.iterrows():
        if crossovers_traites >= N_CROSSOVERS_A_TRAITER:
            break # Objectif atteint !

        crossovers_traites += 1
        
        woofer = row['Woofer (W)']
        tweeter = row['Tweeter (T)']
        nom_projet = f"{woofer}_X_{tweeter}"
        
        # 4. Vérifier si ce projet a déjà été calculé !
        # On vérifie si un dossier à son nom existe déjà dans Crossovers_Finis
        dossier_projet = os.path.join(DOSSIER_SORTIE, nom_projet)
        if os.path.exists(dossier_projet):
            print(f"[Skip] Le projet {nom_projet} a déjà été calculé. Passage au suivant.")
            continue
            
        print(f"\n[>>>] Préparation du projet #{crossovers_traites + 1} : {nom_projet}")
        
        # 5. Récupérer les chemins des fichiers
        w_frd, w_zma = get_driver_files(woofer, 'W')
        t_frd, t_zma = get_driver_files(tweeter, 'T')
        
        # Vérification d'intégrité (Est-ce qu'on a bien extrait ces fichiers ?)
        missing = [f for f in [w_frd, w_zma, t_frd, t_zma] if not os.path.exists(f)]
        if missing:
            print(f"[-] Impossible de lancer {nom_projet}. Fichiers de données introuvables :")
            for m in missing: print(f"    - {m}")
            continue

        # 6. Appel de run.py en utilisant subprocess
        # subprocess.run permet d'exécuter une ligne de commande "python run.py ..."
        cmd = [
            "python", "run.py",
            "--woofer_frd", w_frd,
            "--woofer_zma", w_zma,
            "--tweeter_frd", t_frd,
            "--tweeter_zma", t_zma,
            "--out_dir", dossier_projet,
            "--name", nom_projet,
            "--gen", "10", 
            "--pop", "100"
        ]
        
        print(f"[*] Lancement de l'algorithme génétique...")
        try:
            # Exécution de run.py. L'argument capture_output=False permet de voir les prints de run.py dans la console.
            result = subprocess.run(cmd, check=True)
            
            # 7. Si l'optimisation réussit, on génère le fichier JSON des métadonnées
            metadata = {
                "Project_Name": nom_projet,
                "Woofer": woofer,
                "Tweeter": tweeter,
                "Market_Tier": row['Gamme'],
                "Category": row['Catégorie'],
                "Total_Cost_USD": float(row['Prix Total ($)']),
                "Popularity_Score": int(row['Score Popularité'])
            }
            
            json_path = os.path.join(dossier_projet, "metadata.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            print(f"[+] Projet {nom_projet} terminé avec succès !")
            
        except subprocess.CalledProcessError as e:
            print(f"[-] L'optimisation a échoué (Crash) pour {nom_projet}. Code d'erreur : {e.returncode}")
            # Si ça a planté, on supprime le dossier créé pour qu'il réessaie la prochaine fois
            if os.path.exists(dossier_projet):
                shutil.rmtree(dossier_projet)

    print("\n" + "=" * 50)
    print(f"Mission accomplie. {crossovers_traites} nouveaux crossovers générés.")

if __name__ == "__main__":
    process_batch()