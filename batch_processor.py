import pandas as pd
import os
import shutil
import subprocess
import json

# --- CONFIGURATION ---
N_CROSSOVERS_A_TRAITER = 2
DOSSIER_DATA = r"data" 
DOSSIER_SORTIE = r"crossovers"
FICHIER_CSV = r"data\W_T_pairs.csv"

def check_all_files_exist(driver_name, category):
    """Vérifie que le ZMA et TOUS les FRD (0, 15, 30, 45) existent pour un HP."""
    cat_folder = "Woofers" if category == 'W' else "Tweeters"
    missing = []
    
    # ZMA
    zma_path = os.path.join(DOSSIER_DATA, cat_folder, "ZMA", f"{driver_name}.zma")
    if not os.path.exists(zma_path): missing.append(zma_path)
        
    # FRD Multi-angles
    for angle in ['0deg', '15deg', '30deg', '45deg']:
        frd_path = os.path.join(DOSSIER_DATA, cat_folder, "FRD", angle, f"{driver_name}_{angle}.frd")
        if not os.path.exists(frd_path): missing.append(frd_path)
            
    return missing

def process_batch():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    
    try:
        df = pd.read_csv(FICHIER_CSV)
    except FileNotFoundError:
        print(f"[-] Erreur : Impossible de trouver le fichier '{FICHIER_CSV}'")
        return

    crossovers_traites = 0
    print(f"[*] Démarrage du Traitement par Lots (Objectif : {N_CROSSOVERS_A_TRAITER} crossovers)")
    print("-" * 50)

    for index, row in df.iterrows():
        if crossovers_traites >= N_CROSSOVERS_A_TRAITER:
            break

        woofer = row['Woofer (W)']
        tweeter = row['Tweeter (T)']
        nom_projet = f"{woofer}_X_{tweeter}"
        
        dossier_projet = os.path.join(DOSSIER_SORTIE, nom_projet)
        if os.path.exists(dossier_projet):
            print(f"[Skip] Le projet {nom_projet} a déjà été calculé. Passage au suivant.")
            continue
            
        print(f"\n[>>>] Préparation du projet #{crossovers_traites + 1} : {nom_projet}")
        
        # --- VÉRIFICATION DE SÉCURITÉ ---
        fichiers_manquants = check_all_files_exist(woofer, 'W') + check_all_files_exist(tweeter, 'T')
        
        if fichiers_manquants:
            print(f"[-] Impossible de traiter {nom_projet}. Données directivité ou impédance manquantes :")
            for m in fichiers_manquants: 
                print(f"    - {m}")
            continue

        # --- COMMANDE SIMPLIFIÉE ---
        cmd = [
            "python", "run.py",
            "--woofer", woofer,
            "--tweeter", tweeter,
            "--data_dir", DOSSIER_DATA,
            "--out_dir", dossier_projet,
            "--name", nom_projet,
            "--gen", "10",  # Ajustez le nombre de générations ici
            "--pop", "120"
        ]
        
        print(f"[*] Lancement de l'algorithme génétique...")
        try:
            subprocess.run(cmd, check=True)
            
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
            crossovers_traites += 1
            
        except subprocess.CalledProcessError as e:
            print(f"[-] L'optimisation a échoué (Crash) pour {nom_projet}. Code d'erreur : {e.returncode}")
            if os.path.exists(dossier_projet):
                shutil.rmtree(dossier_projet)

    print("\n" + "=" * 50)
    print(f"Mission accomplie. {crossovers_traites} nouveaux crossovers générés.")

if __name__ == "__main__":
    process_batch()