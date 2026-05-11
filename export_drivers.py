import os
import json
import zipfile
import argparse

def get_base_paths(data_dir, driver_name, category):
    """
    Localise les chemins de base vers les fichiers de mesure FRD (dans l'axe) et ZMA.
    """
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
    parser = argparse.ArgumentParser(description="Archive les mesures FRD/ZMA (axe et hors axe) d'un projet dans un fichier ZIP.")
    parser.add_argument("--name", required=True, help="Nom du dossier projet (ex: RS225-8_X_RS52AN-8_X_RST28F-4)")
    parser.add_argument("--data_dir", default="data", help="Dossier racine des mesures")
    args = parser.parse_args()

    project_dir = os.path.join("crossovers", args.name)
    metadata_file = os.path.join(project_dir, f"{args.name}_metadata.json")
    zip_filename = os.path.join(project_dir, f"{args.name}_Measurements.zip")

    if not os.path.exists(project_dir):
        print(f"[-] Erreur : Le dossier projet '{project_dir}' est introuvable.")
        return

    # 1. Identification des haut-parleurs via les métadonnées
    drivers_to_export = []
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            w_name = meta.get("Woofer", "").replace("2x ", "")
            m_name = meta.get("Midrange")
            t_name = meta.get("Tweeter")

            if w_name: drivers_to_export.append((w_name, 'W'))
            if m_name: drivers_to_export.append((m_name, 'M'))
            if t_name: drivers_to_export.append((t_name, 'T'))
    else:
        print("[!] Avertissement : metadata.json absent. Tentative d'extraction via le nom du dossier.")
        parts = args.name.replace("2x_", "").split("_X_")
        if len(parts) == 3:
            drivers_to_export = [(parts[0], 'W'), (parts[1], 'M'), (parts[2], 'T')]
        elif len(parts) == 2:
            drivers_to_export = [(parts[0], 'W'), (parts[1], 'T')]

    if not drivers_to_export:
        print("[-] Impossible d'identifier les composants du projet.")
        return

    # Liste des angles à rechercher (modifiables selon vos standards de mesure)
    angles_to_check = ['0deg', '15deg', '30deg', '45deg', '60deg']

    # 2. Création de l'archive ZIP
    print(f"[+] Création de l'archive : {zip_filename}")
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for d_name, cat in drivers_to_export:
                print(f"\n[*] Traitement du composant : {d_name} ({cat})")
                frd_base, zma = get_base_paths(args.data_dir, d_name, cat)
                
                # Vérification et ajout des fichiers FRD (0° et hors axe)
                for angle in angles_to_check:
                    # Le remplacement génère le bon dossier (ex: .../FRD/15deg/...) et le bon fichier (_15deg.frd)
                    frd_path = frd_base.replace('0deg', angle)
                    
                    if os.path.exists(frd_path):
                        zipf.write(frd_path, arcname=os.path.basename(frd_path))
                        print(f"    -> Ajouté : {os.path.basename(frd_path)}")
                    elif angle == '0deg':
                        # Seul le 0deg est critique, son absence lève un avertissement
                        print(f"    [!] Fichier fondamental manquant : {frd_path}")

                # Vérification et ajout du fichier ZMA (Impédance)
                if os.path.exists(zma):
                    zipf.write(zma, arcname=os.path.basename(zma))
                    print(f"    -> Ajouté : {os.path.basename(zma)}")
                else:
                    print(f"    [!] Fichier d'impédance manquant : {zma}")

        print(f"\n[+] ✅ Succès. L'archive a été générée avec toutes les courbes disponibles.")
        
    except Exception as e:
        print(f"[-] Erreur lors de la création du ZIP : {e}")

if __name__ == "__main__":
    main()