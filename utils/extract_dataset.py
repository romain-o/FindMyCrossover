import os
import zipfile
import shutil
from pathlib import Path
import re

def extract_model_name(filename):
    """
    Extrait le nom pur du haut-parleur depuis le nom du fichier.
    Ex: "DS315-8@15.frd" -> "DS315-8"
    """
    base = Path(filename).stem
    # On coupe au niveau du '@' ou du '_' si présent
    for separator in ['@', '_']:
        if separator in base:
            base = base.split(separator)[0]
    return base.strip()

def detect_angle(filename):
    """
    Détecte l'angle de mesure à partir du nom du fichier FRD.
    """
    name_lower = filename.lower()
    
    # 0 Degré (Axial)
    if any(x in name_lower for x in ['@0', '_0deg', '0_deg', 'on-axis', 'onaxis']):
        return '0deg'
    # 15 Degrés
    elif any(x in name_lower for x in ['@15', '_15deg', '15_deg']):
        return '15deg'
    # 30 Degrés
    elif any(x in name_lower for x in ['@30', '_30deg', '30_deg']):
        return '30deg'
    # 45 Degrés (On inclut @40 au cas où ce soit une typo fréquente comme mentionné)
    elif any(x in name_lower for x in ['@45', '_45deg', '45_deg']):
        return '45deg'
        
    return '0deg' # Si aucun angle n'est détecté, on assume 0 par défaut

def extract_dataset_full_angles(source_dir, dest_dir):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Définition des sous-dossiers requis
    angles = ['0deg', '15deg', '30deg', '45deg']
    categories = ['Woofers', 'Tweeters']
    
    # Création de l'arborescence complète
    for cat in categories:
        (dest_path / cat / 'ZMA').mkdir(parents=True, exist_ok=True)
        for angle in angles:
            (dest_path / cat / 'FRD' / angle).mkdir(parents=True, exist_ok=True)
            
    extracted_count = 0
    
    for zip_file in source_path.rglob('*.zip'):
        try:
            # 1. Détecter la catégorie principale (Woofer ou Tweeter)
            if "Woofer" in zip_file.parts:
                category_dir = dest_path / 'Woofers'
            elif "Tweeter" in zip_file.parts:
                category_dir = dest_path / 'Tweeters'
            else:
                continue

            with zipfile.ZipFile(zip_file, 'r') as z:
                model_name = None
                
                # 2. Scanner une première fois pour trouver le nom du modèle propre
                for file_info in z.infolist():
                    name_lower = file_info.filename.lower()
                    if '__macosx' in name_lower: continue
                    if name_lower.endswith('.zma'):
                        model_name = extract_model_name(file_info.filename)
                        break
                
                # Fallback si le ZMA n'a pas aidé
                if not model_name:
                    model_name = zip_file.stem.split('--')[-1].replace('_data', '')

                # 3. Scanner et extraire les fichiers
                for file_info in z.infolist():
                    filename_lower = file_info.filename.lower()
                    
                    if '__macosx' in filename_lower or file_info.is_dir():
                        continue
                        
                    # Gestion du fichier d'Impédance (.zma)
                    if filename_lower.endswith('.zma'):
                        dest_file = category_dir / 'ZMA' / f"{model_name}.zma"
                        with z.open(file_info.filename) as source, open(dest_file, "wb") as target:
                            shutil.copyfileobj(source, target)
                        extracted_count += 1
                        
                    # Gestion des fichiers de Réponse (.frd)
                    elif filename_lower.endswith('.frd') or filename_lower.endswith('.txt'):
                        angle_folder = detect_angle(file_info.filename)
                        dest_file = category_dir / 'FRD' / angle_folder / f"{model_name}_{angle_folder}.frd"
                        
                        with z.open(file_info.filename) as source, open(dest_file, "wb") as target:
                            shutil.copyfileobj(source, target)
                        extracted_count += 1
                        
                print(f"[OK] Extraction complète : {model_name}")

        except zipfile.BadZipFile:
            print(f"[Erreur] Fichier ZIP corrompu : {zip_file.name}")
        except Exception as e:
            print(f"[Erreur] Problème avec {zip_file.name} : {str(e)}")

    print(f"\n--- Terminé ! ---")
    print(f"Total de {extracted_count} fichiers (FRD/ZMA) classés dans {dest_dir}")

if __name__ == "__main__":
    dossier_source = r"C:\Geekosphere\FindMyCrossover\Speaker_dataset"
    dossier_destination = r"C:\Geekosphere\FindMyCrossover\data"
    
    extract_dataset_full_angles(dossier_source, dossier_destination)