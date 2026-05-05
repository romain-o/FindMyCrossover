import os
import zipfile
from pathlib import Path
import re

def extract_model_name(filename):
    base = Path(filename).stem
    for separator in ['@', '_']:
        if separator in base:
            base = base.split(separator)[0]
    return base.strip()

def detect_angle(filename):
    name_lower = filename.lower()
    if any(x in name_lower for x in ['@0', '_0deg', '0_deg', 'on-axis', 'onaxis','_0']): return '0deg'
    elif any(x in name_lower for x in ['@15', '_15deg', '15_deg','_15']): return '15deg'
    elif any(x in name_lower for x in ['@30', '_30deg', '30_deg','_30']): return '30deg'
    elif any(x in name_lower for x in ['@45', '_45deg', '45_deg', '_45']): return '45deg'
    return '0deg'

def clean_and_format_data(raw_bytes):
    text = raw_bytes.decode('utf-8', errors='ignore')
    clean_lines = []
    magnitudes = []

    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        
        if re.match(r'^[+-]?\d', line):
            parts = re.split(r'\s+', line)
            try:
                vals = [float(p) for p in parts[:3]]
                if len(vals) >= 2:
                    if len(vals) == 2: 
                        vals.append(0.0)
                    
                    magnitudes.append(vals[1])
                    clean_lines.append(f"{vals[0]:.4f}\t{vals[1]:.4f}\t{vals[2]:.4f}")
            except ValueError:
                pass

    mean_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
    return "\n".join(clean_lines), mean_mag

def build_perfect_dataset(source_dir, dest_dir):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    angles = ['0deg', '15deg', '30deg', '45deg']
    for cat in ['Woofers', 'Tweeters']:
        (dest_path / cat / 'ZMA').mkdir(parents=True, exist_ok=True)
        for angle in angles:
            (dest_path / cat / 'FRD' / angle).mkdir(parents=True, exist_ok=True)
            
    extracted_count = 0
    corrupted_count = 0
    ignored_txt = 0
    
    for zip_file in source_path.rglob('*.zip'):
        try:
            if "Woofer" in zip_file.parts: category_dir = dest_path / 'Woofers'
            elif "Tweeter" in zip_file.parts: category_dir = dest_path / 'Tweeters'
            else: continue

            with zipfile.ZipFile(zip_file, 'r') as z:
                model_name = None
                for file_info in z.infolist():
                    name_lower = file_info.filename.lower()
                    if '__macosx' in name_lower: continue
                    if name_lower.endswith('.zma'):
                        model_name = extract_model_name(file_info.filename)
                        break
                if not model_name: model_name = zip_file.stem.split('--')[-1].replace('_data', '')

                for file_info in z.infolist():
                    filename_lower = file_info.filename.lower()
                    if '__macosx' in filename_lower or file_info.is_dir(): continue
                    
                    # On ignore explicitement les fichiers .txt maintenant !
                    if filename_lower.endswith('.txt'):
                        ignored_txt += 1
                        continue
                        
                    is_zma = filename_lower.endswith('.zma')
                    # Seuls les vrais .frd sont acceptés
                    is_frd = filename_lower.endswith('.frd')
                    
                    if not (is_zma or is_frd): continue

                    with z.open(file_info.filename) as source:
                        raw_data = source.read()
                        clean_text, mean_mag = clean_and_format_data(raw_data)
                    
                    if len(clean_text.strip()) < 10:
                        continue

                    if is_zma:
                        dest_file = category_dir / 'ZMA' / f"{model_name}.zma"
                    else:
                        angle_folder = detect_angle(file_info.filename)
                        dest_file = category_dir / 'FRD' / angle_folder / f"{model_name}_{angle_folder}.frd"
                        
                    with open(dest_file, "w", encoding='utf-8') as target:
                        target.write(clean_text)
                    extracted_count += 1
                    
                print(f"[OK] Traité avec succès : {model_name}")

        except Exception as e:
            print(f"[-] Problème avec {zip_file.name} : {str(e)}")

    print(f"\n--- RAPPORT DE NETTOYAGE ---")
    print(f"✅ [{extracted_count}] fichiers parfaitement formatés dans {dest_dir}.")
    print(f"🗑️  [{ignored_txt}] fichiers '.txt' ignorés (fichiers parasites).")
    if corrupted_count > 0:
        print(f"❌ [{corrupted_count}] fichiers mal nommés par le constructeur détruits.")

if __name__ == "__main__":
    dossier_source = r"C:\Geekosphere\FindMyCrossover\Speaker_dataset"
    dossier_destination = r"C:\Geekosphere\FindMyCrossover\data"
    
    print("Démarrage du grand nettoyage...")
    build_perfect_dataset(dossier_source, dossier_destination)