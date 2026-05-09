from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# ==========================================
# CONFIGURATION
# ==========================================
BASE_URL = "https://www.soundimports.eu"

CATEGORIES = {
    "Capacitors": "https://www.soundimports.eu/en/crossover-components/capacitors/",
    "Inductors": "https://www.soundimports.eu/en/crossover-components/coils/",
    "Resistors": "https://www.soundimports.eu/en/crossover-components/resistors/"
}

# ==========================================
# FONCTIONS DE NETTOYAGE
# ==========================================
def extract_num(text):
    if not text: return None
    match = re.search(r"(\d+[\.\,]\d+|\d+)", text)
    return float(match.group(1).replace(',', '.')) if match else None

def clean_price(price_str):
    if not price_str: return None
    cleaned = re.sub(r'[^\d,\.]', '', price_str).replace(',', '.')
    try: return float(cleaned)
    except: return None

def parse_structured_title(title, category):
    parts = [p.strip() for p in title.split('|')]
    if len(parts) < 2: return None
    
    data = {"PartNumber": parts[0]}
    try:
        if category == "Capacitors":
            data["Value"] = extract_num(parts[1])
            data["Tolerance_pct"] = extract_num(parts[2]) if len(parts) > 2 else 5.0
            data["Voltage_V"] = extract_num(parts[3]) if len(parts) > 3 else 400.0
            data["Type"] = "Film Capacitor"
        elif category == "Resistors":
            data["Value"] = extract_num(parts[1])
            data["Power_W"] = extract_num(parts[2]) if len(parts) > 2 else 10.0
            data["Tolerance_pct"] = extract_num(parts[3]) if len(parts) > 3 else 2.0
        elif category == "Inductors":
            data["Value"] = extract_num(parts[1])
            data["DCR_Ohm"] = extract_num(parts[2]) if len(parts) > 2 else 0.5
            data["Tolerance_pct"] = extract_num(parts[3]) if len(parts) > 3 else 3.0
            data["AWG"] = extract_num(parts[4]) if len(parts) > 4 else 18.0
            data["Type"] = "Air Core"
    except Exception:
        return None
        
    if data.get("Value") is None: return None
    return data

# ==========================================
# SCRIPT PRINCIPAL SELENIUM
# ==========================================
if __name__ == "__main__":
    print("🚀 Démarrage du Robot Selenium (Mode Blindé)...")
    
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    
    datasets = {"Capacitors": [], "Inductors": [], "Resistors": []}
    
    for cat_name, cat_url in CATEGORIES.items():
        print(f"\n[{cat_name}] Ouverture de la page vitrine...")
        
        # --- NOUVEAU : Mécanisme de Retry pour éviter l'erreur ERR_NAME_NOT_RESOLVED ---
        success = False
        for attempt in range(3):
            try:
                driver.get(cat_url)
                success = True
                break # Si ça marche, on sort de la boucle d'essai
            except Exception as e:
                print(f"  ⚠️ Erreur réseau. Tentative {attempt+1}/3 dans 5 secondes...")
                time.sleep(5)
                
        if not success:
            print(f"  ❌ Impossible de charger {cat_name} après 3 essais. On passe à la suite.")
            continue
            
        time.sleep(3) 
        
        # --- Gestion des Cookies ---
        try:
            cookie_btn = driver.find_element(By.XPATH, "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]")
            cookie_btn.click()
            time.sleep(1)
        except:
            pass 
        
        page = 1
        seen_links = set()
        
        while True:
            # 1. Scraping des données affichées à l'écran
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            product_links = soup.find_all('a', href=re.compile(r'/en/[^/]+\.html'))
            
            new_items_count = 0
            
            for a_tag in product_links:
                title = a_tag.get('title') or a_tag.text.strip()
                href = a_tag['href']
                full_link = href if href.startswith('http') else BASE_URL + href
                
                if '|' in title and full_link not in seen_links:
                    seen_links.add(full_link)
                    new_items_count += 1
                    
                    parent = a_tag.find_parent('div', class_=re.compile(r'product|item|card|col-'))
                    
                    # 1. On cherche le nœud texte avec le symbole €
                    price_node = parent.find(string=re.compile(r'€')) if parent else None
                    
                    # 2. On remonte au parent (le <div> ou <span>) et on prend TOUT son texte
                    price_text = price_node.parent.text.strip() if (price_node and price_node.parent) else ""
                    
                    # 3. Le nettoyage fera le reste (ex: "€ 1,\n49" -> 1.49)
                    price_clean = clean_price(price_text)
                    comp_data = parse_structured_title(title, cat_name)
                    
                    if comp_data is not None:
                        comp_data["Description"] = title
                        comp_data["URL"] = full_link
                        comp_data["Price"] = price_clean if price_clean else 0.00
                        datasets[cat_name].append(comp_data)
            
            print(f"  -> Page {page} lue : {new_items_count} NOUVEAUX composants.")
            
            # --- SÉCURITÉ ANTI-BOUCLE ---
            if new_items_count == 0:
                print(f"  🏁 Aucun nouveau composant trouvé. Fin de la catégorie {cat_name} !")
                break
            
            # 2. LE CLIC MAGIQUE
            try:
                next_button = driver.find_element(By.ID, "next-page")
                
                if not next_button.is_displayed() or not next_button.is_enabled():
                    print("  🏁 Bouton 'Suivant' inactif. Fin de la catégorie !")
                    break
                
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(1) 
                driver.execute_script("arguments[0].click();", next_button)
                
                page += 1
                print(f"  [Clic réussi] Chargement de la page {page}...")
                
                # --- NOUVEAU : On attend 10 secondes au lieu de 4 pour laisser le site charger les condensateurs ---
                time.sleep(10) 
                
            except Exception as e:
                print("  🏁 Bouton 'Suivant' introuvable. Fin de la catégorie !")
                break 
                
    driver.quit()
            
    # ==========================================
    # FORMATAGE & SAUVEGARDE EN CSV
    # ==========================================
    print("\n💾 Structuration des bases de données...")
    
    if datasets["Capacitors"]:
        df_cap = pd.DataFrame(datasets["Capacitors"])
        df_cap = df_cap[['PartNumber', 'Description', 'Value', 'Tolerance_pct', 'Voltage_V', 'Type', 'URL', 'Price']]
        df_cap = df_cap.sort_values(by=['Value', 'Price'])
        df_cap.to_csv('catalog_capacitors_SI.csv', index=False, sep=',', encoding='utf-8')
        print(f"  ✅ catalog_capacitors_SI.csv ({len(df_cap)} pièces)")

    if datasets["Inductors"]:
        df_ind = pd.DataFrame(datasets["Inductors"])
        df_ind = df_ind[['PartNumber', 'Description', 'Value', 'DCR_Ohm', 'AWG', 'Type', 'URL', 'Price']]
        df_ind = df_ind.sort_values(by=['Value', 'Price'])
        df_ind.to_csv('catalog_inductors_SI.csv', index=False, sep=',', encoding='utf-8')
        print(f"  ✅ catalog_inductors_SI.csv ({len(df_ind)} pièces)")

    if datasets["Resistors"]:
        df_res = pd.DataFrame(datasets["Resistors"])
        df_res = df_res[['PartNumber', 'Description', 'Value', 'Power_W', 'Tolerance_pct', 'URL', 'Price']]
        df_res = df_res.sort_values(by=['Value', 'Price'])
        df_res.to_csv('catalog_resistors_SI.csv', index=False, sep=',', encoding='utf-8')
        print(f"  ✅ catalog_resistors_SI.csv ({len(df_res)} pièces)")
        
    print("\n🎉 Terminé ! Les catalogues sont prêts à être utilisés par l'IA.")