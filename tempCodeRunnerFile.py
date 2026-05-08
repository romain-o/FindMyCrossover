import pandas as pd
import numpy as np

def clean_catalog():
    # 1. Capacitors (Condensateurs)
    df_c = pd.read_csv('capacitors.csv') # Remplacez par votre vrai nom de fichier
    df_c.columns = ['PartNumber', 'Description', 'Value', 'Tolerance_pct', 'Voltage_V', 'Type', 'URL', 'Price']
    df_c['Price'] = df_c['Price'].replace('Prix non listé', np.nan)
    df_c['Price'] = df_c['Price'].astype(str).str.replace('$', '').astype(float)
    df_c.to_csv('catalog_capacitors.csv', index=False)

    # 2. Resistors (Résistances)
    df_r = pd.read_csv('resistors.csv')
    df_r.columns = ['PartNumber', 'Description', 'Value', 'Power_W', 'Tolerance_pct', 'URL', 'Price']
    df_r['Price'] = df_r['Price'].replace('Prix non listé', np.nan)
    df_r['Price'] = df_r['Price'].astype(str).str.replace('$', '').astype(float)
    df_r.to_csv('catalog_resistors.csv', index=False)

    # 3. Inductors (Bobines)
    df_l = pd.read_csv('inductors.csv')
    df_l.columns = ['PartNumber', 'Description', 'Value', 'DCR_Ohm', 'AWG', 'Type', 'URL', 'Price']
    df_l['Price'] = df_l['Price'].replace('Prix non listé', np.nan)
    df_l['Price'] = df_l['Price'].astype(str).str.replace('$', '').astype(float)
    df_l.to_csv('catalog_inductors.csv', index=False)
    
    print("[+] Les 3 catalogues ont été formatés, traduits et nettoyés !")

if __name__ == "__main__":
    clean_catalog()