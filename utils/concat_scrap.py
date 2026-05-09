import pandas as pd

def concat_catalogs(files, output_file):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  ✅ {f} chargé ({len(df)} pièces)")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False, sep=',', encoding='utf-8')
    print(f"\n🎉 Catalogues combinés enregistrés dans {output_file} ({len(combined_df)} pièces au total)")
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔗 CONCATÉNATION DES CATALOGUES CSV")
    print("="*60)
    
    input_files = [
        ['catalog_capacitors_SI.csv', 'catalog_capacitors_PE.csv'],
        ['catalog_inductors_SI.csv', 'catalog_inductors_PE.csv'],
        ['catalog_resistors_SI.csv', 'catalog_resistors_PE.csv']
    ]
    output_file = [
        'catalog_capacitors.csv',
        'catalog_inductors.csv',
        'catalog_resistors.csv'
    ]
    
    for k in range(3):
        print(f"\n📂 Traitement des {['condensateurs', 'inductances', 'résistances'][k]}...")
        concat_catalogs(input_files[k], output_file[k])
    
    print("\n" + "="*60)