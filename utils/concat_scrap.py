import pandas as pd

def concat_catalogs(files, output_file):
    '''Concatène plusieurs fichiers CSV de catalogue de composants en un seul, en supprimant les doublons et en filtrant les composants non pertinents (ex: AWG > 18).
    Le CSV SoundImports doit être le premier élément de la liste'''
    
    dfs = []
    df_SI = pd.read_csv(files[0])
    df_SI['Price'] = df_SI['Price'] * 1.21  # Application de la TVA à 21% sur les prix SoundImports
    # Arrondis au centime
    df_SI['Price'] = df_SI['Price'].round(2)
    dfs.append(df_SI)
    print(f"  ✅ {files[0]} chargé ({len(df_SI)} pièces)")

    for f in files[1:]:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  ✅ {f} chargé ({len(df)} pièces)")

    combined_df = pd.concat(dfs, ignore_index=True)
    if 'AWG' in combined_df.columns:
        # Remove every row where AWG is superior to 18 (inclusive)
        combined_df = combined_df[combined_df['AWG'] <= 18]
    
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