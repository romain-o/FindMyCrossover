import pandas as pd

def categorize_pair(price_w, price_t, pop_w, pop_t):
    """Détermine la gamme et la catégorie marketing du duo."""
    total_price = price_w + price_t
    
    # 1. Gamme de Prix (Budget combiné Woofer + Tweeter)
    if total_price < 80:
        tier = "Entrée de gamme"
    elif total_price < 200:
        tier = "Milieu de gamme"
    else:
        tier = "High-End"
        
    # 2. Calcul du Score de Popularité (Somme des étoiles)
    pop_score = 0
    for p in [pop_w, pop_t]:
        if pd.isna(p): continue
        p_str = str(p)
        if "★★★★★" in p_str: pop_score += 5
        elif "★★★★☆" in p_str: pop_score += 4
        elif "★★★☆☆" in p_str: pop_score += 3
        elif "★★☆☆☆" in p_str: pop_score += 2
        elif "★☆☆☆☆" in p_str: pop_score += 1
        
    # 3. Catégorisation Marketing
    if pop_score >= 8:
        category = "Ultra Populaire (Classique DIY)"
    elif pop_score >= 6:
        category = "Valeur Sûre"
    else:
        category = "Niche / Original"
        
    return tier, category, total_price, pop_score

def extract_prioritized_pairs(input_file="Dataset_driver.xlsx", output_file="paires_W_T_priorisees.csv"):
    df = pd.read_excel(input_file)
    
    # Dictionnaires pour un accès rapide aux données
    price_dict = dict(zip(df['Driver'], df['Prix PartsExpress ($)']))
    pop_dict = dict(zip(df['Driver'], df['Popularité']))
    driver_types = dict(zip(df['Driver'], df['Type (W-M-T-F)']))

    pairs = set()
    col_combos = 'Combos possibles (drivers de la liste)'
    
    # Extraction des paires uniques (Woofer, Tweeter)
    for index, row in df.iterrows():
        driver1 = row['Driver']
        type1 = row['Type (W-M-T-F)']
        combos = row[col_combos]
        
        if pd.isna(combos): continue
            
        combo_list = [c.strip() for c in str(combos).split(';')]
        for driver2 in combo_list:
            type2 = driver_types.get(driver2)
            if type1 == 'W' and type2 == 'T':
                pairs.add((driver1, driver2))
            elif type1 == 'T' and type2 == 'W':
                pairs.add((driver2, driver1))

    # Enrichissement avec les métadonnées
    results = []
    for w, t in pairs:
        price_w = price_dict.get(w, 0)
        price_t = price_dict.get(t, 0)
        pop_w = pop_dict.get(w, "")
        pop_t = pop_dict.get(t, "")
        
        # Gestion des prix manquants
        if pd.isna(price_w): price_w = 0
        if pd.isna(price_t): price_t = 0
        
        tier, category, total_price, pop_score = categorize_pair(price_w, price_t, pop_w, pop_t)
        
        # 4. ALGORITHME DE PRIORISATION (Plus le score est haut, plus il faut le générer vite)
        priority_score = pop_score * 10 
        
        # Bonus énorme pour les duos très célèbres (Le coeur de cible)
        if category == "Ultra Populaire (Classique DIY)":
            priority_score += 50
            
        # Bonus pour les prix attractifs (les gens qui achètent ces HPs sont friands de DIY pas cher)
        if tier == "Milieu de gamme":
            priority_score += 20
        elif tier == "Entrée de gamme":
            priority_score += 10
            
        results.append({
            'Woofer (W)': w,
            'Tweeter (T)': t,
            'Gamme': tier,
            'Catégorie': category,
            'Prix Total ($)': total_price,
            'Score Popularité': pop_score,
            'Priorité Calcul': priority_score
        })

    df_results = pd.DataFrame(results)
    
    # Tri décroissant par Priorité, puis croissant par Prix (pour commencer par les moins chers à priorité égale)
    df_results = df_results.sort_values(by=['Priorité Calcul', 'Prix Total ($)'], ascending=[False, True])
    
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Extraction terminée ! Les {len(df_results)} paires ont été triées par priorité dans '{output_file}'.")

if __name__ == "__main__":
    extract_prioritized_pairs(input_file=r"C:\Geekosphere\FindMyCrossover\Speaker_dataset\Dataset_driver.xlsx",
                              output_file=r"C:\Geekosphere\FindMyCrossover\Speaker_dataset\W_T_pairs.csv")