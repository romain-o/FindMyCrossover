import pandas as pd
import numpy as np
from src.nodes import Capacitor, Inductor, Resistor

class CatalogManager:
    def __init__(self):
        # Chargement des bases de données
        self.df_c = pd.read_csv('catalog_capacitors.csv')
        self.df_l = pd.read_csv('catalog_inductors.csv')
        self.df_r = pd.read_csv('catalog_resistors.csv')

        # Extraction des valeurs uniques triées et conversion en unités SI pures (Farads, Henries, Ohms)
        self.vals_c = np.sort(self.df_c['Value'].dropna().unique()) * 1e-6
        self.vals_l = np.sort(self.df_l['Value'].dropna().unique()) * 1e-3
        self.vals_r = np.sort(self.df_r['Value'].dropna().unique())

    def get_comp_type(self, comp):
        if isinstance(comp, Capacitor): return 'C'
        elif isinstance(comp, Inductor): return 'L'
        else: return 'R'

    def snap_to_catalog(self, val, comp_type):
        """
        Cherche les composants dans une fenêtre de ±5% autour de la valeur cible,
        et retourne la valeur standard correspondant au composant le moins cher.
        En cas d'égalité de prix, la valeur la plus proche acoustiquement est privilégiée.
        """
        import numpy as np
        import pandas as pd
        import math
        
        if val <= 0: return val
        if comp_type == 'C': arr = self.vals_c
        elif comp_type == 'L': arr = self.vals_l
        else: arr = self.vals_r
        
        if len(arr) == 0: return val
        
        # Fonction utilitaire interne pour extraire le prix de manière sécurisée
        def get_cheapest_price(info):
            if info is None:
                return float('inf')
                
            if isinstance(info, pd.DataFrame):
                if info.empty: return float('inf')
                return float(info['Price'].min(skipna=True))
                
            if isinstance(info, list):
                if not info: return float('inf')
                prices = []
                for item in info:
                    try:
                        p = float(item.get('Price', float('inf')))
                        if not math.isnan(p): prices.append(p)
                    except: pass
                return min(prices) if prices else float('inf')
                
            try:
                p = float(info.get('Price', float('inf')) if hasattr(info, 'get') else info['Price'])
                return p if not math.isnan(p) else float('inf')
            except:
                return float('inf')

        # 1. Définition de la fenêtre de tolérance (±5%)
        lower_bound = val * 0.98
        upper_bound = val * 1.02
        
        # 2. Identification des valeurs du catalogue présentes dans cette fenêtre
        candidates = arr[(arr >= lower_bound) & (arr <= upper_bound)]
        
        # 3. Mécanisme de repli : Si aucun composant n'est dans la fenêtre de 5%
        if len(candidates) == 0:
            idx = np.searchsorted(arr, val)
            if idx == 0: 
                candidates = [arr[0]]
            elif idx == len(arr): 
                candidates = [arr[-1]]
            else: 
                candidates = [arr[idx-1], arr[idx]]

        # 4. Évaluation des candidats pour trouver le meilleur compromis Prix/Précision
        best_val = val
        lowest_price = float('inf')
        closest_dist = float('inf')
        
        for cand_val in candidates:
            info = self.get_part_info(cand_val, comp_type)
            price = get_cheapest_price(info)
            dist = abs(cand_val - val)
            
            # Sélection : On retient le prix le plus bas.
            # En cas d'égalité stricte (ou si les prix sont manquants/infinis), 
            # on privilégie la valeur qui dévie le moins de l'idéal mathématique.
            if price < lowest_price:
                lowest_price = price
                best_val = cand_val
                closest_dist = dist
            elif price == lowest_price and dist < closest_dist:
                best_val = cand_val
                closest_dist = dist
                
        return best_val

    def get_part_info(self, val, comp_type):
        """Récupère la pièce LA MOINS CHÈRE pour une valeur donnée dans le catalogue"""
        if comp_type == 'C':
            df = self.df_c
            target_val = val * 1e6
        elif comp_type == 'L':
            df = self.df_l
            target_val = val * 1e3
        else:
            df = self.df_r
            target_val = val

        # On trouve toutes les pièces qui ont cette valeur exacte
        matches = df[np.isclose(df['Value'], target_val, atol=1e-5)]
        
        if not matches.empty:
            # On trie par prix croissant et on prend la première (la moins chère dont le prix est connu)
            return matches.sort_values(by='Price', ascending=True).iloc[0]
        else:
            # Sécurité (fallback standard)
            idx = np.abs(df['Value'] - target_val).argmin()
            return df.iloc[idx]
