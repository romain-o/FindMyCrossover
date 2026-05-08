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
        """Trouve la valeur EXACTE la plus proche disponible dans votre catalogue."""
        if val <= 0: return val
        if comp_type == 'C': arr = self.vals_c
        elif comp_type == 'L': arr = self.vals_l
        else: arr = self.vals_r
        
        idx = np.abs(arr - val).argmin()
        return arr[idx]

    def get_part_info(self, val, comp_type):
        """Récupère toutes les infos du composant pour la Parts List"""
        if comp_type == 'C':
            match = self.df_c.iloc[np.abs(self.df_c['Value'] - val*1e6).argmin()]
        elif comp_type == 'L':
            match = self.df_l.iloc[np.abs(self.df_l['Value'] - val*1e3).argmin()]
        else:
            match = self.df_r.iloc[np.abs(self.df_r['Value'] - val).argmin()]
        return match
