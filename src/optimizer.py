import numpy as np
import random
import json
import os
import itertools # <-- NOUVEL IMPORT POUR LE BRUTE FORCE
import csv

import matplotlib.pyplot as plt
from src.nodes import DriverNode, SeriesNode, ParallelNode, ShuntNode, Capacitor, Inductor, Resistor, Node, ComponentNode
from src.evaluator import CircuitEvaluator
from src.mutator import TreeMutator
from src.catalog_manager import CatalogManager
from scipy.optimize import minimize
from src.schematic import SchematicRenderer
import pandas as pd
import time

# ============================================================
# OPTIMISATION #1 : Fonctions module-level pour multiprocessing
# ============================================================
from multiprocessing import Pool, cpu_count

_pool_optimizer = None

def _pool_init(opt):
    global _pool_optimizer
    _pool_optimizer = opt

def _pool_fitness(ind):
    score = _pool_optimizer.fitness(ind)
    return score, ind.get('wiring', {})

def _pool_elite(args):
    return _pool_optimizer._elite_worker(args)

def _pool_lamarckian(child):
    return _pool_optimizer._lamarckian_worker(child)


BOUNDS_R = (0.1, 33)
BOUNDS_C = (0.1e-6, 300e-6)
BOUNDS_L = (0.05e-3, 5e-3)

CATALOG = CatalogManager()

WEIGHTS = {
    'crossover': 3.2549,
    'fc_err': 20.0,
    'impedance': 142.9232,
    'tweeter_low': 48.4022,
    'woofer_high': 10.0,
    'woofer_attenuation': 218.2858,
    'thermal': 20,
    #'components': 0.6808,
    'components': 0.5,
    'resistors': 0.4,
    'mse_sum': 1.0000,
    'n_comps': 10,
    
    'midrange_low': 25.0,
    'midrange_high': 25.0,
    'midrange_participation': 60.0,       
    'midrange_attenuation': 200,
}
class WayConfig:
    """Configuration d'une voie acoustique (Grave, Médium, Aigu, etc.)"""
    # NOUVEAU : Ajout de l'argument 'count'
    def __init__(self, label, frd_path, zma_path, order=4, z_offset=0.0, y_offset=0.0, x_offset=0.0, count=1):
        self.label = label
        self.frd_path = frd_path
        self.zma_path = zma_path
        self.order = order
        self.z_offset = z_offset
        self.y_offset = y_offset 
        self.x_offset = x_offset
        self.count = count # <-- Nombre de drivers pour cette voie
        self.driver = DriverNode(label, frd_path, zma_path)

class CrossoverOptimizer:
    def __init__(self, ways_configs, target_fc=0.0, app_config=None):
        self.ways = ways_configs
        self.app_config = app_config if app_config else {}
        self.mutator = TreeMutator(
            prob_value_mut=0.2, prob_type_mut=0.1, prob_topology_mut=0.3, 
            prob_add_node=0.35, prob_remove_node=0.05
        )
        
        global WEIGHTS
        self.weights = self.app_config.get("weights", WEIGHTS)
        self.spl_offset = self.app_config.get("spl_settings", {}).get("target_offset_db", 0.0)
        self.range_config = self.app_config.get("optimization_range", {"mode": "auto"})
        
        self.target_fc = target_fc

        # --- Démarrage en mode "Ultra-Rapide" (Exploration) ---
        self._set_resolution(1000)

    def _prepare_driver(self, way):
        d = way.driver
        # 1. Sauvegarde définitive des données brutes (Indispensable pour changer de résolution)
        if not hasattr(d, 'H_raw'):
            d.H_raw = d.H_acoustic.copy()
            d.Z_raw = d.Z_complex.copy()

        raw_name = os.path.basename(way.frd_path).split('.')[0].split('@')[0]
        d.model_name = raw_name.replace('_0deg', '')
        
        # 2. Interpolation basée STRICTEMENT sur le raw
        mag_db = 20 * np.log10(np.abs(d.H_raw) + 1e-10)
        ph_unwrapped = np.unwrap(np.angle(d.H_raw))
        
        mag_interp = np.interp(self.freqs, d.frd_freqs, mag_db)
        ph_interp = np.interp(self.freqs, d.frd_freqs, ph_unwrapped)
        d.H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
        
        delay_s = np.linalg.norm([way.x_offset, way.y_offset, way.z_offset + 2]) / 343.0
        phase_delay = np.exp(-1j * 2 * np.pi * self.freqs * delay_s)
        d.H_acoustic *= phase_delay

        z_mag = np.abs(d.Z_raw)
        z_ph = np.unwrap(np.angle(d.Z_raw))
        d.Z_complex = np.interp(self.freqs, d.zma_freqs, z_mag) * np.exp(1j * np.interp(self.freqs, d.zma_freqs, z_ph))

        d.H_base = d.H_acoustic.copy()
        d.Z_base = d.Z_complex.copy()

    def _set_resolution(self, n_points):
        """Reconstruit entièrement la grille de calcul avec 'n_points' fréquences."""
        print(f"[*] Ajustement de la résolution d'optimisation : {n_points} points")
        self.freqs = np.geomspace(20, 20000, n_points)
        self.evaluator = CircuitEvaluator(self.freqs)

        # Re-préparation de tous les HPs sur la nouvelle grille
        for way in self.ways:
            self._prepare_driver(way)

        # Recalcul des constantes pour la fonction fitness
        self._cache_1e12 = 1e-12
        self._mask_power = self.freqs < 1000.0
        self.V_amp_test = 28.28 * np.where(self.freqs < 500.0, 1.0, 500.0 / self.freqs).astype(complex)

        # Configuration des masques
        if len(self.ways) == 2:
            self.mask_ref = [
                (self.freqs > 200) & (self.freqs < 1000),
                (self.freqs > 4000) & (self.freqs < 14000)
            ]
        elif len(self.ways) == 3:
            self.mask_ref = [
                (self.freqs > 100) & (self.freqs < 700),
                (self.freqs > 1000) & (self.freqs < 4000),
                (self.freqs > 4000) & (self.freqs < 13000)
            ]

        self.apply_wiring({w.label: 'parallel' for w in self.ways if getattr(w, 'count', 1) > 1})
        min_raw_avg = np.inf 
        for j in range(len(self.ways) - 1):
            raw_mag = np.abs(self.ways[j].driver.H_acoustic)
            raw_avg = np.mean(20 * np.log10(raw_mag[self.mask_ref[j]] + 1e-12)) if np.any(self.mask_ref[j]) else 85.0
            if raw_avg <= min_raw_avg: min_raw_avg = raw_avg
                
        self.target_spl = min_raw_avg
        self.apply_wiring({})

        max_raw_spl = np.zeros_like(self.freqs)
        for way in self.ways:
            max_raw_spl = np.maximum(max_raw_spl, 20 * np.log10(np.abs(way.driver.H_base) + 1e-12))
            
        valid_indices = np.where(max_raw_spl >= (self.target_spl - 10.0))[0]
        if len(valid_indices) > 0:
            f_min = max(self.freqs[min(valid_indices[0] + 5, len(self.freqs)-1)], 80.0)
            f_max = min(self.freqs[max(valid_indices[-1] - 5, 0)], 20000.0)
        else:
            f_min, f_max = 80, 20000
            
        if self.range_config.get("mode") == "manual":
            f_min = self.range_config.get("manual_min_hz", 100)
            f_max = self.range_config.get("manual_max_hz", 15000)
  
        self.mask_flat = (self.freqs >= f_min) & (self.freqs <= f_max)
        self._base_weight = np.zeros(len(self.freqs))
        self._base_weight[self.mask_flat] = 1.0

    def apply_wiring(self, wiring_dict):
        """Applique dynamiquement les modifications Z et H selon le câblage choisi."""
        for way in self.ways:
            d = way.driver
            count = getattr(way, 'count', 1)
            if count > 1:
                w_type = wiring_dict.get(way.label, 'parallel')
                if w_type == 'series':
                    d.Z_complex = d.Z_base * count
                    d.H_acoustic = d.H_base * 1.0  # La sensibilité en tension (2.83V) reste identique
                else: # parallel
                    d.Z_complex = d.Z_base / count
                    d.H_acoustic = d.H_base * count # +6dB par doublement à tension constante
            else:
                if hasattr(d, 'Z_base'):
                    d.Z_complex = d.Z_base
                    d.H_acoustic = d.H_base

    def fitness(self, individual, return_components=False):
        if not return_components and '_cached_score' in individual:
            return individual['_cached_score']
        root = individual['tree']
        
        
        # 1. Vérification des limites physiques
        for comp in root.get_all_nodes():
            if isinstance(comp, Resistor):
                comp.value = float(np.clip(comp.value, BOUNDS_R[0], BOUNDS_R[1]))
            elif isinstance(comp, Capacitor):
                comp.value = float(np.clip(comp.value, BOUNDS_C[0], BOUNDS_C[1]))
            elif isinstance(comp, Inductor):
                comp.value = float(np.clip(comp.value, BOUNDS_L[0], BOUNDS_L[1]))

        if not isinstance(root, ParallelNode): 
            return 1e9

        # 2. Détermination des combinaisons de câblage à tester (Brute Force)
        multi_ways = [w for w in self.ways if getattr(w, 'count', 1) > 1]
        if not multi_ways:
            combos = [{}]
        else:
            labels = [w.label for w in multi_ways]
            # options = [['series', 'parallel'] for _ in multi_ways]
            # combos = [dict(zip(labels, c)) for c in itertools.product(*options)]
            options = [['parallel'] for _ in multi_ways] 
            combos = [dict(zip(labels, c)) for c in itertools.product(*options)]

        best_final_score = float('inf')
        best_comps_track = None
        best_wiring = None

        # 3. Évaluation de chaque configuration
        for wiring in combos:
            # On applique les modifications Z et H virtuelles au DriverNode
            self.apply_wiring(wiring)
            
            res = self.evaluator.evaluate(root)
            
            p_sum_test = np.zeros_like(self.freqs, dtype=complex)
            p_ways = []
            
            for i, way in enumerate(self.ways):
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_ways.append(p_real)
                p_sum_test += p_real
                
            spl_sum_test = 20 * np.log10(np.abs(p_sum_test) + 1e-12)
            dynamic_spl = self.target_spl - self.spl_offset
            diff = spl_sum_test - dynamic_spl
            
            comps_track = {
                'MSE_SPL': 0.0,
                'FC_Penalty': 0.0,
                'Impedance_Penalty': 0.0,
                'Tweeter_LowFreq_Penalty': 0.0,
                'Woofer_HighFreq_Penalty': 0.0,
                'Woofer_Attenuation_Penalty': 0.0,
                'Thermal_Penalty': 0.0,
                'Component_Count_Penalty': 0.0,
                'Resistor_Count_Penalty': 0.0
            }
            
            dynamic_weight = self._base_weight.copy()
            detected_fc = []
            
            for j in range(len(self.ways) - 1):
                mag1 = 20 * np.log10(np.abs(p_ways[j]) + 1e-12)
                mag2 = 20 * np.log10(np.abs(p_ways[j+1]) + 1e-12)
                
                if len(self.ways) == 2:
                    search_mask = (self.freqs > 800) & (self.freqs < 5000)
                else:
                    if j == 0: search_mask = (self.freqs > 200) & (self.freqs < 3000)
                    elif j == 1: search_mask = (self.freqs > 1500) & (self.freqs < 12000)
                    else: search_mask = (self.freqs > 2000) & (self.freqs < 12000)
                
                if np.any(search_mask):
                    m1_sub = mag1[search_mask]
                    m2_sub = mag2[search_mask]
                    f_sub = self.freqs[search_mask]
                    
                    cross_points = np.where(m2_sub > m1_sub)[0]
                    if len(cross_points) > 0: idx_cross = cross_points[0]
                    else: idx_cross = np.argmin(np.abs(m1_sub - m2_sub))
                        
                    f_cross = f_sub[idx_cross]
                    detected_fc.append(f_cross)
                    dynamic_weight[(self.freqs > f_cross / 2.0) & (self.freqs < f_cross * 2.0)] = self.weights['crossover']

                    if self.target_fc is not None:
                    # Résolution de la cible : float (2-voies) ou tuple (3-voies)
                        if isinstance(self.target_fc, (tuple, list)):
                            target = self.target_fc[j] if j < len(self.target_fc) else 0.0
                        else:
                            target = self.target_fc if j == 0 else 0.0

                        if target > 0.0:
                            octave_err = np.log2(f_cross / target)
                            comps_track['FC_Penalty'] += (octave_err ** 2) * self.weights['fc_err']
                    
            raw_mse = np.mean(np.where(diff > 0, (diff**2)*2, diff**2) * dynamic_weight)
            comps_track['MSE_SPL'] = (raw_mse * self.weights['mse_sum'])
            
            # ==========================================
            # NOUVEAU : ANTI-ANNULATION DE PHASE (Overshoot Acoustique)
            # ==========================================
            comps_track['Acoustic_Overshoot_Penalty'] = 0.0
            for p_real in p_ways:
                way_spl = 20 * np.log10(np.abs(p_real) + 1e-12)
                # Aucun HP ne doit dépasser la cible globale de plus de 1.5 dB
                excess = np.maximum(0, way_spl - (self.target_spl + 1.5))
                if np.any(excess > 0):
                    # Pénalité très agressive (coefficient x20) pour interdire ce comportement
                    comps_track['Acoustic_Overshoot_Penalty'] += np.sum(excess ** 2) * self.weights.get('mse_sum', 1.0) * 10
            # ==========================================
            
            Z_in = self.evaluator.get_impedance(root)
            min_Z = np.min(np.abs(Z_in))
            if min_Z < 3: 
                comps_track['Impedance_Penalty'] += ((3 - min_Z) ** 3) * self.weights['impedance']
            
            # ==========================================
            # 1. SÉCURITÉ DU TWEETER (DYNAMIQUE)
            # ==========================================
            if len(self.ways) >= 3:
                twt_fc = detected_fc[1] if len(detected_fc) > 1 else 4000.0
            else:
                twt_fc = detected_fc[0] if len(detected_fc) > 0 else 2000.0
            twt_low_thresh = twt_fc * 0.75 
            last_way_v = res.get(self.ways[-1].label, {}).get("V_complex", np.zeros_like(self.freqs))
            v_low = np.abs(last_way_v)[self.freqs < twt_low_thresh]
            
            # Tolérance ultra-stricte : on autorise max 5% de tension (0.05) au lieu de 10%
            v_excess = np.maximum(0, v_low - 0.05)
            if np.any(v_excess > 0):
                comps_track['Tweeter_LowFreq_Penalty'] += np.sum(v_excess ** 2) * self.weights['tweeter_low']
            
            # ==========================================
            # Gestion du MID
            # ==========================================
            if len(self.ways) >= 3:
                comps_track['Midrange_LowFreq_Penalty']  = 0.0
                comps_track['Midrange_HighFreq_Penalty'] = 0.0
                comps_track['Midrange_Participation']    = 0.0

                # Déduction des seuils depuis target_fc si dispo
                if isinstance(self.target_fc, (tuple, list)) and len(self.target_fc) >= 2:
                    fc_low, fc_high = self.target_fc[0], self.target_fc[1]
                else:
                    # --- NOUVEAU : MODE AUTO INTELLIGENT ---
                    fc_low = detected_fc[0] if len(detected_fc) > 0 else 400.0
                    fc_high = detected_fc[1] if len(detected_fc) > 1 else 3000.0
                    
                    # On force un écartement naturel pour laisser vivre le médium.
                    # Si le tweeter croise trop près du woofer, on punit l'écrasement.
                    min_gap_ratio = 2.5 # Minimum 1.3 octaves d'écart (ex: 500Hz -> 1250Hz minimum)
                    if fc_high < fc_low * min_gap_ratio:
                        octave_squash = np.log2((fc_low * min_gap_ratio) / fc_high)
                        # On simule une erreur de fc pour forcer l'IA à écarter les drivers
                        comps_track['FC_Penalty'] += (octave_squash ** 2) * self.weights['fc_err'] * 3.0
                        
                        # On décale fc_high virtuellement pour garantir un bon test de participation
                        fc_high = fc_low * min_gap_ratio

                # On adapte parfaitement la zone de jeu aux fréquences trouvées
                mid_low_thresh  = fc_low  * 0.65   # Le médium doit se taire vite en bas
                mid_high_thresh = fc_high * 1.35   # Le médium doit se taire vite en haut 
                mid_band_lo     = fc_low  * 0.8    # Le médium doit jouer juste après fc_low
                mid_band_hi     = fc_high * 1.2    # Et s'arrêter juste avant fc_high

                for mid_idx in range(1, len(self.ways) - 1):
                    mid_v    = res.get(self.ways[mid_idx].label, {}).get("V_complex", np.zeros_like(self.freqs))
                    mid_vmag = np.abs(mid_v)

                    # --- Trop bas ---
                    v_low    = mid_vmag[self.freqs < mid_low_thresh]
                    excess_l = np.maximum(0, v_low - 0.15)
                    if np.any(excess_l > 0):
                        comps_track['Midrange_LowFreq_Penalty'] += (
                            np.sum(excess_l ** 2) * self.weights.get('midrange_low', 25.0)
                        )

                    # --- Trop haut ---
                    v_high   = mid_vmag[self.freqs > mid_high_thresh]
                    excess_h = np.maximum(0, v_high - 0.15)
                    if np.any(excess_h > 0):
                        comps_track['Midrange_HighFreq_Penalty'] += (
                            np.sum(excess_h ** 2) * self.weights.get('midrange_high', 25.0)
                        )
                        
                    # --- Participation OBLIGATOIRE dans la bande propre ---
                    mid_band_mask = (self.freqs >= mid_band_lo) & (self.freqs <= mid_band_hi)
                    v_in_band     = mid_vmag[mid_band_mask]
                    max_activity  = np.max(v_in_band) if len(v_in_band) > 0 else 0.0

                    MIN_ACTIVITY = 0.70  
                    if max_activity < MIN_ACTIVITY:
                        shortfall = MIN_ACTIVITY - max_activity
                        comps_track['Midrange_Participation'] += (
                            shortfall ** 2 * self.weights.get('midrange_participation', 80.0) * 3
                        )
                        
                    # CORRECTION : On utilise la pression acoustique (P_acoustic), pas la tension (V) !
                    mid_p_acoustic = res.get(self.ways[mid_idx].label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                    mid_spl = 20 * np.log10(np.abs(mid_p_acoustic[mid_band_mask]) + 1e-12)
                    sum_spl_in_band = 20 * np.log10(np.abs(np.abs(p_sum_test)[mid_band_mask]) + 1e-12)

                    if len(mid_spl) > 0 and len(sum_spl_in_band) > 0:
                        # Le médium doit être à moins de 6dB de la somme dans sa bande
                        mid_vs_sum_gap = np.mean(sum_spl_in_band) - np.mean(mid_spl)
                        if mid_vs_sum_gap > 6.0:
                            # On limite l'écart mathématique à 15dB pour ne pas faire exploser le gradient
                            gap_excess = min(mid_vs_sum_gap - 6.0, 15.0) 
                            comps_track['Midrange_Participation'] += gap_excess ** 2 * self.weights.get('midrange_participation', 80.0)

            # ========================================== # 2. GRADIENT DE SÉCURITÉ DU WOOFER # ========================================== 
            # ========================================== 
            # 2. GRADIENT DE SÉCURITÉ DU WOOFER (DYNAMIQUE)
            # ========================================== 
            woof_fc = detected_fc[0] if len(detected_fc) > 0 else 1000.0
            
            # Le woofer ne doit plus jouer au-dessus de 1.3x sa fréquence de coupure
            woof_high_thresh = woof_fc * 1.3 
            first_way_v = res.get(self.ways[0].label, {}).get("V_complex", np.zeros_like(self.freqs))
            v_high = np.abs(first_way_v)[self.freqs > woof_high_thresh]
            v_excess = np.maximum(0, v_high - 0.05)
            if np.any(v_excess > 0):
                comps_track['Woofer_HighFreq_Penalty'] += np.sum(v_excess ** 2) * self.weights['woofer_high']
                
            v_woofer = res.get(self.ways[0].label, {}).get("V_complex", np.zeros_like(self.freqs))
            max_w_gain = np.max(np.abs(v_woofer))
            if max_w_gain < 0.95:  
                comps_track['Woofer_Attenuation_Penalty'] += ((0.95 - max_w_gain) ** 3) * self.weights['woofer_attenuation']

            if len(self.ways) >= 3:
                for mid_idx in range(1, len(self.ways) - 1):
                    mid_v = res.get(self.ways[mid_idx].label, {}).get("V_complex", np.zeros_like(self.freqs))
                    # Dans la bande propre du médium, il ne doit pas être trop atténué
                    if len(detected_fc) >= 2:
                        mid_own_mask = (self.freqs >= detected_fc[0]) & (self.freqs <= detected_fc[1])
                        v_mid_inband = np.max(np.abs(mid_v)[mid_own_mask]) if np.any(mid_own_mask) else 0.0
                        if v_mid_inband < 0.80:
                            shortfall = 0.80 - v_mid_inband
                            comps_track['Midrange_Participation'] += (shortfall ** 2) * self.weights.get('midrange_attenuation', 100.0)


            # ==========================================
            # ÉVALUATION THERMIQUE ULTRA-RAPIDE (Sans récursion)
            # ==========================================
            max_res_power = 0.0
            tot_res_power = np.zeros_like(self.freqs)
            tot_drv_power = np.zeros_like(self.freqs)
            all_nodes = root.get_all_nodes()
            
            # On utilise le transfert électrique (node._V) calculé lors de l'évaluation acoustique
            for c in all_nodes: # 'all_nodes' est déjà dispo via root.get_all_nodes()
                t = type(c)
                if t is Resistor:
                    # Tension Réelle = Fonction de Transfert * Signal de Test de l'Ampli
                    v_real = c._V * self.V_amp_test
                    p_array = (np.abs(v_real)**2) / float(c.value)
                    
                    max_p = np.max(p_array)
                    if max_p > max_res_power: max_res_power = max_p
                    tot_res_power += p_array
                    
                elif t is DriverNode:
                    way = next(w for w in self.ways if w.label == c.label)
                    Z = way.driver.Z_complex if hasattr(way.driver, 'Z_complex') else np.full_like(self.freqs, 8.0)
                    
                    v_real = c._V * self.V_amp_test
                    tot_drv_power += (np.abs(v_real)**2) * np.real(1.0 / (Z + 1e-15))
            
            # Application des pénalités exactement comme avant
            if max_res_power > 20.0:
                comps_track['Thermal_Penalty'] += ((max_res_power - 20.0) ** 2) * self.weights['thermal']
                
            if np.any(self._mask_power):
                res_waste_ratio = tot_res_power[self._mask_power] / (tot_drv_power[self._mask_power] + self._cache_1e12)
                waste_excess = np.maximum(0, res_waste_ratio - 0.20)
                if np.any(waste_excess > 0):
                    wasted_watts = waste_excess * tot_drv_power[self._mask_power]
                    comps_track['Thermal_Penalty'] += np.sum(wasted_watts ** 2) * self.weights['thermal'] * 15.0
                        
            comps = [n for n in all_nodes if isinstance(n, ComponentNode)]
            n_comps = len(comps)

            if n_comps <= self.weights['n_comps']:
                comps_track['Component_Count_Penalty'] = 0.0
            else:
                excess = n_comps - self.weights['n_comps']
                comps_track['Component_Count_Penalty'] = (excess ** 2.5) * self.weights['components']

            n_resistors = sum(1 for c in comps if isinstance(c, Resistor))
            comps_track['Resistor_Count_Penalty'] = n_resistors * self.weights['resistors']
            for c in comps:
                if isinstance(c, Resistor) and c.value > 12.0:
                    comps_track['Resistor_Count_Penalty'] += (c.value - 12.0) * self.weights.get('resistors', 0.4)


            final_score = sum(comps_track.values())
            
            # 4. On garde le meilleur câblage
            if final_score < best_final_score:
                best_final_score = final_score
                best_comps_track = comps_track
                best_wiring = wiring

        # 5. On restaure le meilleur câblage dans l'arbre physique pour l'individu
        if best_wiring is not None:
            individual['wiring'] = best_wiring
            self.apply_wiring(best_wiring)
        else:
            self.apply_wiring({})

        individual['_cached_score'] = best_final_score
        if return_components:
            return best_final_score, best_comps_track
            
        return best_final_score


    def _elite_worker(self, args):
        ind, max_opt, snap = args
        
        if not snap and ind.get('is_optimized', False):
            return (self.fitness(ind), ind)
            
        if not snap:
            self.optimize_values(ind, max_iter=max_opt)
            ind['is_optimized'] = True
        else:
            for comp in ind['tree'].get_all_nodes():
                if isinstance(comp, ComponentNode):
                    comp.value = CATALOG.snap_to_catalog(comp.value, CATALOG.get_comp_type(comp))
            self.optimize_catalog_values(ind)
            ind['is_optimized'] = False
            
        return (self.fitness(ind), ind)

    def _lamarckian_worker(self, child_ind):
        if random.random() < 0.40:
            self.optimize_values(child_ind, max_iter=8)
            child_ind['is_optimized'] = True
        return child_ind

    def optimize_catalog_values(self, individual):
        individual.pop('_cached_score', None)
        comps = [n for n in individual['tree'].get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        best_score = self.fitness(individual)
        improved = True
        
        while improved:
            improved = False
            for comp in comps:
                ctype = CATALOG.get_comp_type(comp)
                original_val = CATALOG.snap_to_catalog(comp.value, ctype)
                
                if ctype == 'C': arr = CATALOG.vals_c
                elif ctype == 'L': arr = CATALOG.vals_l
                else: arr = CATALOG.vals_r
                
                idx = np.abs(arr - original_val).argmin()
                best_comp_val = original_val
                
                for step in [-1, 1]:
                    new_idx = idx + step
                    if 0 <= new_idx < len(arr):
                        test_val = arr[new_idx]
                        comp.value = test_val
                        
                        individual.pop('_cached_score', None)
                        new_score = self.fitness(individual)
                        if new_score < best_score:
                            best_score = new_score
                            best_comp_val = test_val
                            improved = True
                
                comp.value = best_comp_val
                individual.pop('_cached_score', None)
                
        return individual

    def optimize_values(self, individual, max_iter=5):
        individual.pop('_cached_score', None)

        root = individual['tree']
        comps = [n for n in root.get_all_nodes() if isinstance(n, ComponentNode)]
        if not comps: return individual
        
        init = [np.log10(np.clip(c.value, 1e-12, 1e2)) for c in comps]
        bounds = [(np.log10(BOUNDS_R[0]), np.log10(BOUNDS_R[1])) if isinstance(c, Resistor) else 
                  (np.log10(BOUNDS_C[0]), np.log10(BOUNDS_C[1])) if isinstance(c, Capacitor) else 
                  (np.log10(BOUNDS_L[0]), np.log10(BOUNDS_L[1])) for c in comps]
        
                  
        def obj(x_log):
            for i, v in enumerate(x_log): comps[i].value = 10**v
            individual.pop('_cached_score', None)
            return self.fitness(individual)
            
        res = minimize(
            obj, init, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={
                'maxiter': max_iter,
                'ftol': 1e-4,
                'eps': 1e-3
            }
        )
        for i, v in enumerate(res.x): comps[i].value = 10**v
        return individual

    def run(self, generations=50, pop_size=60, checkpoint_path=None):
        population = []
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f: data = json.load(f)
                tree = Node.from_dict(data["tree"])
                for n in tree.get_all_nodes():
                    if isinstance(n, DriverNode):
                        way = next(w for w in self.ways if w.label == n.label)
                        n.H_acoustic, n.Z_complex = way.driver.H_acoustic, way.driver.Z_complex
                
                population.append({'tree': tree})
                print(f"[+] Champion chargé depuis {checkpoint_path}")
            except Exception as e: 
                print(f"Erreur chargement du checkpoint: {e}")
        else:
            if len(self.ways) == 2:
                try:
                    seeds = []
                    
                    w_drv = lambda: self.ways[0].driver.copy()
                    t_drv = lambda: self.ways[1].driver.copy()

                    w1 = SeriesNode(Inductor(1.0e-3), w_drv())
                    t1 = SeriesNode(Capacitor(4.7e-6), t_drv())
                    seeds.append(ParallelNode(w1, t1))

                    w2 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(10e-6), w_drv()))
                    t2 = SeriesNode(Capacitor(5.6e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w2, t2))

                    w3 = SeriesNode(Inductor(1.5e-3), ParallelNode(Capacitor(10e-6), SeriesNode(Inductor(0.5e-3), w_drv())))
                    t3 = SeriesNode(Capacitor(4.7e-6), ParallelNode(Inductor(0.3e-3), SeriesNode(Capacitor(10e-6), t_drv())))
                    seeds.append(ParallelNode(w3, t3))

                    notch = ParallelNode(Capacitor(15e-6), Inductor(0.1e-3))
                    w4 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(8.2e-6), SeriesNode(notch, w_drv())))
                    t4 = SeriesNode(Capacitor(5.6e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w4, t4))

                    lpad_t = SeriesNode(Resistor(3.3), ParallelNode(Resistor(10.0), t_drv()))
                    w5 = SeriesNode(Inductor(1.2e-3), ParallelNode(Capacitor(10e-6), w_drv()))
                    t5 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.4e-3), lpad_t))
                    seeds.append(ParallelNode(w5, t5))

                    print(f"[+] Injection de {len(seeds)} templates fondamentaux dans la population.")
                    for s in seeds:
                        population.append({'tree': s, 'is_optimized': False})
                        for _ in range(10):
                            mutated_s = self.mutator.mutate(s.copy())
                            population.append({'tree': mutated_s, 'is_optimized': False})
                            
                    print(f"[+] {len(seeds) * 10} mutants de première génération créés avec succès.")
                            
                except Exception as e:
                    print(f"[-] Erreur lors de l'injection des graines : {e}")
                    pass
            elif len(self.ways) == 3:
                try:
                    seeds = []
                    w_drv = lambda: self.ways[0].driver.copy()
                    m_drv = lambda: self.ways[1].driver.copy()
                    t_drv = lambda: self.ways[2].driver.copy()

                    # Template 1 : 2ème ordre sur chaque voie
                    # Woofer LP : L série + C shunt
                    # Médium BP : C série (coupe grave) + L shunt (coupe aigu)
                    # Tweeter HP : C série + L shunt
                    w1 = SeriesNode(Inductor(2.5e-3), ParallelNode(Capacitor(33e-6), w_drv()))
                    m1 = SeriesNode(Capacitor(22e-6), ParallelNode(Inductor(0.5e-3), m_drv()))
                    t1 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w1, ParallelNode(m1, t1)))

                    # Template 2 : 3ème ordre woofer, 2ème ordre médium/tweeter
                    w2 = SeriesNode(Inductor(2.0e-3), ParallelNode(Capacitor(22e-6), SeriesNode(Inductor(1.0e-3), w_drv())))
                    m2 = SeriesNode(Capacitor(15e-6), ParallelNode(Inductor(0.6e-3), SeriesNode(Capacitor(10e-6), m_drv())))
                    t2 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.4e-3), t_drv()))
                    seeds.append(ParallelNode(w2, ParallelNode(m2, t2)))

                    # Template 3 : Filtre bouchon sur le woofer (correction résonance)
                    notch = ParallelNode(Capacitor(33e-6), Inductor(0.1e-3))
                    w3 = SeriesNode(Inductor(2.5e-3), ParallelNode(Capacitor(33e-6), SeriesNode(notch, w_drv())))
                    m3 = SeriesNode(Capacitor(22e-6), ParallelNode(Inductor(0.5e-3), m_drv()))
                    t3 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.3e-3), t_drv()))
                    seeds.append(ParallelNode(w3, ParallelNode(m3, t3)))

                    # Template 4 : L-Pad sur le médium (atténuation si sensibilité trop haute)
                    lpad_m = SeriesNode(Resistor(2.2), ParallelNode(Resistor(15.0), m_drv()))
                    w4 = SeriesNode(Inductor(2.5e-3), ParallelNode(Capacitor(33e-6), w_drv()))
                    m4 = SeriesNode(Capacitor(22e-6), ParallelNode(Inductor(0.5e-3), lpad_m))
                    t4 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.3e-3), t_drv()))
                    seeds.append(ParallelNode(w4, ParallelNode(m4, t4)))

                    # Template 5 : L-Pad tweeter + Zobel médium
                    zobel_m = SeriesNode(Resistor(6.8), Capacitor(10e-6))
                    lpad_t = SeriesNode(Resistor(3.3), ParallelNode(Resistor(10.0), t_drv()))
                    w5 = SeriesNode(Inductor(2.5e-3), ParallelNode(Capacitor(33e-6), w_drv()))
                    m5 = SeriesNode(Capacitor(22e-6), ParallelNode(Inductor(0.5e-3), ParallelNode(zobel_m, m_drv())))
                    t5 = SeriesNode(Capacitor(6.8e-6), ParallelNode(Inductor(0.3e-3), lpad_t))
                    seeds.append(ParallelNode(w5, ParallelNode(m5, t5)))

                    print(f"[+] Injection de {len(seeds)} templates 3-voies dans la population.")
                    for s in seeds:
                        population.append({'tree': s, 'is_optimized': False})
                        for _ in range(10):
                            mutated_s = self.mutator.mutate(s.copy())
                            population.append({'tree': mutated_s, 'is_optimized': False})
                    print(f"[+] {len(seeds) * 10} mutants de première génération créés.")

                except Exception as e:
                    print(f"[-] Erreur injection seeds 3-voies : {e}")

        while len(population) < pop_size:
            branches = []
            for way in self.ways:
                branches.append(self.mutator.generate_random_tree(way.driver.copy(), max_depth=2))
            root = branches[0]
            for b in branches[1:]:
                root = ParallelNode(root, b)
            population.append({'tree': root})

        best_score = float('inf')
        best_ind = population[0]
        current_res = len(self.freqs)

        n_workers = cpu_count()
        chunksize = max(1, pop_size // (n_workers * 4))

        with Pool(processes=n_workers, initializer=_pool_init, initargs=(self,)) as pool:
            for gen in range(generations):
                
                if gen < int(generations * 0.4):      
                    max_opt_iter = 5
                    target_res = 1000
                    snap_to_standard = False
                elif gen < int(generations * 0.8):    
                    max_opt_iter = 12
                    target_res = 1000
                    snap_to_standard = False
                else:                                 
                    max_opt_iter = 20
                    target_res = 1000
                    snap_to_standard = True
                    
                if gen == int(generations * 0.8):
                    print("Passage en PHASE 3 (Standardisation CATALOGUE)")
                    best_score = float('inf') 
                    
                if target_res != current_res:
                    print(f"\n[!] Transition de phase (Génération {gen}/{generations})")
                    # 1. On ferme les anciens processus de calcul
                    pool.close()
                    pool.join()
                    
                    # 2. On met à jour le coeur de calcul principal
                    self._set_resolution(target_res)
                    current_res = target_res
                    
                    # 3. CRUCIAL: On efface les vieux scores en cache car l'échelle de notation a changé !
                    for ind in population:
                        ind.pop('_cached_score', None)
                        # --- NOUVEAU : On met à jour la résolution des HP dans les arbres ---
                        for node in ind['tree'].get_all_nodes():
                            if isinstance(node, DriverNode):
                                way = next((w for w in self.ways if w.label == node.label), None)
                                if way:
                                    node.H_acoustic = way.driver.H_acoustic.copy()
                                    node.Z_complex = way.driver.Z_complex.copy()
                        
                    best_score = float('inf') # On reset le meilleur score pour la nouvelle échelle
                    
                    # 4. On relance des processus tout neufs
                    pool = Pool(processes=n_workers, initializer=_pool_init, initargs=(self,))
                # -------------------------------------------------------------
                
                fitness_results = pool.map(_pool_fitness, population, chunksize=chunksize)
                
                scores = []
                for (fit, wiring), ind in zip(fitness_results, population):
                    if wiring is not None:
                        ind['wiring'] = wiring
                    scores.append((fit, ind))
                    
                scores.sort(key=lambda x: x[0])
                
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                
                top_10_count = max(1, pop_size // 10)
                top_inds = [s[1] for s in scores[:top_10_count]]
                
                gen_comps = []
                for ind in top_inds:
                    _, comps = self.fitness(ind, return_components=True)
                    gen_comps.append(comps)
                
                avg_comps = {k: np.mean([c[k] for c in gen_comps]) for k in gen_comps[0].keys()}
                self.loss_history.append(avg_comps)
                

                elite_count = max(2, pop_size // 10)
                elite_args = [(scores[i][1], max_opt_iter, snap_to_standard) for i in range(elite_count)]
                optimized_elites = pool.map(_pool_elite, elite_args, chunksize=1)
                
                for i in range(elite_count):
                    scores[i] = optimized_elites[i]
                scores.sort(key=lambda x: x[0])
                
                if scores[0][0] < best_score:
                    best_score = scores[0][0]
                    best_ind = scores[0][1]
                    
                    if checkpoint_path:
                        save_tree = best_ind['tree'].copy()
                        for comp in save_tree.get_all_nodes():
                            if isinstance(comp, ComponentNode):
                                comp.value = CATALOG.snap_to_catalog(comp.value, CATALOG.get_comp_type(comp))
                        with open(checkpoint_path, "w") as f:
                            json.dump({
                                "tree": save_tree.to_dict(), 
                            }, f, indent=4)

                new_pop = [best_ind]
                for i in range(1, elite_count):
                    new_pop.append(scores[i][1])
                    
                raw_children = []
                while len(new_pop) + len(raw_children) < pop_size:
                    def tournament():
                        competitors = random.sample(scores, 3)
                        return min(competitors, key=lambda x: x[0])[1]

                    parent1 = tournament()
                    if random.random() < 0.30:
                        parent2 = tournament()
                        child_tree = self.mutator.crossover(parent1['tree'], parent2['tree'])
                    else:
                        child_tree = self.mutator.mutate(parent1['tree'].copy())

                    raw_children.append({'tree': child_tree, 'is_optimized': False})
                
                if gen < int(generations * 0.9):
                    trained_children = pool.map(_pool_lamarckian, raw_children, chunksize=chunksize)
                    new_pop.extend(trained_children)
                else:
                    new_pop.extend(raw_children)
                
                population = new_pop

        if not snap_to_standard:
            self.optimize_values(best_ind, max_iter=150)
        else:
            self.optimize_catalog_values(best_ind)
            
        for comp in best_ind['tree'].get_all_nodes():
            if isinstance(comp, ComponentNode):
                comp.value = CATALOG.snap_to_catalog(comp.value, CATALOG.get_comp_type(comp))
            elif isinstance(comp, DriverNode):
                way = next(w for w in self.ways if w.label == comp.label)
                comp.model_name = way.driver.model_name

        return best_ind

    def plot_result(self, individual, filename_response="response.png", filename_filter="filter.png"):
        self.apply_wiring(individual.get('wiring', {})) # NOUVEAU
        
        root = individual['tree']
        res = self.evaluator.evaluate(root)
        
        plt.figure(figsize=(12, 7))
        p_sum = np.zeros_like(self.freqs, dtype=complex)
        
        for way in self.ways:
            p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
            p_sum += p_real
            spl_real = 20 * np.log10(np.abs(p_real) + 1e-12)
            plt.semilogx(self.freqs, spl_real, label=way.driver.model_name, linewidth=2)

        spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
        plt.semilogx(self.freqs, spl_sum, label="System Sum", color='red', linewidth=3)
        plt.axhline(self.target_spl, color='green', linestyle='--', alpha=0.5, label="Target SPL")
            
        mask_range = (self.freqs >= 300) & (self.freqs <= 17000)
        if np.any(mask_range):
            spl_in_range = spl_sum[mask_range]
            spl_diff = np.max(spl_in_range) - np.min(spl_in_range)
            plt.text(0.02, 0.95, f"Ripple (300Hz-17kHz): {spl_diff:.1f} dB", 
                     transform=plt.gca().transAxes, 
                     fontsize=11, fontweight='bold', color='black',
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
                     
        plt.title(f"System SPL Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.xlim(20, 20000)
        plt.ylim(self.target_spl - 30, self.target_spl + 10)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.savefig(filename_response)
        plt.close()

        plt.figure(figsize=(12, 8))
        for way in self.ways:
            v_complex = res.get(way.label, {}).get("V_complex", np.zeros_like(self.freqs))
            filter_db = 20 * np.log10(np.abs(v_complex) + 1e-12)
            plt.semilogx(self.freqs, filter_db, label=f"{way.driver.model_name} Filter", linewidth=2)
            
        plt.title("Electrical Transfer Functions")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.xlim(20, 20000)
        plt.ylim(-40, 5)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.savefig(filename_filter)
        plt.close()

    def plot_directivity(self, individual, filename="directivity.png"):
        self.apply_wiring(individual.get('wiring', {}))
        
        plt.figure(figsize=(12, 7))
        root = individual['tree']
        angles = ['0deg', '15deg', '30deg', '45deg', '60deg']
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        alphas = [1.0, 0.8, 0.6, 0.5, 0.4]
        labels = ['0° (On-Axis)', '15°', '30°', '45°', '60°']
        
        original_H = {way.label: way.driver.H_acoustic.copy() for way in self.ways}
        
        for idx, angle in enumerate(angles):
            valid_angle = True
            
            # Masque pour mémoriser les fréquences couvertes par au moins 1 fichier FRD
            system_valid_mask = np.zeros_like(self.freqs, dtype=bool)
            
            for way in self.ways:
                # CORRECTION 1 : On lit le fichier même pour 0deg (finit les courbes plates !)
                new_H = self._get_off_axis_H(way, angle)
                if new_H is None:
                    valid_angle = False
                    break
                    
                # On extrait les limites réelles du fichier pour notre masque visuel
                path = way.frd_path.replace('0deg', angle)
                frd_data = np.loadtxt(path)
                f_min, f_max = frd_data[0, 0], frd_data[-1, 0]
                system_valid_mask |= (self.freqs >= f_min) & (self.freqs <= f_max)

                for node in root.get_all_nodes():
                    if isinstance(node, DriverNode) and node.label == way.label:
                        node.H_acoustic = new_H
                        
            if not valid_angle: 
                continue 
                
            res = self.evaluator.evaluate(root)
            p_sum = np.zeros_like(self.freqs, dtype=complex)
            for way in self.ways:
                p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                p_sum += p_real
                
            spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
            
            # CORRECTION 2 : On cache les extrémités de la courbe juste avant de dessiner
            spl_sum[~system_valid_mask] = np.nan
            
            plt.semilogx(self.freqs, spl_sum, label=labels[idx], color=colors[idx], linewidth=3 if idx==0 else 2, alpha=alphas[idx])
            
        for way in self.ways:
            for node in root.get_all_nodes():
                if isinstance(node, DriverNode) and node.label == way.label:
                    node.H_acoustic = original_H[way.label]
                    
        plt.axhline(self.target_spl, color='black', linestyle='--', alpha=0.3, label="Target SPL")
        plt.ylim(self.target_spl - 30, self.target_spl + 10)
        plt.xlim(20, 20000)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.title("Off-Axis SPL Directivity")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.savefig(filename)
        plt.close()

    def draw_schematic(self, individual, filename="schematic.png"):
        # On récupère l'arbre et on lui injecte le câblage trouvé
        root = individual['tree']
        root.wiring = individual.get('wiring', {})
        
        # INJECTION DE LA QUANTITÉ pour SchematicRenderer
        for node in root.get_all_nodes():
            if isinstance(node, DriverNode):
                way = next((w for w in self.ways if w.label == node.label), None)
                if way:
                    node.count = getattr(way, 'count', 1)
                    
        renderer = SchematicRenderer(root)
        renderer.save(filename)
        
    def _get_off_axis_H(self, way, angle):
        off_axis_path = way.frd_path.replace('0deg', angle)
        if not os.path.exists(off_axis_path): return None
            
        try:
            frd_data = np.loadtxt(off_axis_path)
            frd_freqs = frd_data[:, 0]
            mag_db = frd_data[:, 1]
            ph_unwrapped = np.unwrap(np.deg2rad(frd_data[:, 2]))
            
            # CORRECTION : On utilise -200 dB (silence absolu) aux extrémités !
            # Cela permet aux autres haut-parleurs de continuer à jouer normalement.
            mag_interp = np.interp(self.freqs, frd_freqs, mag_db, left=-200, right=-200)
            ph_interp = np.interp(self.freqs, frd_freqs, ph_unwrapped, left=0, right=0)
            H_acoustic = (10 ** (mag_interp / 20)) * np.exp(1j * ph_interp)
            
            x_mm = getattr(way, 'x_offset', 0.0)
            y_mm = getattr(way, 'y_offset', 0.0)
            z_mm = getattr(way, 'z_offset', 0.0)
            listen_dist_mm = 2000.0
            
            dist_to_mic_mm = np.sqrt(x_mm**2 + y_mm**2 + (listen_dist_mm - z_mm)**2)
            path_diff_m = (dist_to_mic_mm - listen_dist_mm) / 1000.0
            delay_s = path_diff_m / 343.0
            
            phase_delay = np.exp(-1j * 2 * np.pi * self.freqs * delay_s)
            H_acoustic *= phase_delay
            
            count = getattr(way, 'count', 1)
            w_type = getattr(way.driver, 'current_wiring', 'parallel')
            if count > 1 and w_type == 'parallel':
                H_acoustic *= count
                
            return H_acoustic
        except Exception as e:
            return None
        
    def plot_sonogram(self, individual, filename="sonogram.png"):
        self.apply_wiring(individual.get('wiring', {}))
        
        import matplotlib.ticker as ticker
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.interpolate import RectBivariateSpline
        import numpy as np
        import os
        
        root = individual['tree']
        test_angles = [0, 15, 30, 45, 60]
        valid_data = {}
        original_H = {way.label: way.driver.H_acoustic.copy() for way in self.ways}
        
        # --- NOUVEAU : Traqueurs de zone valide globale ---
        angle_f_mins = []
        angle_f_maxs = []
        
        for angle in test_angles:
            angle_str = f"{angle}deg"
            valid_angle = True
            
            sys_f_min = float('inf')
            sys_f_max = 0.0
            
            for way in self.ways:
                new_H = original_H[way.label] if angle == 0 else self._get_off_axis_H(way, angle_str)
                if new_H is None:
                    valid_angle = False
                    break
                
                # Lecture des vraies limites du fichier FRD
                path = way.frd_path if angle == 0 else way.frd_path.replace('0deg', angle_str)
                try:
                    frd_data = np.loadtxt(path)
                    sys_f_min = min(sys_f_min, frd_data[0, 0])
                    sys_f_max = max(sys_f_max, frd_data[-1, 0])
                except:
                    pass

                for node in root.get_all_nodes():
                    if isinstance(node, DriverNode) and node.label == way.label:
                        node.H_acoustic = new_H
                        
            if valid_angle:
                angle_f_mins.append(sys_f_min)
                angle_f_maxs.append(sys_f_max)
                
                res = self.evaluator.evaluate(root)
                p_sum = np.zeros_like(self.freqs, dtype=complex)
                for way in self.ways:
                    p_real = res.get(way.label, {}).get("P_acoustic", np.zeros_like(self.freqs))
                    p_sum += p_real
                spl_sum = 20 * np.log10(np.abs(p_sum) + 1e-12)
                valid_data[angle] = spl_sum
                
        for way in self.ways:
            for node in root.get_all_nodes():
                if isinstance(node, DriverNode) and node.label == way.label:
                    node.H_acoustic = original_H[way.label]
                    
        if len(valid_data) < 2: return

        # === DÉTERMINATION DE LA FENÊTRE VISUELLE PARFAITE ===
        # On recadre le graphique à l'intersection des zones valides de tous les angles.
        # Si une mesure s'arrête à 300Hz, le sonogramme ne descendra pas plus bas !
        plot_f_min = max(angle_f_mins) if angle_f_mins else 100.0
        plot_f_max = min(angle_f_maxs) if angle_f_maxs else 20000.0
        
        # On limite tout de même visuellement pour ne pas montrer l'infra-grave inutile
        if plot_f_min < 100.0: plot_f_min = 100.0
        if plot_f_max > 20000.0: plot_f_max = 20000.0

        angles_raw = []
        spl_raw = []
        for angle in sorted(valid_data.keys(), reverse=True):
            if angle != 0:
                angles_raw.append(-angle)
                spl_raw.append(valid_data[angle])
        for angle in sorted(valid_data.keys()):
            angles_raw.append(angle)
            spl_raw.append(valid_data[angle])
            
        angles_raw = np.array(angles_raw)
        spl_matrix = np.array(spl_raw)

        spl_matrix = np.nan_to_num(spl_matrix, nan=(self.target_spl - 40))

        spline = RectBivariateSpline(angles_raw, self.freqs, spl_matrix, kx=2, ky=3)
        angles_hd = np.linspace(angles_raw[0], angles_raw[-1], 500)
        
        # === RECADRAGE DU RENDU HAUTE-DÉFINITION ===
        freqs_hd = np.geomspace(plot_f_min, plot_f_max, 1000)
        spl_hd = spline(angles_hd, freqs_hd)

        vituix_colors = [
            (0.00, '#000000'), (0.14, '#000088'), (0.28, '#0000FF'), (0.42, '#00FFFF'),
            (0.57, '#00FF00'), (0.71, '#FFFF00'), (0.85, '#FF0000'), (1.00, '#550000')
        ]
        vituix_cmap = LinearSegmentedColormap.from_list('vituix_pro', vituix_colors, N=512)

        max_spl = self.target_spl + 6
        min_spl = max_spl - 40
        
        fig, ax = plt.subplots(figsize=(12, 6))
        X, Y = np.meshgrid(freqs_hd, angles_hd)
        c = ax.contourf(X, Y, spl_hd, levels=np.linspace(min_spl, max_spl, 200), cmap=vituix_cmap, extend='both', antialiased=True)
        levels_3db = np.arange(int(min_spl), int(max_spl) + 1, 3)
        ax.contour(X, Y, spl_hd, levels=levels_3db, colors='black', linewidths=0.5, alpha=0.7, antialiased=True)

        ax.set_xscale('log')
        
        # === RECADRAGE DE L'AXE X ===
        ax.set_xlim(plot_f_min, plot_f_max)
        ax.set_ylim(angles_raw[0], angles_raw[-1])
        
        # Masquage des repères (ticks) qui seraient désormais en dehors de l'image
        ticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        valid_ticks = [t for t in ticks if plot_f_min <= t <= plot_f_max]
        ax.set_xticks(valid_ticks)
        
        def format_freq(x, pos):
            if x == 100: return '100Hz'
            elif x >= 1000: return f'{int(x/1000)}k'
            else: return f'{int(x)}'
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_freq))
        
        ax.set_yticks(angles_raw)
        ax.yaxis.tick_right() 
        ax.set_ylabel('deg', loc='top', rotation=0, labelpad=-20)
        ax.grid(True, which='major', color='white', alpha=0.3, linewidth=0.5)
        ax.grid(True, which='minor', color='white', alpha=0.1, linewidth=0.3)
        ax.set_title('Directivity (hor)', pad=10)

        plt.subplots_adjust(left=0.15, right=0.95) 
        cbar_ax = fig.add_axes([0.05, 0.15, 0.02, 0.7]) 
        cbar = fig.colorbar(c, cax=cbar_ax, ticks=np.arange(int(min_spl), int(max_spl), 8))
        cbar.ax.set_title('dB', pad=10)
        cbar.ax.yaxis.set_ticks_position('left')
        plt.savefig(filename, dpi=200, bbox_inches='tight') 
        plt.close()
        
    def plot_loss_history(self, filename="loss_history.png"):
        if not hasattr(self, 'loss_history') or not self.loss_history: return
        generations = np.arange(len(self.loss_history))
        keys = list(self.loss_history[0].keys())
        plt.figure(figsize=(12, 7))
        bottom = np.zeros(len(generations))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, key in enumerate(keys):
            values = np.array([gen[key] for gen in self.loss_history])
            plt.bar(generations, values, bottom=bottom, width=1.0, label=key.replace('_', ' '), color=colors[i % len(colors)], edgecolor='none')
            bottom += values
            
        plt.title("Evolution of Loss Components (Top 10% Average)", fontsize=14, fontweight='bold')
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Absolute Loss Score", fontsize=12)
        
        focus_idx = max(0, int(len(generations) * 0.15)) 
        if len(bottom) > focus_idx:
            max_y = np.max(bottom[focus_idx:]) * 1.5 
            plt.ylim(0, max_y)
        
        plt.xlim(0, len(generations) - 1)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _calc_node_impedance(self, node):
        name = node.__class__.__name__
        if name == "Resistor":
            return np.full_like(self.freqs, node.value, dtype=complex)
        elif name == "Capacitor":
            return 1.0 / (1j * 2 * np.pi * self.freqs * node.value + 1e-15)
        elif name == "Inductor":
            return 1j * 2 * np.pi * self.freqs * node.value
        elif name == "DriverNode":
            for way in self.ways:
                if way.label == node.label:
                    if hasattr(way.driver, 'Z_complex'): return way.driver.Z_complex
                    if hasattr(way.driver, 'Z'): return way.driver.Z
            return np.full_like(self.freqs, 8.0, dtype=complex)
        elif name == "SeriesNode":
            return self._calc_node_impedance(node.left) + self._calc_node_impedance(node.right)
        elif name == "ParallelNode":
            z1 = self._calc_node_impedance(node.left)
            z2 = self._calc_node_impedance(node.right)
            return 1.0 / (1.0 / (z1 + 1e-15) + 1.0 / (z2 + 1e-15))
        return np.full_like(self.freqs, 1e6, dtype=complex)
    
    def _get_power_dissipation_stats(self, node, V_in):
        """
        Calcule la répartition de la puissance active dans le circuit.
        Retourne : (max_resistor_power, array_total_resistor_power, array_total_driver_power)
        """
        name = node.__class__.__name__
        
        if name == "Resistor":
            # Puissance active d'une résistance : P = |V|^2 / R
            p_array = (np.abs(V_in)**2) / node.value
            return np.max(p_array), p_array, np.zeros_like(self.freqs)
            
        elif name == "DriverNode":
            # Puissance active réelle livrée au HP : P = |V|^2 * Re(Y) où Y = 1/Z
            for way in self.ways:
                if way.label == node.label:
                    Z = way.driver.Z_complex if hasattr(way.driver, 'Z_complex') else np.full_like(self.freqs, 8.0)
                    p_array = (np.abs(V_in)**2) * np.real(1.0 / (Z + 1e-15))
                    return 0.0, np.zeros_like(self.freqs), p_array
            return 0.0, np.zeros_like(self.freqs), np.zeros_like(self.freqs)
            
        elif name in ["Capacitor", "Inductor"]:
            # Les composants réactifs purs ne dissipent pas de puissance active (chaleur)
            return 0.0, np.zeros_like(self.freqs), np.zeros_like(self.freqs)
            
        elif name == "ParallelNode":
            max_l, tot_r_l, tot_d_l = self._get_power_dissipation_stats(node.left, V_in)
            max_r, tot_r_r, tot_d_r = self._get_power_dissipation_stats(node.right, V_in)
            return max(max_l, max_r), tot_r_l + tot_r_r, tot_d_l + tot_d_r
            
        elif name == "SeriesNode":
            z_left = self._calc_node_impedance(node.left)
            z_right = self._calc_node_impedance(node.right)
            z_tot = z_left + z_right + 1e-15
            
            # Diviseur de tension complexe
            v_left = V_in * (z_left / z_tot)
            v_right = V_in * (z_right / z_tot)
            
            max_l, tot_r_l, tot_d_l = self._get_power_dissipation_stats(node.left, v_left)
            max_r, tot_r_r, tot_d_r = self._get_power_dissipation_stats(node.right, v_right)
            return max(max_l, max_r), tot_r_l + tot_r_r, tot_d_l + tot_d_r
            
        return 0.0, np.zeros_like(self.freqs), np.zeros_like(self.freqs)

    def plot_impedance(self, individual, filename="impedance.png"):
        # Plot impedance in log scale
        self.apply_wiring(individual.get('wiring', {})) # NOUVEAU
        
        import matplotlib.ticker as ticker
        root = individual['tree']
        Z_in = self._calc_node_impedance(root)
        mag_Z = np.abs(Z_in)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color1 = '#0077BB' 
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Impedance (Ω)', color=color1, fontweight='bold')
        ax1.semilogx(self.freqs, mag_Z, color=color1, linewidth=2.5, label="Magnitude")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim(20, 20000)
        
        y_max = min(60, max(20, np.max(mag_Z) * 1.1))
        ax1.set_yscale('log')
        # Horizontal line at 3ohm
        ax1.axhline(3, color='red', linestyle='--', alpha=0.7, label="3Ω Reference")
        
        def format_freq(x, pos):
            if x == 100: return '100Hz'
            elif x >= 1000: return f'{int(x/1000)}k'
            else: return f'{int(x)}'
        ax1.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_freq))
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        ax1.legend(lines_1, labels_1, loc='upper right')
            
        plt.title(f"System Impedance")
        fig.tight_layout()  
        plt.savefig(filename)
        plt.close()
        
    def generate_parts_list(self, individual, filename="BOM_Parts_List.csv"):
        """Génère la nomenclature finale au format CSV et LaTeX."""
        comps = [n for n in individual['tree'].get_all_nodes() if isinstance(n, ComponentNode)]
        
        inventory = {}
        total_price = 0.0
        
        # ==========================================
        # 1. GÉNÉRATION DU CSV ET INVENTAIRE GLOBAL
        # ==========================================
        for comp in comps:
            ctype = CATALOG.get_comp_type(comp)
            val_cat = CATALOG.snap_to_catalog(comp.value, ctype)
            comp.value = val_cat 
            part_info = CATALOG.get_part_info(val_cat, ctype)
            part_num = part_info['PartNumber']
            
            if part_num not in inventory:
                inventory[part_num] = {
                    'Qty': 0, 'Description': part_info['Description'],
                    'Value': part_info['Value'], 'Type': ctype,
                    'Price': part_info['Price'], 'URL': part_info['URL']
                }
            inventory[part_num]['Qty'] += 1

        print("\n" + "="*60)
        print("🛒 CROSSOVER BILL OF MATERIALS (BOM)")
        print("="*60)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Part Number', 'Quantity', 'Value', 'Unit', 'Component', 'Description', 'Unit Price ($)', 'Line Total ($)', 'URL'])
            
            for part_num, data in inventory.items():
                qty = data['Qty']
                unit_price = data['Price'] if pd.notna(data['Price']) else 0.0
                line_total = qty * unit_price
                total_price += line_total
                
                unit_str = "μF" if data['Type'] == 'C' else "mH" if data['Type'] == 'L' else "Ω"
                comp_type_full = "Capacitor" if data['Type'] == 'C' else "Inductor" if data['Type'] == 'L' else "Resistor"
                
                writer.writerow([part_num, qty, data['Value'], unit_str, comp_type_full, data['Description'], round(unit_price, 2), round(line_total, 2), data['URL']])
                print(f"[{qty}x] {data['Value']}{unit_str} - {data['Description']} (Part #{part_num})")
                print(f"      Price: ${unit_price:.2f} each -> Total: ${line_total:.2f}")

            writer.writerow([])
            writer.writerow(['', '', '', '', '', 'TOTAL (1 Speaker):', f"${total_price:.2f}", '', ''])
            writer.writerow(['', '', '', '', '', 'TOTAL (Pair):', f"${total_price * 2:.2f}", '', ''])
            
        print("="*60)
        print(f"💰 TOTAL ESTIMATED COST: ${total_price:.2f} / Speaker")
        print(f"📄 Nomenclature (CSV) sauvegardée dans : {filename}")

        # ==========================================
        # 2. GÉNÉRATION DU FICHIER LATEX (OVERLEAF)
        # ==========================================
        latex_filename = filename.replace('.csv', '.tex') if filename.endswith('.csv') else filename + ".tex"
        
        with open(latex_filename, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[H]\n")
            f.write("    \\centering\n")
            f.write("    \\renewcommand{\\arraystretch}{1.5}\n")
            f.write("    \\begin{tabular}{@{}lllrl@{}}\n")
            f.write("        \\toprule\n")
            f.write("        \\textbf{ID} & \\textbf{Component} & \\textbf{Value} & \\textbf{Price (\\$/€)} & \\textbf{Buy Link} \\\\\n")
            f.write("        \\midrule\n")
            
            counts = {'C': 0, 'L': 0, 'R': 0}
            total_price_latex = 0.0  # Initialisation du cumulatif
            
            for comp in comps:
                ctype = CATALOG.get_comp_type(comp)
                counts[ctype] += 1
                comp_id = f"{ctype}{counts[ctype]}"
                
                val_cat = CATALOG.snap_to_catalog(comp.value, ctype)
                part_info = CATALOG.get_part_info(val_cat, ctype)
                
                unit_str = "$\\mu$F" if ctype == 'C' else "mH" if ctype == 'L' else "$\\Omega$"
                comp_type_full = "Capacitor" if ctype == 'C' else "Inductor" if ctype == 'L' else "Resistor"
                
                # Extraction du prix pour le cumul
                price = part_info['Price'] if pd.notna(part_info['Price']) else 0.0
                total_price_latex += price
                
                url = str(part_info['URL']).replace('%', '\\%').replace('#', '\\#')
                
                f.write(f"        {comp_id} & {comp_type_full} & {part_info['Value']} {unit_str} & {price:.2f} & \\href{{{url}}}{{Link}} \\\\\n")
            
            # --- AJOUT DE LA LIGNE DE TOTAL ---
            f.write("        \\midrule\n")
            f.write(f"        \\multicolumn{{3}}{{r}}{{\\textbf{{Estimated Total}}}} & \\textbf{{{total_price_latex:.2f}}} & \\\\\n")
            f.write("        \\bottomrule\n")
            f.write("    \\end{tabular}\n")
            f.write("\\end{table}\n")
            
        print(f"📝 Tableau LaTeX sauvegardé dans : {latex_filename}\n")
        
    def plot_geometry(self, filename="geometry.png"):
        """Génère un plan 2D de la façade (baffle) avec les distances entre axes des haut-parleurs."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Création d'une figure verticale typique d'une enceinte
        fig, ax = plt.subplots(figsize=(6, 8))

        points = []
        for way in self.ways:
            # Conversion des mètres vers centimètres pour un affichage plus lisible
            x = getattr(way, 'x_offset', 0.0) * 100  
            y = getattr(way, 'y_offset', 0.0) * 100  
            z = getattr(way, 'z_offset', 0.0) * 100  
            
            # Gestion de l'affichage (ex: "2x Woofer" si configuré)
            count = getattr(way, 'count', 1)
            count_str = f"{count}x " if count > 1 else ""
            label_text = f"{way.label}\n{count_str}{way.driver.model_name}"
            
            points.append({
                'label': label_text, 
                'x': x, 'y': y, 'z': z
            })

        # Tri des haut-parleurs du plus haut au plus bas (axe Y décroissant)
        points.sort(key=lambda p: p['y'], reverse=True)

        # 1. Tracé des Haut-Parleurs (Points et Labels)
        for p in points:
            # Cercle représentant l'encombrement du haut-parleur (transparent pour voir au travers)
            ax.scatter(p['x'], p['y'], s=1200, facecolors='none', edgecolors='black', linewidth=2, zorder=3)
            # Marqueur central représentant l'axe exact
            ax.scatter(p['x'], p['y'], s=30, color='black', zorder=4)
            
            # Étiquette descriptive (placée à droite du point)
            ax.text(p['x'] + 2.5, p['y'], 
                    f"{p['label']}\nZ offset : {p['z']:.1f} cm",
                    va='center', ha='left', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#ced4da", alpha=0.9), zorder=5)

        # 2. Tracé des flèches et des distances (Cotes)
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            # Calcul de la distance 2D (sur la façade) et 3D (avec la profondeur)
            dist_2d = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
            dist_3d = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

            # Dessin de la flèche de cotation (shrink à 0 pour relier les centres exacts)
            ax.annotate('', xy=(p2['x'], p2['y']), xytext=(p1['x'], p1['y']),
                        arrowprops=dict(arrowstyle='<->', color='#e63946', lw=2, shrinkA=0, shrinkB=0), zorder=2)

            # Positionnement du texte de distance (au milieu, placé à gauche)
            mid_x = (p1['x'] + p2['x']) / 2
            mid_y = (p1['y'] + p2['y']) / 2

            dist_text = f"{dist_2d:.1f} cm"
            # Affichage de la distance 3D seulement si elle diffère significativement de la 2D
            if abs(dist_3d - dist_2d) > 0.1:
                dist_text += f"\n(3D: {dist_3d:.1f} cm)"

            ax.text(mid_x - 2.5, mid_y, dist_text, color='#d62828',
                    va='center', ha='right', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fdf0d5", ec="none", alpha=0.8), zorder=5)

        ax.set_title("Baffle Geometry", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("X axis - cm")
        ax.set_ylabel("Y axis - cm")

        # CRUCIAL : Force les axes à avoir la même échelle réelle (aspect proportionnel)
        ax.set_aspect('equal', 'datalim')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axvline(0, color='gray', linestyle='-.', alpha=0.3, zorder=1) # Ligne de centre

        # Marges adaptatives pour ne pas couper le texte sur les côtés
        all_x = [p['x'] for p in points]
        all_y = [p['y'] for p in points]
        
        width_x = max(all_x) - min(all_x)
        if width_x < 10:
            x_min, x_max = min(all_x) - 15, max(all_x) + 20
        else:
            x_min, x_max = min(all_x) - 15, max(all_x) + 25
            
        y_pad = max(10, (max(all_y) - min(all_y)) * 0.15 if all_y else 10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    start_time = time.time()
    config = [
        # NOUVEAU : On peut maintenant passer count=2 à l'initialisation
        WayConfig("Woofer", r"crossovers\ER18RNX+27TDFC\SEAS_H1456-08_ER18RNX_SPL.frd", r"crossovers\ER18RNX+27TDFC\SEAS_H1456-08_ER18RNX_ZR.zma",
                  z_offset=0.00, y_offset=-100, x_offset=0.00, count=2),
        WayConfig("Tweeter", r"crossovers\ER18RNX+27TDFC\Tweeter_SPL.frd", r"crossovers\ER18RNX+27TDFC\Tweeter_ZR.zma")
    ]

    opt = CrossoverOptimizer(config)
    best = opt.run(generations=100, pop_size=120)
    best['tree'].display()
    end_time = time.time()
    print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")