"""
design.py — INFERENCE IA PURE pour concevoir un crossover concret.

Ce script n'utilise plus la force brute. Il interroge directement le modèle 
d'Apprentissage par Renforcement (rl_best.pt) pour générer la topologie 
sur mesure, puis optimise les valeurs sur GPU.

Utilisation :
    python -m design                               # 1re paire compatible trouvée
    python -m design --low RS150-8 --high RST28F-4 # Pour une paire spécifique
"""
from __future__ import annotations
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import tree_actions as TA
from data_pipeline import (discover_drivers, make_grid, resample_driver, build_task,
                           crossover_compatible, is_clean)
from inner_optim import evaluate_topology, resistor_watts


def _short_name(low_tok, high_tok):
    """Étiquette lisible de la topologie générée."""
    return f"basse={low_tok} | haute={high_tok}"


def _select_pair(raws, grid, low_name, high_name):
    """Sélectionne et prépare la paire de haut-parleurs."""
    L = {r.name: r for r in raws[C.ROLE_LOW]}
    H = {r.name: r for r in raws[C.ROLE_HIGH]}
    if low_name and high_name:
        return resample_driver(L[low_name], grid), resample_driver(H[high_name], grid)
    
    lows = [resample_driver(r, grid) for r in raws[C.ROLE_LOW]]
    highs = [resample_driver(r, grid) for r in raws[C.ROLE_HIGH]]
    lows = [d for d in lows if is_clean(d)] or lows
    highs = [d for d in highs if is_clean(d)] or highs
    
    for lo in lows:
        for hi in highs:
            if crossover_compatible(lo, hi, grid):
                return lo, hi
    return lows[0], highs[0]


def _eval(task, low_tok, high_tok, restarts, steps):
    """Évalue la topologie IA en cherchant les meilleures valeurs de composants."""
    root, slots = TA.build_topology(low_tok, high_tok)
    loss, vals, circ, _ = evaluate_topology(task, root, slots,
                                            n_restarts=restarts, steps=steps, verbose=False)
    return {"loss": float(loss), "vals": vals, "circ": circ, "slots": slots,
            "root": root, "low": low_tok, "high": high_tok}


def _get_policy_topology(task):
    """Topologie générée exclusivement par le meilleur modèle RL."""
    from driver_encoder import SystemEncoder
    from policy import TopologyPolicy
    from utils import load_ckpt

    # Force l'utilisation exclusive de rl_best0.pt
    ckpt_path = "checkpoints/rl_best0.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "rl_best0.pt" # Fallback au cas où il a été copié à la racine
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("[-] Erreur : 'rl_best0.pt' est introuvable. Avez-vous terminé l'entraînement ?")

    enc = SystemEncoder().to(C.DEVICE)
    pol = TopologyPolicy(sys_dim=enc.sys_dim).to(C.DEVICE)
    
    load_ckpt(ckpt_path, enc, pol, map_location=C.DEVICE)
    enc.eval()
    pol.eval()

    sys_emb = enc.encode_task(task)
    # greedy=True force l'IA à prendre la décision mathématique la plus sûre (plus d'exploration aléatoire)
    low_t, high_t, *_ = pol.rollout(sys_emb, greedy=True)
    
    return [TA.VOCAB[i] for i in low_t], [TA.VOCAB[i] for i in high_t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", default=None)
    ap.add_argument("--high", default=None)
    ap.add_argument("--restarts", type=int, default=1024)  # Défaut monté à 1024 pour un lissage parfait
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    raws = discover_drivers()
    grid = make_grid(C.DEVICE)
    low, high = _select_pair(raws, grid, args.low, args.high)
    task = build_task(low, high, grid)
    
    print(f"[design] {task.meta['low']} ({task.meta['sens_low']:.1f}dB) + "
          f"{task.meta['high']} ({task.meta['sens_high']:.1f}dB)  cible={task.target_spl:.1f}dB")

    # ==========================================
    # 1. L'IA PENSE LA TOPOLOGIE
    # ==========================================
    print("[design] 🧠 Interrogation du réseau de neurones (rl_best0.pt)...")
    low_tok, high_tok = _get_policy_topology(task)
    print(f"[design] Architecture choisie : {_short_name(low_tok, high_tok)}")

    # ==========================================
    # 2. LE GPU OPTIMISE LES VALEURS
    # ==========================================
    print(f"[design] ⚡ Optimisation des valeurs sur GPU ({args.restarts} restarts x {args.steps} steps)...")
    best = _eval(task, low_tok, high_tok, args.restarts, args.steps)

    root, slots, vals, circ = best["root"], best["slots"], best["vals"], best["circ"]
    print(f"\n[design] >>> RÉSULTAT FINAL  (loss={best['loss']:.3f})")

    # Nomenclature + dissipation des résistances
    watts = {slot: p for (slot, R, p) in resistor_watts(root, vals, task)}
    print("[BOM]")
    for j, ((kind, _), v) in enumerate(zip(slots, vals.cpu().tolist())):
        if kind == "R":
            p = watts.get(j, 0.0)
            flag = "  <-- TROP CHAUD" if p > C.RES_POWER_MAX else ""
            print(f"   R = {v:8.2f} Ω   (P max ~{p:5.1f} W sous 100W){flag}")
        elif kind == "C": 
            print(f"   C = {v*1e6:8.2f} µF")
        else:             
            print(f"   L = {v*1e3:8.3f} mH")

    # Tracé
    with torch.no_grad():
        resp = circ.forward(vals.unsqueeze(0))
    f = grid.cpu().numpy()
    Plow = resp["low"]["P"][0].cpu().numpy()
    Phigh = resp["high"]["P"][0].cpu().numpy()
    Psum = Plow + Phigh
    spl = lambda P: 20*np.log10(np.abs(P)+1e-12)
    
    plt.figure(figsize=(9, 5))
    plt.semilogx(f, spl(Psum), 'k', lw=2.2, label="Somme")
    plt.semilogx(f, spl(Plow), 'b', lw=1, label=task.meta['low'])
    plt.semilogx(f, spl(Phigh), 'r', lw=1, label=task.meta['high'])
    plt.axhline(task.target_spl, color='g', ls='--', lw=0.8, label="Cible")
    ymax = max(spl(Psum).max(), task.target_spl) + 6
    plt.xlim(20, 20000); plt.ylim(ymax-45, ymax)
    plt.xlabel("Hz"); plt.ylabel("dB SPL"); plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.title(f"Crossover généré par IA — loss={best['loss']:.2f}")
    plt.tight_layout()
    plt.savefig("design_response.png", dpi=130)
    print("[design] tracé : design_response.png")

    # Schéma via votre code
    try:
        from bridge import to_legacy_tree
        try:
            from nodes import DriverNode
            from schematic import SchematicRenderer
        except ImportError:
            from src.nodes import DriverNode
            from src.schematic import SchematicRenderer
            
        raw_low = next(r for r in raws[C.ROLE_LOW] if r.name == task.meta['low'])
        raw_high = next(r for r in raws[C.ROLE_HIGH] if r.name == task.meta['high'])
        drivers_real = {
            "low": DriverNode(task.meta['low'], raw_low.frd_path, raw_low.zma_path),
            "high": DriverNode(task.meta['high'], raw_high.frd_path, raw_high.zma_path)
        }
        legacy_tree = to_legacy_tree(root, vals, drivers_real)
        SchematicRenderer(legacy_tree).save("crossover_schematic.png")
        print("[design] tracé : crossover_schematic.png")
    except Exception as e:
        print(f"[design] Impossible de dessiner le schéma : {e}")


if __name__ == "__main__":
    main()