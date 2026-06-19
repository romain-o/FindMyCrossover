"""
check_fit.py — Vérification VISUELLE de l'inner-loop sur un vrai couple Woofer/Mid.

    CROSSOVER_DATA=/chemin/vers/data python -m crossover_ai.check_fit

Choisit le meilleur des 6 templates pour la 1re paire (low, high), optimise les valeurs,
et trace : SPL sommé + par voie + cible, et |Zin|. Génère deux PNG.
C'est le test "à l'œil" : la somme doit être plate en bande, la coupure cohérente.
"""
from __future__ import annotations
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import tree_actions as TA
from data_pipeline import (discover_drivers, make_grid, resample_driver, build_task,
                            crossover_compatible, is_clean)
from inner_optim import evaluate_topology
from torch_reward import make_reward


def _pick_compatible(raws, grid):
    """Cherche un premier couple (woofer, tweeter) propre ET physiquement sensé."""
    lows = [resample_driver(r, grid) for r in raws[C.ROLE_LOW]]
    highs = [resample_driver(r, grid) for r in raws[C.ROLE_HIGH]]
    lows = [d for d in lows if is_clean(d)] or lows
    highs = [d for d in highs if is_clean(d)] or highs
    for lo in lows:
        for hi in highs:
            if crossover_compatible(lo, hi, grid):
                return lo, hi
    return lows[0], highs[0]   # à défaut


def main():
    raws = discover_drivers()
    grid = make_grid(C.DEVICE)
    if not raws.get(C.ROLE_LOW) or not raws.get(C.ROLE_HIGH):
        print("[!] Pas de paire disponible — lancez d'abord check_data.")
        return
    low, high = _pick_compatible(raws, grid)
    task = build_task(low, high, grid)
    print(f"[fit] {task.meta['low']} ({task.meta['sens_low']:.1f}dB) + "
          f"{task.meta['high']} ({task.meta['sens_high']:.1f}dB)  "
          f"cible={task.target_spl:.1f}dB  bande=[{task.meta['fmin']:.0f},{task.meta['fmax']:.0f}]Hz")

    # Meilleur template
    best = None
    for tmpl in TA.TEMPLATES:
        root, slots = TA.build_topology(*tmpl)
        loss, vals, circ, _ = evaluate_topology(task, root, slots, n_restarts=128, steps=250, verbose=False)
        if best is None or loss < best[0]:
            best = (loss, vals, circ, slots, tmpl)
    loss, vals, circ, slots, tmpl = best
    print(f"[fit] meilleur template low={tmpl[0]} high={tmpl[1]}  loss={loss:.3f}")

    with torch.no_grad():
        resp = circ.forward(vals.unsqueeze(0))
    f = grid.cpu().numpy()
    Plow = resp["low"]["P"][0].cpu().numpy(); Phigh = resp["high"]["P"][0].cpu().numpy()
    Psum = Plow + Phigh
    spl = lambda P: 20*np.log10(np.abs(P)+1e-12)
    Zin = circ.root._Z[0].cpu().numpy()

    # --- Tracé SPL ---
    plt.figure(figsize=(9, 5))
    plt.semilogx(f, spl(Psum), 'k', lw=2.2, label="Somme")
    plt.semilogx(f, spl(Plow), 'b', lw=1, label="Voie basse")
    plt.semilogx(f, spl(Phigh), 'r', lw=1, label="Voie haute")
    plt.axhline(task.target_spl, color='g', ls='--', lw=0.8, label="Cible")
    plt.axvspan(task.meta['fmin'], task.meta['fmax'], color='gray', alpha=0.08)
    ymax = max(spl(Psum).max(), task.target_spl) + 6
    plt.xlim(20, 20000); plt.ylim(ymax - 45, ymax)
    plt.xlabel("Hz"); plt.ylabel("dB SPL"); plt.grid(True, which='both', alpha=0.3)
    plt.legend(); plt.title(f"Réponse — loss={loss:.2f}")
    plt.tight_layout(); plt.savefig("check_fit_spl.png", dpi=130)

    # --- Tracé impédance ---
    plt.figure(figsize=(9, 3.5))
    plt.semilogx(f, np.abs(Zin), 'm', lw=1.8)
    plt.axhline(3.0, color='r', ls='--', lw=0.8, label="seuil 3Ω")
    plt.xlim(20, 20000); plt.xlabel("Hz"); plt.ylabel("|Z| (Ω)")
    plt.grid(True, which='both', alpha=0.3); plt.legend()
    plt.title(f"Impédance — min={np.abs(Zin).min():.1f}Ω")
    plt.tight_layout(); plt.savefig("check_fit_impedance.png", dpi=130)

    print("[OK] tracés : check_fit_spl.png , check_fit_impedance.png")


if __name__ == "__main__":
    main()