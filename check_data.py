"""
check_data.py — Vérifie le chargement de VOS données réelles.

    CROSSOVER_DATA=/chemin/vers/data python -m crossover_ai.check_data

Contrôle : appariement FRD/ZMA, sensibilités plausibles, impédance plausible, NaN.
"""
from __future__ import annotations
import numpy as np
import config as C
from data_pipeline import discover_drivers, make_grid, resample_driver, is_clean


def main():
    print(f"[data] racine = {C.DATA_ROOT}  | classes = {C.CLASSES}")
    raws = discover_drivers()
    grid = make_grid("cpu")

    usable = {}
    for cls in C.CLASSES:
        lst = raws.get(cls, [])
        print(f"\n=== {cls} : {len(lst)} driver(s) appariés ===")
        if not lst:
            print(f"  [!] AUCUN driver. Vérifiez data/{cls}/ (paires .frd + .zma).")
            continue
        n_ok = 0
        for r in lst:
            try:
                d = resample_driver(r, grid)
                minZ = float(d.Z.abs().min())
                clean = is_clean(d)
                n_ok += clean
                flag = "" if clean else "  [écarté]"
                print(f"  {r.name:<24} sens={d.sens_db:6.1f}dB  min|Z|={minZ:6.1f}Ω{flag}")
            except Exception as e:
                print(f"  {r.name:<24} ERREUR : {e}")
        usable[cls] = n_ok
        print(f"  -> {n_ok}/{len(lst)} exploitables")

    if len(usable) >= 2:
        vals = list(usable.values())
        print(f"\n[OK] paires possibles ≈ {vals[0]} x {vals[1]} = {vals[0]*vals[1]:,} "
              f"(avant filtrage de compatibilité et augmentation).")


if __name__ == "__main__":
    main()