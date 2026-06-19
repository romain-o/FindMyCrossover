"""
bridge.py — Pont entre la représentation interne (torch_sim/tree_actions) et VOTRE code.

Permet de :
  - convertir une topologie interne + valeurs optimisées en arbre nodes.py réel,
  - puis réutiliser vos briques existantes : snap catalogue, schematic, vituix_exporter.

Les imports de `src.*` sont protégés : le package s'importe même si `src` est absent
(utile pour les tests). Les fonctions qui en dépendent lèvent une erreur claire si besoin.
"""
from __future__ import annotations
import tree_actions as TA
from torch_sim import Series, Parallel, Comp, Driver as TDriver

try:
    from src.nodes import (SeriesNode, ParallelNode, Resistor, Capacitor,
                           Inductor, DriverNode)
    _HAS_SRC = True
except Exception:
    _HAS_SRC = False


def to_legacy_tree(root, vals, drivers_real):
    """
    root  : TNode interne (torch_sim) ; vals : tenseur [n_comp] des valeurs SI optimisées.
    drivers_real : dict label('low'/'high') -> DriverNode réel (nodes.py) à insérer.
    Retourne un arbre nodes.py exploitable par schematic/vituix/catalog.
    """
    if not _HAS_SRC:
        raise RuntimeError("src.nodes introuvable : placez crossover_ai/ à côté de src/.")
    v = vals.detach().cpu().tolist()

    def conv(n):
        t = type(n)
        if t is Comp:
            x = float(v[n.slot])
            if n.kind == "R": return Resistor(x)
            if n.kind == "C": return Capacitor(x)
            return Inductor(x)
        if t is TDriver:
            return drivers_real[n.label]
        if t is Series:
            return SeriesNode(conv(n.left), conv(n.right))
        if t is Parallel:
            return ParallelNode(conv(n.left), conv(n.right))
        raise ValueError(f"noeud interne inconnu: {t}")
    return conv(root)


def tree_to_tokens(legacy_root):
    """
    Convertit un arbre nodes.py (champion GA) en (low_ids, high_ids) pour l'imitation.
    Hypothèse : racine ParallelNode(branche_low, branche_high), chaque branche étant une
    chaîne penchée à droite Series/Parallel terminée par un DriverNode.
    """
    if not _HAS_SRC:
        raise RuntimeError("src.nodes introuvable.")

    def branch_tokens(node):
        toks = []
        cur = node
        while True:
            cn = type(cur).__name__
            if cn == "DriverNode":
                break
            if cn == "SeriesNode":
                comp, rest = cur.left, cur.right
                pref = "SER_"
            elif cn == "ParallelNode":
                comp, rest = cur.left, cur.right
                pref = "SH_"
            else:
                break
            kind = {"Resistor": "R", "Capacitor": "C", "Inductor": "L"}.get(type(comp).__name__)
            if kind is None:   # composant non terminal -> on simplifie en s'arrêtant
                break
            toks.append(TA.VOCAB_ID[pref + kind])
            cur = rest
        return toks + [TA.STOP]

    assert type(legacy_root).__name__ == "ParallelNode", "racine doit être ParallelNode"
    low = branch_tokens(legacy_root.left)
    high = branch_tokens(legacy_root.right)
    return low, high
