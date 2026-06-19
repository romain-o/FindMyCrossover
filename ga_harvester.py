"""
ga_harvester.py — Source des DONNÉES d'imitation (cold start).

Deux professeurs possibles :

1) TemplateTeacher (utilisable TOUT DE SUITE, sans votre GA)
   Pour une tâche donnée, il évalue les familles canoniques (tree_actions.TEMPLATES)
   par une optimisation de valeurs RAPIDE et garde la meilleure. On distille ainsi une
   "recherche par force brute sur templates" dans la politique. C'est un excellent
   warm-start avant le RL.

2) GATeacher (à brancher quand vous voulez la qualité maximale)
   Enveloppe VOTRE CrossoverOptimizer (src/optimizer.py). Il faut convertir l'arbre
   nodes.py renvoyé par le GA en (low_tokens, high_tokens) via bridge.tree_to_tokens.
   Non exécuté ici (nécessite vos vrais fichiers + le package src).
"""
from __future__ import annotations
import torch
import tree_actions as TA
from inner_optim import evaluate_topology


class TemplateTeacher:
    def __init__(self, fast_restarts=48, fast_steps=80):
        self.fr, self.fs = fast_restarts, fast_steps

    def best_for(self, task):
        """Renvoie (low_ids, high_ids, loss) du meilleur template pour cette tâche."""
        best = None
        for tmpl in TA.TEMPLATES:
            low, high = tmpl
            root, slots = TA.build_topology(low, high)
            loss, *_ = evaluate_topology(task, root, slots,
                                         n_restarts=self.fr, steps=self.fs, verbose=False)
            if best is None or loss < best[2]:
                low_ids = [TA.VOCAB_ID[t] for t in low] + [TA.STOP]
                high_ids = [TA.VOCAB_ID[t] for t in high] + [TA.STOP]
                best = (low_ids, high_ids, loss)
        return best


def harvest_dataset(sampler, n_pairs=400, out_path="imitation_dataset.pt",
                    teacher=None, verbose=True):
    """
    Étiquette N paires RÉELLES compatibles (woofer, tweeter) avec le meilleur template,
    UNE fois. Sauvegarde [(low_name, high_name, low_ids, high_ids, loss)].
    On couvre train ET val (le split honnête se fait ensuite par nom).
    """
    import random as _r
    from data_pipeline import crossover_compatible, build_task
    import config as C
    teacher = teacher or TemplateTeacher()
    rng = _r.Random(0)
    lows = sampler.train.get(C.ROLE_LOW, []) + sampler.val.get(C.ROLE_LOW, [])
    highs = sampler.train.get(C.ROLE_HIGH, []) + sampler.val.get(C.ROLE_HIGH, [])

    seen, records = set(), []
    tries = 0
    while len(records) < n_pairs and tries < n_pairs * 30:
        tries += 1
        low, high = rng.choice(lows), rng.choice(highs)
        key = (low.name, high.name)
        if key in seen:
            continue
        if not crossover_compatible(low, high, sampler.freqs):
            continue
        seen.add(key)
        task = build_task(low, high, sampler.freqs)
        low_ids, high_ids, loss = teacher.best_for(task)
        records.append({"low": low.name, "high": high.name,
                        "low_ids": low_ids, "high_ids": high_ids, "loss": loss})
        if verbose and len(records) % 25 == 0:
            print(f"  récolté {len(records)}/{n_pairs}  ({low.name}+{high.name} loss={loss:.2f})")
    torch.save(records, out_path)
    if verbose:
        print(f"[+] dataset sauvegardé : {out_path}  ({len(records)} paires)")
    return records


class GATeacher:
    """Stub : à compléter avec votre optimizer (voir bridge.tree_to_tokens)."""
    def __init__(self, app_config=None):
        self.app_config = app_config or {}

    def best_for(self, task):
        raise NotImplementedError(
            "Branchez ici src.optimizer.CrossoverOptimizer : "
            "lancez run(), récupérez le champion (arbre nodes.py), "
            "puis bridge.tree_to_tokens(champion) -> (low_ids, high_ids).")