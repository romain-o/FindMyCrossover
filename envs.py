"""
envs.py — Environnement "bandit de topologie".

Un épisode = la politique construit une topologie complète (2 branches), on optimise
ses valeurs (inner_optim), et la récompense = -loss_finale. Ce cadrage épisodique
(une topologie = une décision) colle parfaitement à REINFORCE et reste simple.
"""
from __future__ import annotations
from dataclasses import dataclass
import tree_actions as TA
from inner_optim import evaluate_topology


@dataclass
class Episode:
    low_ids: list
    high_ids: list
    logprob: object        # tenseur scalaire (graph autograd de la politique)
    entropy: object
    value: object
    reward: float
    loss: float
    n_comp: int
    best_vals: object


def run_episode(policy, encoder, task, greedy=False, n_restarts=128, steps=150):
    """Échantillonne une topologie via la politique et l'évalue. Renvoie un Episode."""
    sys_emb = encoder.encode_task(task)
    low_t, high_t, lp, ent, v = policy.rollout(sys_emb, greedy=greedy)

    # ids -> tokens texte pour la grammaire
    low_tokens = [TA.VOCAB[i] for i in low_t]
    high_tokens = [TA.VOCAB[i] for i in high_t]
    root, slots = TA.build_topology(low_tokens, high_tokens)

    loss, best_vals, _, _ = evaluate_topology(task, root, slots,
                                              n_restarts=n_restarts, steps=steps, verbose=False)
    reward = -loss
    return Episode(low_ids=low_t + [TA.STOP], high_ids=high_t + [TA.STOP],
                   logprob=lp, entropy=ent, value=v, reward=reward, loss=loss,
                   n_comp=len(slots), best_vals=best_vals)
