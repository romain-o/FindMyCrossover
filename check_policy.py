"""
check_policy.py — Évalue une politique ENTRAÎNÉE (greedy) vs l'oracle-templates.

    python -m crossover_ai.check_policy            # utilise rl_ckpt.pt si présent, sinon imitation_ckpt.pt

Pour chaque tâche de validation : compare la loss de la topologie choisie par la
politique à la meilleure loss obtenue par force brute sur les 6 templates.
Un bon résultat : politique <= oracle (la politique peut FAIRE MIEUX que les templates
après RL, puisqu'elle explore hors de ces 6 familles).
"""
from __future__ import annotations
import os, torch
import config as C
from data_pipeline import TaskSampler
from driver_encoder import SystemEncoder
from policy import TopologyPolicy
from ga_harvester import TemplateTeacher
from envs import run_episode
import tree_actions as TA


def main(n_tasks=10):
    ckpt = "rl_ckpt.pt" if os.path.exists("rl_ckpt.pt") else "imitation_ckpt.pt"
    if not os.path.exists(ckpt):
        print("[!] Aucun checkpoint — lancez train_imitation puis train_rl.")
        return
    sampler = TaskSampler()
    enc = SystemEncoder().to(C.DEVICE); pol = TopologyPolicy(sys_dim=enc.sys_dim).to(C.DEVICE)
    ck = torch.load(ckpt, weights_only=True)
    enc.load_state_dict(ck["enc"]); pol.load_state_dict(ck["pol"])
    enc.eval(); pol.eval()
    teacher = TemplateTeacher(fast_restarts=64, fast_steps=120)
    print(f"[eval] checkpoint = {ckpt}")

    wins = 0
    for _ in range(n_tasks):
        task = sampler.sample_val()
        with torch.no_grad():
            ep = run_episode(pol, enc, task, greedy=True, n_restarts=128, steps=200)
        _, _, oracle = teacher.best_for(task)
        better = ep.loss <= oracle + 1e-3
        wins += better
        print(f"  {task.meta['low']:>12}+{task.meta['high']:<12} "
              f"politique={ep.loss:7.3f}  oracle={oracle:7.3f}  "
              f"{'>=oracle ✓' if better else ''}  "
              f"topo=low{[TA.VOCAB[i] for i in ep.low_ids[:-1]]} high{[TA.VOCAB[i] for i in ep.high_ids[:-1]]}")
    print(f"\n[résumé] politique >= oracle sur {wins}/{n_tasks} tâches.")


if __name__ == "__main__":
    main()
