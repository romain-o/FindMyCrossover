"""
smoke_test.py — Vérifie que TOUTE la chaîne tourne, sur données synthétiques.

    python -m crossover_ai.smoke_test
"""
from __future__ import annotations
import tempfile, os, torch
import config as C
from data_pipeline import write_synthetic_dataset, TaskSampler
from driver_encoder import SystemEncoder
from policy import TopologyPolicy
from ga_harvester import TemplateTeacher
from envs import run_episode
import train_imitation, train_rl
import tree_actions as TA


def main():
    print(f"[device] {C.DEVICE}\n")

    # 1) Dataset synthétique réaliste (vrais fichiers .frd/.zma)
    tmp = tempfile.mkdtemp()
    write_synthetic_dataset(tmp, n_per_class=4)
    print(f"[data] dataset synthétique écrit dans {tmp}")
    sampler = TaskSampler(root=tmp, val_frac=0.25, use_cache=False)
    print(f"[data] stats train/val : {sampler.stats()}")
    task = sampler.sample_train()
    print(f"[data] tâche : low={task.meta['low']} high={task.meta['high']} "
          f"target={task.target_spl:.1f}dB bande=[{task.meta['fmin']:.0f},{task.meta['fmax']:.0f}]Hz\n")

    # 2) Encodeur + politique : un rollout
    enc = SystemEncoder().to(C.DEVICE)
    pol = TopologyPolicy(sys_dim=enc.sys_dim).to(C.DEVICE)
    sys_emb = enc.encode_task(task)
    low_t, high_t, lp, ent, v = pol.rollout(sys_emb)
    print(f"[policy] topo échantillonnée : low={[TA.VOCAB[i] for i in low_t]} "
          f"high={[TA.VOCAB[i] for i in high_t]}  logp={lp.item():.2f}\n")

    # 3) Professeur templates : meilleure topo + son score
    teacher = TemplateTeacher(fast_restarts=32, fast_steps=60)
    low_ids, high_ids, loss = teacher.best_for(task)
    print(f"[teacher] meilleur template : low={[TA.VOCAB[i] for i in low_ids]} "
          f"high={[TA.VOCAB[i] for i in high_ids]}  loss={loss:.3f}\n")

    # 4) Un épisode RL complet (rollout + inner-optim + reward)
    ep = run_episode(pol, enc, task, n_restarts=64, steps=100)
    print(f"[env] épisode : reward={ep.reward:.3f} loss={ep.loss:.3f} n_comp={ep.n_comp}\n")

    # 5) Micro-entraînement imitation puis RL (quelques itérations)
    print("[imitation] 30 itérations :")
    enc, pol = train_imitation.train(steps=30, log_every=10, sampler=sampler, teacher=teacher)
    print("\n[RL] 15 itérations (warm start imitation) :")
    train_rl.train(iters=15, warm_start="imitation_ckpt.pt",
                   n_restarts=48, steps=60, sampler=sampler, log_every=5)

    print("\n[OK] chaîne complète fonctionnelle de bout en bout.")


if __name__ == "__main__":
    main()
