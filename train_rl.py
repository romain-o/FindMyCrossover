"""
train_rl.py — Phase C : fine-tuning par RL (REINFORCE + baseline), AVEC checkpoints.

La politique propose une topologie -> optimisation des valeurs -> reward = -loss.
On démarre du meilleur checkpoint d'imitation (warm start), on suit la récompense en
VALIDATION (drivers jamais vus), on checkpointe régulièrement + le meilleur.

Lancement :
    python -m crossover_ai.train_rl
"""
from __future__ import annotations
import os
import numpy as np
import torch

import config as C
from data_pipeline import TaskSampler
from driver_encoder import SystemEncoder
from policy import TopologyPolicy
from envs import run_episode
from utils import save_ckpt, load_ckpt


def _val_reward(pol, enc, sampler, n=8, n_restarts=96, steps=150):
    """Récompense moyenne (greedy) sur des paires de validation."""
    pol.eval(); enc.eval()
    rs = []
    for _ in range(n):
        ep = run_episode(pol, enc, sampler.sample_val(), greedy=True,
                         n_restarts=n_restarts, steps=steps)
        rs.append(ep.reward)
    pol.train(); enc.train()
    return float(np.mean(rs))


def train(iters=2000, lr=1e-4, entropy_coef=0.02, value_coef=0.5,
          warm_start="checkpoints/imitation_best.pt",
          ckpt_dir="checkpoints", eval_every=100, save_every=200,
          n_restarts=128, steps=150, sampler=None, val_frac=0.2, seed=0):
    sampler = sampler or TaskSampler(val_frac=val_frac, seed=seed)
    enc = SystemEncoder().to(C.DEVICE)
    pol = TopologyPolicy(sys_dim=enc.sys_dim).to(C.DEVICE)
    if warm_start and os.path.exists(warm_start):
        load_ckpt(warm_start, enc, pol)
        print(f"[+] warm start depuis {warm_start}")
    elif os.path.exists("imitation_ckpt.pt"):
        ck = torch.load("imitation_ckpt.pt", weights_only=True)
        enc.load_state_dict(ck["enc"]); pol.load_state_dict(ck["pol"])
        print("[+] warm start depuis imitation_ckpt.pt")
    else:
        print("[!] pas de warm start — RL à froid (déconseillé)")

    opt = torch.optim.Adam(list(enc.parameters()) + list(pol.parameters()), lr=lr)
    os.makedirs(ckpt_dir, exist_ok=True)
    run_r, best_val = None, -float("inf")

    for it in range(iters):
        ep = run_episode(pol, enc, sampler.sample_train(),
                         n_restarts=n_restarts, steps=steps)
        r = torch.tensor(ep.reward, device=C.DEVICE)
        adv = (r - ep.value.detach())
        loss = -adv * ep.logprob + value_coef * (r - ep.value).pow(2) - entropy_coef * ep.entropy
        opt.zero_grad(); loss.backward(); opt.step()

        run_r = ep.reward if run_r is None else 0.97 * run_r + 0.03 * ep.reward
        if it % 25 == 0:
            print(f"  it={it:5d}  reward~{run_r:8.3f}  loss_topo={ep.loss:7.3f}  n_comp={ep.n_comp}")

        if it > 0 and it % eval_every == 0:
            vr = _val_reward(pol, enc, sampler)
            print(f"     [val] reward={vr:.3f}  (train~{run_r:.3f})")
            if vr > best_val:
                best_val = vr
                save_ckpt(os.path.join(ckpt_dir, "rl_best.pt"), enc, pol, it,
                          meta={"val_reward": vr})
                print(f"       -> meilleur RL sauvegardé (val={vr:.3f})")
        if it > 0 and it % save_every == 0:
            save_ckpt(os.path.join(ckpt_dir, "rl_last.pt"), enc, pol, it, opt=opt)

    save_ckpt(os.path.join(ckpt_dir, "rl_last.pt"), enc, pol, iters, opt=opt)
    best = os.path.join(ckpt_dir, "rl_best.pt")
    if os.path.exists(best):
        import shutil; shutil.copy(best, "rl_ckpt.pt")
    print(f"[+] terminé. Meilleur val_reward={best_val:.3f}. Checkpoints dans {ckpt_dir}/")
    return enc, pol


if __name__ == "__main__":
    train(
    iters=10000,                 # démarrez à 4000, prolongez si val_reward monte encore
    warm_start="checkpoints/imitation_best.pt",   # IMPORTANT au 1er run RL
    lr=1e-4,
    entropy_coef=0.008,          
    value_coef=0.5,
    n_restarts=128,             # qualité de l'inner-optim (voir ci-dessous)
    steps=200,
    eval_every=200,
    save_every=400,
)