"""
train_imitation.py — Phase B : behavior cloning AVEC suivi de validation.

Recette du VRAI entraînement :
  1) on récolte UNE fois le professeur (meilleur template) sur des paires réelles
     compatibles -> dataset étiqueté (ga_harvester) ;
  2) on entraîne la politique sur plusieurs époques, en mesurant l'écart train/val
     sur des paires contenant des drivers JAMAIS VUS (détection d'overfitting) ;
  3) checkpoints réguliers + meilleur modèle (par NLL de validation) + early stopping.

Lancement :
    python -m crossover_ai.train_imitation
Options utiles : voir les arguments de train().
"""
from __future__ import annotations
import os, random
import numpy as np
import torch

import config as C
from data_pipeline import TaskSampler
from driver_encoder import SystemEncoder
from policy import TopologyPolicy
from ga_harvester import TemplateTeacher, harvest_dataset
from utils import save_ckpt, load_ckpt


def _nll_on(records, enc, pol, sampler, augment=False):
    """NLL moyenne (teacher forcing) sur une liste de records."""
    tot = 0.0
    for r in records:
        task = sampler.task_from_names(r["low"], r["high"], augment_data=augment)
        sys_emb = enc.encode_task(task)
        tot += (-pol.log_prob(sys_emb, r["low_ids"], r["high_ids"])).item()
    return tot / max(1, len(records))


def train(epochs=40, lr=3e-4, weight_decay=1e-4,
          dataset="imitation_dataset.pt", n_pairs=400,
          ckpt_dir="checkpoints", patience=6, seed=0,
          sampler=None, teacher=None, val_frac=0.2):
    rng = random.Random(seed)
    sampler = sampler or TaskSampler(val_frac=val_frac, seed=seed)
    teacher = teacher or TemplateTeacher()
    print(f"[data] {sampler.stats()}  | écartés: "
          f"{ {k: len(v) for k, v in sampler.dropped.items()} }")

    # 1) Dataset étiqueté (récolte unique, mise en cache)
    if os.path.exists(dataset):
        records = torch.load(dataset, weights_only=False)
        print(f"[data] dataset chargé : {dataset} ({len(records)} paires)")
    else:
        print(f"[data] récolte du professeur ({n_pairs} paires)…")
        records = harvest_dataset(sampler, n_pairs=n_pairs, out_path=dataset, teacher=teacher)

    # 2) Split honnête train/val par nom de driver
    train_rec = [r for r in records if not sampler.is_val_pair(r["low"], r["high"])]
    val_rec = [r for r in records if sampler.is_val_pair(r["low"], r["high"])]
    print(f"[split] train={len(train_rec)} paires  |  val={len(val_rec)} paires (drivers jamais vus)")
    if not train_rec:
        print("[!] pas de paires d'entraînement — baissez val_frac ou ajoutez des drivers."); return

    enc = SystemEncoder().to(C.DEVICE)
    pol = TopologyPolicy(sys_dim=enc.sys_dim).to(C.DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pol.parameters()),
                           lr=lr, weight_decay=weight_decay)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_val, bad = float("inf"), 0
    for ep in range(epochs):
        enc.train(); pol.train()
        rng.shuffle(train_rec)
        tr_loss = 0.0
        for r in train_rec:
            # augmentation de l'entrée encodeur (décalage/warp/bruit) : régularise
            task = sampler.task_from_names(r["low"], r["high"], augment_data=True)
            sys_emb = enc.encode_task(task)
            loss = -pol.log_prob(sys_emb, r["low_ids"], r["high_ids"])
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_rec)

        enc.eval(); pol.eval()
        with torch.no_grad():
            val_loss = _nll_on(val_rec, enc, pol, sampler) if val_rec else float("nan")
        gap = (val_loss - tr_loss) if val_rec else float("nan")
        print(f"  époque {ep:3d}  train_nll={tr_loss:.3f}  val_nll={val_loss:.3f}  "
              f"écart={gap:+.3f}  {'(overfit?)' if gap > 1.0 else ''}")

        save_ckpt(os.path.join(ckpt_dir, "imitation_last.pt"), enc, pol, ep,
                  meta={"train_nll": tr_loss, "val_nll": val_loss}, opt=opt)
        crit = val_loss if val_rec else tr_loss
        if crit < best_val - 1e-3:
            best_val, bad = crit, 0
            save_ckpt(os.path.join(ckpt_dir, "imitation_best.pt"), enc, pol, ep,
                      meta={"val_nll": val_loss})
            print(f"       -> meilleur modèle sauvegardé (val={crit:.3f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"[stop] pas d'amélioration val depuis {patience} époques — early stopping.")
                break

    # compat : on copie aussi le meilleur au chemin attendu par train_rl
    best = os.path.join(ckpt_dir, "imitation_best.pt")
    if os.path.exists(best):
        import shutil; shutil.copy(best, "imitation_ckpt.pt")
    print(f"[+] terminé. Meilleur val_nll={best_val:.3f}. Checkpoints dans {ckpt_dir}/")
    return enc, pol


if __name__ == "__main__":
    train(
    n_pairs=800,        # paires réelles récoltées (one-time, ~25% du catalogue)
    val_frac=0.2,       # ~18 woofers + 7 tweeters jamais vus
    epochs=120,
    patience=12,
    lr=3e-4,
    weight_decay=1e-4,
)