"""utils.py — sauvegarde / chargement de checkpoints (avec étape, méta, optimizer)."""
from __future__ import annotations
import os, torch


def save_ckpt(path, enc, pol, step=0, meta=None, opt=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"enc": enc.state_dict(), "pol": pol.state_dict(),
                "step": int(step), "meta": meta or {},
                "opt": opt.state_dict() if opt is not None else None}, path)


def load_ckpt(path, enc, pol, opt=None, map_location=None):
    ck = torch.load(path, weights_only=False, map_location=map_location)
    enc.load_state_dict(ck["enc"]); pol.load_state_dict(ck["pol"])
    if opt is not None and ck.get("opt") is not None:
        try: opt.load_state_dict(ck["opt"])
        except Exception: pass
    return ck.get("step", 0), ck.get("meta", {})