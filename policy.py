"""
policy.py — Politique autorégressive qui construit la topologie token par token.

Génère la branche BASSE (jusqu'à STOP) puis la branche HAUTE (jusqu'à STOP),
conditionnée par l'embedding système (encodeur de drivers). Les actions invalides
sont masquées (cf. tree_actions.valid_mask). Travaille par système (batch=1) : le
parallélisme massif est dans l'optimisation des VALEURS, pas dans la topologie.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical
import tree_actions as TA

BOS = TA.N_ACTIONS   # token de départ


class TopologyPolicy(nn.Module):
    def __init__(self, sys_dim=128, d=48, hidden=128):
        super().__init__()
        self.tok_emb = nn.Embedding(TA.N_ACTIONS + 1, d)
        self.init_h = nn.Linear(sys_dim, hidden)
        self.cell = nn.GRUCell(d + sys_dim + 1, hidden)
        self.out = nn.Linear(hidden, TA.N_ACTIONS)
        self.value = nn.Linear(hidden, 1)   # tête de valeur (baseline RL)
        self.hidden = hidden

    def _logits(self, prev_tok, sys_emb, branch, h):
        x = torch.cat([self.tok_emb(prev_tok), sys_emb,
                       torch.full((sys_emb.shape[0], 1), float(branch), device=sys_emb.device)], dim=-1)
        h = self.cell(x, h)
        return self.out(h), h

    @staticmethod
    def _mask_logits(logits, tokens_so_far):
        m = TA.valid_mask([TA.VOCAB[i] for i in tokens_so_far])
        mask = torch.tensor(m, device=logits.device)
        return logits.masked_fill(~mask, float("-inf"))

    def rollout(self, sys_emb, greedy=False, max_len=TA.MAX_TOKENS_PER_BRANCH):
        """Retourne (low_tokens, high_tokens, logprob_total, entropy_total, value)."""
        dev = sys_emb.device
        h = torch.tanh(self.init_h(sys_emb))
        total_lp = torch.zeros((), device=dev)
        total_ent = torch.zeros((), device=dev)
        v0 = self.value(h).squeeze()
        branches = []
        for branch in (0, 1):
            toks, prev = [], torch.tensor([BOS], device=dev)
            for _ in range(max_len + 1):
                logits, h = self._logits(prev, sys_emb, branch, h)
                logits = self._mask_logits(logits[0], toks).unsqueeze(0)
                dist = Categorical(logits=logits)
                a = logits.argmax(-1) if greedy else dist.sample()
                total_lp = total_lp + dist.log_prob(a).squeeze()
                total_ent = total_ent + dist.entropy().squeeze()
                ai = int(a.item())
                if ai == TA.STOP:
                    break
                toks.append(ai); prev = a
            branches.append(toks)
        return branches[0], branches[1], total_lp, total_ent, v0

    def log_prob(self, sys_emb, low_ids, high_ids):
        """Log-prob (teacher forcing) des séquences fournies (imitation). Termine par STOP."""
        dev = sys_emb.device
        h = torch.tanh(self.init_h(sys_emb))
        lp = torch.zeros((), device=dev)
        for branch, ids in ((0, low_ids), (1, high_ids)):
            toks, prev = [], torch.tensor([BOS], device=dev)
            for a in ids:
                logits, h = self._logits(prev, sys_emb, branch, h)
                logits = self._mask_logits(logits[0], toks).unsqueeze(0)
                dist = Categorical(logits=logits)
                at = torch.tensor([a], device=dev)
                lp = lp + dist.log_prob(at).squeeze()
                if a == TA.STOP:
                    break
                toks.append(a); prev = at
        return lp
