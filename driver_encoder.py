"""
driver_encoder.py — Encode les courbes FRD/ZMA d'un système 2 voies en un vecteur.

Entrée par driver : 4 canaux x F  =  [SPL_dB(norm), SPL_phase, log|Z|, Z_phase]
On encode chaque driver par un CNN 1D, puis on concatène (low, high) + scalaires.
Le vecteur "système" conditionne ensuite la politique de topologie.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import config as C


def driver_channels(Z, H):
    """(Z[F] complexe, H[F] complexe) -> tenseur [4, F] normalisé (float32)."""
    spl = 20 * torch.log10(H.abs() + 1e-10)
    spl = (spl - 85.0) / 15.0                     # normalisation grossière autour de 85 dB
    sph = torch.angle(H) / 3.14159
    zlm = torch.log10(Z.abs() + 1e-6) - 0.8       # ~log10(6 ohm)
    zph = torch.angle(Z) / 3.14159
    return torch.stack([spl, sph, zlm, zph], dim=0).to(torch.float32)


class _Conv1dStack(nn.Module):
    def __init__(self, cin=4, hidden=32, emb=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, hidden, 7, padding=3), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, 5, padding=2, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden * 2, emb)

    def forward(self, x):                          # x : [B, 4, F]
        h = self.net(x).squeeze(-1)                # [B, hidden*2]
        return self.proj(h)                        # [B, emb]


class SystemEncoder(nn.Module):
    """Encode (driver_low, driver_high) -> embedding système [E]."""
    def __init__(self, emb=64, sys_dim=128, dropout=0.1):
        super().__init__()
        self.drv = _Conv1dStack(emb=emb)
        self.head = nn.Sequential(
            nn.Linear(emb * 2 + 2, sys_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(sys_dim, sys_dim),
        )
        self.sys_dim = sys_dim

    def encode_task(self, task):
        """task -> embedding [1, sys_dim] (1 système). Drivers sur le device du modèle."""
        dev = next(self.parameters()).device
        Zl, Hl = task.drivers["low"]; Zh, Hh = task.drivers["high"]
        cl = driver_channels(Zl.to(dev), Hl.to(dev)).unsqueeze(0)
        ch = driver_channels(Zh.to(dev), Hh.to(dev)).unsqueeze(0)
        el = self.drv(cl); eh = self.drv(ch)
        # scalaires : sensibilités relatives (low, high) recentrées
        sl = 20 * torch.log10(Hl.abs() + 1e-10)
        sh = 20 * torch.log10(Hh.abs() + 1e-10)
        scal = torch.tensor([[float(sl.mean()) / 100.0, float(sh.mean()) / 100.0]], device=dev)
        return self.head(torch.cat([el, eh, scal], dim=-1))   # [1, sys_dim]