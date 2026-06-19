"""
torch_reward.py — Reward DIFFÉRENTIABLE pour le 2 voies (portage d'un sous-ensemble
fidèle de votre fitness()). Sert à la fois :
  - de fonction-objectif pour l'optimisation interne des valeurs (gradients exacts),
  - de récompense (= -loss) pour le RL de topologie.

Termes implémentés (mêmes intentions que optimizer.py) :
  MSE_SPL                : platitude de la somme vs cible, asymétrique (x2 si dépassement),
                           sur-pondérée autour de la fréquence de coupure détectée.
  Overshoot              : aucune voie ne doit dépasser cible+1.5 dB (anti-annulation de phase).
  Impedance              : |Zin| min ne doit pas descendre sous 3 ohm.
  high_low_leak          : la voie HAUTE ne doit pas conduire en dessous de fc.
  low_high_leak          : la voie BASSE ne doit pas conduire au-dessus de fc.
  components / resistors  : parcimonie (constants vs valeurs -> n'affectent pas l'inner-loop,
                           mais comptent dans la récompense RL).

Différentiabilité : la détection de fc (argmin) ne sert qu'à fabriquer des masques de
pondération -> on la fait en .no_grad() puis on l'utilise comme poids constant (detach).
"""
from __future__ import annotations
import torch
import config as C


def _detect_fc(spl_low, spl_high, freqs):
    """Fréquence de croisement (1ère où la voie haute dépasse la basse). Non différentiable -> detach."""
    with torch.no_grad():
        m = (freqs > 800) & (freqs < 6000)
        idx = torch.where(m)[0]
        if len(idx) == 0:
            return 2000.0
        sub = (spl_high - spl_low)[..., idx].mean(dim=0) if spl_high.dim() > 1 else (spl_high - spl_low)[idx]
        cross = torch.where(sub > 0)[0]
        if len(cross) > 0:
            return float(freqs[idx[cross[0]]])
        return float(freqs[idx[torch.argmin(sub.abs())]])


def make_reward(task, circuit, slots, weights=None):
    """
    Fabrique une fonction reward_fn(responses, circuit) -> loss[B] liée à `task`.
    `slots` = [(kind, val_init)] pour compter les composants/résistances.
    """
    W = weights or C.WEIGHTS
    freqs = task.freqs.to(circuit.device)
    band = task.band_mask.to(circuit.device)
    target = task.target_spl

    # pénalités de structure (constantes vs valeurs) — simple INDICATEUR, pas un mur.
    # Linéaire et douce : départage deux topologies de qualité acoustique équivalente,
    # mais n'empêche JAMAIS une topologie plus riche qui sonne mieux de l'emporter.
    n_comp = len(slots)
    n_res = sum(1 for (k, _) in slots if k == "R")
    extra = max(0, n_comp - int(W["n_comps_free"]))
    struct_pen = extra * W["components"] + n_res * W["resistors"]

    def reward_fn(responses, circ):
        P_low = responses["low"]["P"]; P_high = responses["high"]["P"]   # [B,F]
        V_low = responses["low"]["V"]; V_high = responses["high"]["V"]
        B = P_low.shape[0]
        p_sum = P_low + P_high
        spl = 20 * torch.log10(p_sum.abs() + 1e-12)
        spl_low = 20 * torch.log10(P_low.abs() + 1e-12)
        spl_high = 20 * torch.log10(P_high.abs() + 1e-12)

        # --- poids dynamique : base sur bande + bosse autour de fc ---
        fc = _detect_fc(spl_low, spl_high, freqs)
        w = torch.zeros_like(freqs); w[band] = 1.0
        w[(freqs > fc / 2) & (freqs < fc * 2)] = W["crossover"]
        w = w.detach().unsqueeze(0)                                       # [1,F]

        # --- MSE_SPL asymétrique ---
        diff = spl - target
        asym = torch.where(diff > 0, 2.0, 1.0)
        mse = (asym * diff.pow(2) * w).mean(dim=1) * W["mse_sum"]

        # --- overshoot (par voie) ---
        over = torch.relu(spl_low - (target + 1.5)).pow(2).mean(dim=1) \
             + torch.relu(spl_high - (target + 1.5)).pow(2).mean(dim=1)
        over = over * W["overshoot"]

        # --- impédance d'entrée ---
        Zin = circ.root._Z
        minZ = Zin.abs().min(dim=1).values
        imp = torch.relu(3.0 - minZ).pow(3) * (W["impedance"] / 100.0)

        # --- fuites de bande ---
        below = freqs < fc * 0.75
        above = freqs > fc * 1.4
        leak_h = torch.relu(V_high.abs()[:, below] - 0.05).pow(2).mean(dim=1) * W["high_low_leak"] \
                 if below.any() else torch.zeros(B, device=circ.device)
        leak_l = torch.relu(V_low.abs()[:, above] - 0.05).pow(2).mean(dim=1) * W["low_high_leak"] \
                 if above.any() else torch.zeros(B, device=circ.device)

        return mse + over + imp + leak_h + leak_l + struct_pen

    return reward_fn