import torch
import math
import config as C
from torch_sim import Driver as TDriver, Comp, Series, Parallel
from torch_reward import make_reward


def compute_impedance(node, vals_b, freqs, task):
    """Impédance complexe [Batch, Freqs] de tout le circuit (récursif, batché GPU)."""
    if isinstance(node, TDriver):
        Z_drv = task.drivers[node.label][0]
        return Z_drv.unsqueeze(0).expand(vals_b.size(0), -1)
    elif isinstance(node, Comp):
        v = vals_b[:, node.slot].unsqueeze(1)            # [Batch, 1]
        w = (2 * math.pi * freqs).unsqueeze(0)           # [1, Freqs]
        if node.kind == 'R':
            return v.expand(-1, freqs.size(0)).to(torch.complex64)
        elif node.kind == 'C':
            return 1.0 / (1j * w * v + 1e-15)
        elif node.kind == 'L':
            return 1j * w * v + (v * C.DCR_COEFF)
    elif isinstance(node, Series):
        return compute_impedance(node.left, vals_b, freqs, task) + \
               compute_impedance(node.right, vals_b, freqs, task)
    elif isinstance(node, Parallel):
        z1 = compute_impedance(node.left, vals_b, freqs, task)
        z2 = compute_impedance(node.right, vals_b, freqs, task)
        return (z1 * z2) / (z1 + z2 + 1e-15)


def compute_voltage(node, V_in, vals_b, freqs, task, voltages_dict):
    """Passe descendante : diviseur de tension -> V_in de chaque haut-parleur."""
    if isinstance(node, TDriver):
        voltages_dict[node.label] = V_in
        return
    elif isinstance(node, Comp):
        return
    elif isinstance(node, Series):
        z_left = compute_impedance(node.left, vals_b, freqs, task)
        z_right = compute_impedance(node.right, vals_b, freqs, task)
        z_tot = z_left + z_right + 1e-15
        compute_voltage(node.left,  V_in * (z_left / z_tot),  vals_b, freqs, task, voltages_dict)
        compute_voltage(node.right, V_in * (z_right / z_tot), vals_b, freqs, task, voltages_dict)
    elif isinstance(node, Parallel):
        compute_voltage(node.left,  V_in, vals_b, freqs, task, voltages_dict)
        compute_voltage(node.right, V_in, vals_b, freqs, task, voltages_dict)


class _CircView:
    """Petit objet exposant ce dont make_reward a besoin (.device, .root._Z)."""
    def __init__(self, device):
        self.device = device
        self.root = type("R", (), {})()
        self.root._Z = None


def _responses(root, vals_b, freqs, task):
    """Renvoie (responses {lbl:{'V','P'}}, Zin) pour le batch courant."""
    V_in = torch.ones((vals_b.size(0), len(freqs)), dtype=torch.complex64, device=freqs.device)
    vd = {}
    compute_voltage(root, V_in, vals_b, freqs, task, vd)
    resp = {lbl: {"V": V, "P": V * task.drivers[lbl][1].unsqueeze(0)} for lbl, V in vd.items()}
    Zin = compute_impedance(root, vals_b, freqs, task)
    return resp, Zin


# ============================================================ DISSIPATION (= optimizer.py)
def compute_power_stats(node, V_in, vals_b, freqs, task):
    """
    Propage la tension de test et renvoie, batché [B,...] :
      (max_res_power[B], tot_res_power[B,F], tot_drv_power[B,F])
    P_resistance = |V|^2 / R   ;   P_driver = |V|^2 * Re(1/Z).  L/C : 0 (réactif pur).
    Réplique fidèlement _get_power_dissipation_stats du GA.
    """
    B, F = V_in.shape
    dev = V_in.device
    zF = torch.zeros((B, F), device=dev)
    if isinstance(node, Comp):
        if node.kind == 'R':
            R = vals_b[:, node.slot].unsqueeze(1)            # [B,1]
            p = (V_in.abs() ** 2) / (R + 1e-15)              # [B,F]
            return p.amax(dim=1), p, zF
        return torch.zeros(B, device=dev), zF, zF            # L/C : pas de chaleur
    if isinstance(node, TDriver):
        Z = task.drivers[node.label][0].unsqueeze(0)         # [1,F]
        p = (V_in.abs() ** 2) * torch.real(1.0 / (Z + 1e-15))
        return torch.zeros(B, device=dev), zF, p
    if isinstance(node, Parallel):
        ml, rl, dl = compute_power_stats(node.left,  V_in, vals_b, freqs, task)
        mr, rr, dr = compute_power_stats(node.right, V_in, vals_b, freqs, task)
        return torch.maximum(ml, mr), rl + rr, dl + dr
    if isinstance(node, Series):
        zl = compute_impedance(node.left,  vals_b, freqs, task)
        zr = compute_impedance(node.right, vals_b, freqs, task)
        zt = zl + zr + 1e-15
        ml, rl, dl = compute_power_stats(node.left,  V_in * (zl / zt), vals_b, freqs, task)
        mr, rr, dr = compute_power_stats(node.right, V_in * (zr / zt), vals_b, freqs, task)
        return torch.maximum(ml, mr), rl + rr, dl + dr
    return torch.zeros(B, device=dev), zF, zF


def thermal_penalty(root, vals_b, task):
    """Pénalité [B] de dissipation : résistance trop chaude + gaspillage dans les graves."""
    freqs = task.freqs
    B = vals_b.size(0)
    V = torch.full((B, len(freqs)), C.POWER_TEST_V, dtype=torch.complex64, device=freqs.device)
    maxr, totr, totd = compute_power_stats(root, V, vals_b, freqs, task)
    w = C.WEIGHTS.get("thermal", 20.0)

    # Règle 1 : aucune résistance ne doit dépasser RES_POWER_MAX (W)
    gate = (maxr > C.RES_POWER_MAX).float()
    pen1 = gate * (maxr - (C.RES_POWER_MAX - 2.0)).pow(2)          # mesuré depuis ~15 W
    pen1 = pen1 * (w * 0.02)

    # Règle 2 : < 1 kHz, gaspillage en R limité à WASTE_FRAC_MAX de la puissance HP
    mask = freqs < C.WASTE_FREQ_MAX
    if mask.any():
        ratio = totr[:, mask] / (totd[:, mask] + 1e-9)
        excess = torch.relu(ratio - C.WASTE_FRAC_MAX)
        pen2 = excess.pow(2).mean(dim=1) * (w * 5.0)
    else:
        pen2 = torch.zeros(B, device=freqs.device)
    return pen1 + pen2


def resistor_watts(root, vals, task):
    """Liste [(slot, R, P_watts)] pour le rapport de design (1 jeu de valeurs)."""
    vb = vals.unsqueeze(0)
    V = torch.full((1, len(task.freqs)), C.POWER_TEST_V, dtype=torch.complex64, device=task.freqs.device)
    out = []
    def walk(node, Vn):
        if isinstance(node, Comp) and node.kind == 'R':
            R = float(vals[node.slot]); p = (Vn.abs() ** 2 / R).max().item()
            out.append((node.slot, R, p))
        elif isinstance(node, Series):
            zl = compute_impedance(node.left, vb, task.freqs, task)
            zr = compute_impedance(node.right, vb, task.freqs, task)
            zt = zl + zr + 1e-15
            walk(node.left, Vn * (zl / zt)); walk(node.right, Vn * (zr / zt))
        elif isinstance(node, Parallel):
            walk(node.left, Vn); walk(node.right, Vn)
    walk(root, V)
    return out


def evaluate_topology(task, root, slots, n_restarts=128, steps=150, verbose=False):
    device = task.freqs.device
    F = len(task.freqs)
    N_comps = len(slots)

    circ = _CircView(device)
    reward_fn = make_reward(task, circ, slots)        # reward complet (incl. parcimonie)

    # Cas extrême : branchement direct sans composant
    if N_comps == 0:
        with torch.no_grad():
            resp, Zin = _responses(root, torch.empty((1, 0), device=device), task.freqs, task)
            circ.root._Z = Zin
            loss = reward_fn(resp, circ)
        return float(loss.min()), torch.empty(0, device=device), _DummyCirc(root, task, device, F), None

    # Init vectorisée en log-espace + bruit (exploration de plusieurs optima)
    init_vals = torch.tensor([v for (_, v) in slots], dtype=torch.float32, device=device)
    log_vals = torch.log10(init_vals).unsqueeze(0).expand(n_restarts, -1).clone()
    log_vals = log_vals + torch.randn_like(log_vals) * 0.5
    log_vals.requires_grad_(True)
    optimizer = torch.optim.Adam([log_vals], lr=0.1)

    loss = None
    for step in range(steps):
        optimizer.zero_grad()
        actual_vals = 10 ** log_vals
        resp, Zin = _responses(root, actual_vals, task.freqs, task)
        circ.root._Z = Zin
        loss = reward_fn(resp, circ)                  # [B]
        if any(k == 'R' for (k, _) in slots):
            loss = loss + thermal_penalty(root, actual_vals, task)
        loss.sum().backward()
        optimizer.step()
        with torch.no_grad():
            for i, (kind, _) in enumerate(slots):
                b_min, b_max = C.BOUNDS[kind]
                log_vals[:, i].clamp_(math.log10(b_min), math.log10(b_max))
        if verbose and step % 50 == 0:
            print(f"    step {step:4d}  loss_min={loss.min().item():.3f}")

    with torch.no_grad():
        best_idx = torch.argmin(loss)
        best_loss = loss[best_idx].item()
        best_vals = (10 ** log_vals[best_idx]).detach()
    return best_loss, best_vals, _DummyCirc(root, task, device, F), None


class _DummyCirc:
    """Objet minimal pour le tracé de design.py (forward -> {'P','V'} par driver)."""
    def __init__(self, root, task, device, F):
        self.root, self.task, self.device, self.F = root, task, device, F
        self.root._Z = None
    def forward(self, v):
        resp, Zin = _responses(self.root, v, self.task.freqs, self.task)
        self.root._Z = Zin
        return resp