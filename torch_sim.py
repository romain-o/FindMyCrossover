"""
torch_sim.py
============
Simulateur de filtre passif DIFFÉRENTIABLE et BATCHÉ, porté fidèlement depuis
`evaluator.py`, mais écrit en PyTorch pour tourner sur GPU (CUDA) et fournir des
gradients analytiques exacts via autograd.

Pourquoi ce fichier est la pierre angulaire du projet IA :
  - Votre `optimize_values` actuel utilise scipy L-BFGS-B avec des gradients par
    DIFFÉRENCES FINIES (eps=1e-3) et max_iter=5..8. C'est le goulot d'étranglement.
  - Ici, le même calcul d'impédance / division de tension est exprimé en torch :
    autograd donne d(loss)/d(valeurs) EXACT, et on peut optimiser des MILLIERS de
    candidats en parallèle sur une seule passe GPU (axe batch B + axe fréquence F).

Sémantique IDENTIQUE à evaluator.py (mêmes formules, mêmes "fudge factors") :
  R        : Z = value
  C        : Z = 1/(jw*value) + esr_c            (esr_c = 0.01 par défaut)
  L        : Z = jw*value + dcr_coeff*value      (dcr_coeff = 400.0 par défaut)
  Driver   : Z = Z_complex,  P_acoustic = V * H_acoustic
  Series   : Z = Zl + Zr  ; Vl = Vn*Zl/Ztot ; Vr = Vn*Zr/Ztot
  Parallel : Z = Zl*Zr/(Zl+Zr) ; Vl = Vr = Vn
  Shunt    : Z = Zc ; Vc = Vn

Le module est volontairement autonome (aucun import de `src.*`) pour être testable
sans vos CSV/FRD. Un adaptateur `from_legacy_tree` permet de convertir vos arbres
`nodes.py` par duck-typing (via le nom de classe), sans dépendance.
"""

from __future__ import annotations
import time
import torch

# ----------------------------------------------------------------------------
# 1. Représentation interne de l'arbre (miroir minimal de nodes.py)
# ----------------------------------------------------------------------------
class TNode:
    pass

class Comp(TNode):
    """Composant passif. kind in {'R','C','L'}. `slot` = index dans le vecteur de params."""
    def __init__(self, kind, slot):
        self.kind, self.slot = kind, slot

class Driver(TNode):
    def __init__(self, label):
        self.label = label

class Series(TNode):
    def __init__(self, left, right):
        self.left, self.right = left, right

class Parallel(TNode):
    def __init__(self, left, right):
        self.left, self.right = left, right

class Shunt(TNode):
    def __init__(self, child):
        self.child = child


def from_legacy_tree(node, slots):
    """
    Convertit un arbre de `nodes.py` (duck-typing par nom de classe) en arbre interne.
    `slots` est une liste mutable qui accumule la spec des composants : (kind, valeur_initiale).
    L'index dans `slots` devient le `slot` du Comp -> mappe vers le vecteur de paramètres.
    """
    cls = type(node).__name__
    if cls in ("SeriesNode",):
        return Series(from_legacy_tree(node.left, slots), from_legacy_tree(node.right, slots))
    if cls in ("ParallelNode",):
        return Parallel(from_legacy_tree(node.left, slots), from_legacy_tree(node.right, slots))
    if cls in ("ShuntNode",):
        return Shunt(from_legacy_tree(node.component, slots))
    if cls in ("Resistor", "Capacitor", "Inductor"):
        kind = {"Resistor": "R", "Capacitor": "C", "Inductor": "L"}[cls]
        slot = len(slots)
        slots.append((kind, float(node.value)))
        return Comp(kind, slot)
    if cls in ("DriverNode",):
        return Driver(node.label)
    raise ValueError(f"Type d'arbre inconnu: {cls}")


# ----------------------------------------------------------------------------
# 2. Compilation en plan plat (postfix pour Z, prefix pour V) -- comme compile_tree
# ----------------------------------------------------------------------------
class CompiledCircuit:
    """
    Détient le plan d'exécution + les buffers complexes des drivers.
    Les valeurs de composants sont fournies à chaque forward (axe batch supporté).
    """
    def __init__(self, root, comp_kinds, drivers, freqs, device,
                 esr_c=0.01, dcr_coeff=400.0, cdtype=torch.complex64):
        self.root = root
        self.comp_kinds = comp_kinds          # liste de 'R'/'C'/'L' par slot
        self.n_comp = len(comp_kinds)
        self.device = device
        self.esr_c = esr_c
        self.dcr_coeff = dcr_coeff
        self.cdtype = cdtype                   # complex64 (GPU/rapide) ou complex128 (validation)
        self.rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32

        self.freqs = freqs.to(device, self.rdtype)
        self.jw = 1j * 2 * torch.pi * self.freqs.to(cdtype)            # [F]
        # drivers : dict label -> (Z_complex[F], H_acoustic[F])
        self.drivers = {lbl: (Z.to(device, cdtype), H.to(device, cdtype))
                        for lbl, (Z, H) in drivers.items()}

        # Plans (postfix Z, prefix V), liste d'ids de noeuds dans l'ordre d'exécution
        self.z_plan, self.v_plan, self.driver_nodes = [], [], []
        self._build_plans(root)

    def _build_plans(self, root):
        def build_z(n):
            t = type(n)
            if t in (Series, Parallel):
                build_z(n.left); build_z(n.right)
                self.z_plan.append((t, n, n.left, n.right))
            elif t is Shunt:
                build_z(n.child)
                self.z_plan.append((t, n, n.child, None))
            else:
                self.z_plan.append((t, n, None, None))
                if t is Driver:
                    self.driver_nodes.append(n)
        build_z(root)

        def build_v(n):
            t = type(n)
            if t in (Series, Parallel):
                self.v_plan.append((t, n, n.left, n.right))
                build_v(n.left); build_v(n.right)
            elif t is Shunt:
                self.v_plan.append((t, n, n.child, None))
                build_v(n.child)
        build_v(root)

    # ------------------------------------------------------------------
    def forward(self, values, v_in=1.0):
        """
        values : tenseur réel [n_comp] ou [B, n_comp] (valeurs LINÉAIRES en SI).
        Retourne dict label -> {'V': [.,F], 'P': [.,F]} en complexe.
        Diffusion automatique batch/fréquence : Z de chaque noeud -> [..., F].
        """
        if values.dim() == 1:
            values = values.unsqueeze(0)           # -> [1, n_comp]
        B = values.shape[0]
        F = self.freqs.shape[0]
        jw = self.jw.view(1, F)                     # [1,F]
        cvals = values.to(self.cdtype)

        # --- PASSE 1 : impédances ---
        for t, n, l, r in self.z_plan:
            if t is Comp:
                v = cvals[:, n.slot:n.slot + 1]     # [B,1]
                if n.kind == 'R':
                    n._Z = v.expand(B, F)
                elif n.kind == 'C':
                    n._Z = (1.0 / (jw * v)) + self.esr_c
                else:  # 'L'
                    n._Z = (jw * v) + (self.dcr_coeff * v)
            elif t is Driver:
                Z, _ = self.drivers[n.label]
                n._Z = Z.view(1, F).expand(B, F)
            elif t is Series:
                n._Z = l._Z + r._Z
            elif t is Parallel:
                n._Z = (l._Z * r._Z) / (l._Z + r._Z + 1e-15)
            elif t is Shunt:
                n._Z = l._Z

        # --- PASSE 2 : tensions ---
        self.root._V = torch.as_tensor(v_in, dtype=self.cdtype, device=self.device).expand(B, F)
        for t, n, l, r in self.v_plan:
            if t is Series:
                ztot = n._Z + 1e-15
                l._V = n._V * (l._Z / ztot)
                r._V = n._V * (r._Z / ztot)
            elif t is Parallel:
                l._V = n._V
                r._V = n._V
            elif t is Shunt:
                l._V = n._V

        # --- PASSE 3 : récolte acoustique ---
        out = {}
        for d in self.driver_nodes:
            _, H = self.drivers[d.label]
            V = d._V
            out[d.label] = {"V": V, "P": V * H.view(1, F)}
        return out


# ----------------------------------------------------------------------------
# 3. Paramétrage différentiable + reward réduit (démo)
# ----------------------------------------------------------------------------
# Bornes physiques (identiques à optimizer.py)
BOUNDS = {'R': (0.1, 33.0), 'C': (0.1e-6, 300e-6), 'L': (0.05e-3, 5e-3)}

def init_log_params(slots_init, comp_kinds, B, device):
    """Paramètres optimisés = log10(valeur). Init bruitée pour B restarts indépendants."""
    theta = torch.empty(B, len(comp_kinds), device=device)
    for j, (kind, v0) in enumerate(slots_init):
        lo, hi = (torch.log10(torch.tensor(BOUNDS[kind][0])),
                  torch.log10(torch.tensor(BOUNDS[kind][1])))
        center = torch.log10(torch.tensor(max(v0, 1e-12)))
        noise = (torch.rand(B, device=device) - 0.5) * (hi - lo) * 0.30
        theta[:, j] = torch.clamp(center + noise, lo.item(), hi.item())
    return theta.clone().requires_grad_(True)

def linear_from_log(theta, comp_kinds, device):
    """log10 -> valeur SI, clampée dans les bornes (clamp dur, sous-différentiable)."""
    out = torch.empty_like(theta)
    for j, kind in enumerate(comp_kinds):
        lo, hi = BOUNDS[kind]
        out[:, j] = torch.clamp(10 ** theta[:, j], lo, hi)
    return out

def demo_reward(responses, freqs, target_spl, band_mask, Zin=None):
    """
    REWARD RÉDUIT pour la démo (à NE PAS confondre avec votre fitness complet).
    = MSE de platitude de la somme + petite pénalité d'impédance mini.
    Tout est différentiable en les valeurs des composants.
    """
    p_sum = None
    for d in responses.values():
        p_sum = d["P"] if p_sum is None else p_sum + d["P"]
    spl = 20 * torch.log10(p_sum.abs() + 1e-12)              # [B,F]
    diff = spl - target_spl
    # asymétrie douce : on punit 2x les dépassements (comme votre np.where)
    w = torch.where(diff > 0, 2.0, 1.0)
    mse = (w * diff.pow(2))[:, band_mask].mean(dim=1)        # [B]
    loss = mse
    if Zin is not None:
        minZ = Zin.abs().min(dim=1).values                  # [B]
        loss = loss + torch.relu(3.0 - minZ).pow(3) * 5.0
    return loss


def optimize_values_torch(circuit, slots_init, target_spl, band_mask,
                          n_restarts=256, steps=300, lr=0.05, verbose=True,
                          reward_fn=None):
    """
    Optimise les valeurs de composants pour UNE topologie, en BATCH (n_restarts) sur GPU.
    Remplace `optimize_values` (scipy, diff. finies) par Adam + gradients exacts.
    `reward_fn(responses, circuit) -> loss[B]` : si None, utilise demo_reward.
    Retourne (meilleures_valeurs_SI [n_comp], meilleure_loss, theta_final).
    """
    device = circuit.device
    if reward_fn is None:
        def reward_fn(resp, circ):
            return demo_reward(resp, circ.freqs, target_spl, band_mask, circ.root._Z)

    theta = init_log_params(slots_init, circuit.comp_kinds, n_restarts, device)
    opt = torch.optim.Adam([theta], lr=lr)

    t0 = time.time()
    for it in range(steps):
        opt.zero_grad()
        vals = linear_from_log(theta, circuit.comp_kinds, device)
        resp = circuit.forward(vals)
        loss = reward_fn(resp, circuit)
        loss.sum().backward()                                # somme : restarts indépendants
        opt.step()
        if verbose and (it % 50 == 0 or it == steps - 1):
            print(f"  it={it:4d}  loss_min={loss.min().item():.4f}  loss_med={loss.median().item():.4f}")
    dt = time.time() - t0

    with torch.no_grad():
        vals = linear_from_log(theta, circuit.comp_kinds, device)
        resp = circuit.forward(vals)
        loss = reward_fn(resp, circuit)
        best = torch.argmin(loss)
    if verbose:
        print(f"  [{n_restarts} restarts x {steps} steps] en {dt:.2f}s  "
              f"({n_restarts*steps/dt:,.0f} évals/s) sur {device}")
    return vals[best].detach(), loss[best].item(), theta.detach()


# ----------------------------------------------------------------------------
# 4. Drivers analytiques de synthèse (pour test sans vos fichiers FRD/ZMA)
# ----------------------------------------------------------------------------
def _butter2(f, fc, kind):
    """Réponse analogique Butterworth 2e ordre (complexe), passe-bas ou passe-haut."""
    s = 1j * (f / fc)
    if kind == 'lp':
        return 1.0 / (1 + 1.41421356 * s + s * s)
    else:  # 'hp'
        return (s * s) / (1 + 1.41421356 * s + s * s)

def _impedance(f, Re, Le, Fs, Rp, Qp):
    """Z = Re + jwLe + bosse de résonance (RLC parallèle simplifié)."""
    w = 2 * torch.pi * f
    res = Rp / (1 + 1j * Qp * (f / Fs - Fs / f))
    return Re + 1j * w * Le + res

def synth_two_way(F=400, device="cpu"):
    """Renvoie (freqs, drivers) : un woofer et un tweeter complexes plausibles."""
    freqs = torch.logspace(torch.log10(torch.tensor(20.)),
                           torch.log10(torch.tensor(20000.)), F)
    f = freqs.to(torch.complex64)
    # Woofer : passe-bande acoustique naturelle ~ [45 Hz .. 2.5 kHz], sensibilité ~90 dB
    Hw = (10 ** (90 / 20)) * _butter2(f, 45.0, 'hp') * _butter2(f, 2500.0, 'lp')
    Zw = _impedance(f, Re=6.0, Le=0.5e-3, Fs=45.0, Rp=40.0, Qp=2.0)
    # Tweeter : passe-haut naturel ~700 Hz, sensibilité ~92 dB
    Ht = (10 ** (92 / 20)) * _butter2(f, 700.0, 'hp')
    Zt = _impedance(f, Re=5.0, Le=0.05e-3, Fs=800.0, Rp=8.0, Qp=1.0)
    drivers = {"woofer": (Zw, Hw), "tweeter": (Zt, Ht)}
    return freqs, drivers


# ----------------------------------------------------------------------------
# 5. Référence NumPy (validation : torch doit matcher evaluator.py au bit près)
# ----------------------------------------------------------------------------
def _numpy_reference(circuit, values_np):
    """Réimplémente la passe forward en numpy pur (formules d'evaluator.py)."""
    import numpy as np
    freqs = circuit.freqs.cpu().numpy()
    jw = 1j * 2 * np.pi * freqs
    cache = {}
    def Z(n):
        t = type(n)
        if t is Comp:
            v = values_np[n.slot]
            if n.kind == 'R': return np.full_like(freqs, v, dtype=complex)
            if n.kind == 'C': return (1.0 / (jw * v)) + circuit.esr_c
            return (jw * v) + (circuit.dcr_coeff * v)
        if t is Driver:
            return circuit.drivers[n.label][0].cpu().numpy()
        if t is Series:   return Z(n.left) + Z(n.right)
        if t is Parallel:
            zl, zr = Z(n.left), Z(n.right); return (zl*zr)/(zl+zr+1e-15)
        if t is Shunt:    return Z(n.child)
    def V(n, vin):
        cache[id(n)] = vin
        t = type(n)
        if t is Series:
            ztot = Z(n) + 1e-15
            V(n.left, vin * Z(n.left)/ztot); V(n.right, vin * Z(n.right)/ztot)
        elif t is Parallel:
            V(n.left, vin); V(n.right, vin)
        elif t is Shunt:
            V(n.child, vin)
    V(circuit.root, np.ones_like(freqs, dtype=complex))
    out = {}
    for d in circuit.driver_nodes:
        H = circuit.drivers[d.label][1].cpu().numpy()
        out[d.label] = cache[id(d)] * H
    return out


# ----------------------------------------------------------------------------
# 6. Self-test
# ----------------------------------------------------------------------------
def _build_demo_topology(slots):
    """
    Topologie 2 voies classique (la racine DOIT être un Parallel, comme dans votre fitness):
      Woofer : L série -> (C shunt // woofer)                  = passe-bas 2e ordre
      Tweeter: C série -> (L shunt // [Rs -> (Rp // tweeter)]) = passe-haut 2e ordre + L-pad
    Le L-pad (Rs, Rp) permet d'atténuer le tweeter (92 dB) au niveau du woofer (90 dB).
    `slots` accumule (kind, val_init).
    """
    def comp(kind, v0):
        s = len(slots); slots.append((kind, v0)); return Comp(kind, s)
    woofer = Series(comp('L', 1.0e-3), Parallel(comp('C', 10e-6), Driver("woofer")))
    lpad = Series(comp('R', 3.3), Parallel(comp('R', 10.0), Driver("tweeter")))
    tweeter = Series(comp('C', 4.7e-6), Parallel(comp('L', 0.4e-3), lpad))
    return Parallel(woofer, tweeter)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  | torch {torch.__version__}\n")

    F = 400
    freqs, drivers = synth_two_way(F=F, device=device)

    slots = []
    root = _build_demo_topology(slots)
    comp_kinds = [k for (k, _) in slots]
    print(f"[circuit] {len(comp_kinds)} composants : {comp_kinds}")

    # --- (a) Validation numérique torch vs numpy, EN FLOAT64 (équivalence machine) ---
    import numpy as np
    circuit64 = CompiledCircuit(root, comp_kinds, drivers, freqs, device, cdtype=torch.complex128)
    vtest = torch.tensor([v for (_, v) in slots], device=device, dtype=torch.float64)
    with torch.no_grad():
        resp = circuit64.forward(vtest)
    ref = _numpy_reference(circuit64, np.array([v for (_, v) in slots]))
    err = max(np.max(np.abs(resp[lbl]["P"][0].cpu().numpy() - ref[lbl])) for lbl in ref)
    print(f"[check] écart max torch(f64) vs numpy (réf evaluator.py) : {err:.2e}  "
          f"{'OK' if err < 1e-6 else 'DIVERGENCE!'}\n")

    # --- (b) Optimisation batchée des valeurs (float32, rapide, CUDA-ready) ---
    circuit = CompiledCircuit(root, comp_kinds, drivers, freqs, device, cdtype=torch.complex64)
    target_spl = 89.0
    band_mask = (freqs.to(device) >= 80) & (freqs.to(device) <= 18000)
    print("[optimize] Adam, gradients exacts, batch de restarts :")
    best_vals, best_loss, _ = optimize_values_torch(
        circuit, slots, target_spl, band_mask,
        n_restarts=256, steps=400, lr=0.05)

    print(f"\n[result] meilleure loss = {best_loss:.4f}")
    for k, v in zip(comp_kinds, best_vals.cpu().tolist()):
        if k == 'R':  print(f"   R = {v:8.3f} Ohm")
        elif k == 'C': print(f"   C = {v*1e6:8.3f} µF")
        else:         print(f"   L = {v*1e3:8.3f} mH")


if __name__ == "__main__":
    main()
