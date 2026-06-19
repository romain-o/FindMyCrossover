"""
data_pipeline.py — Gestion complète des données d'entraînement.

Chaîne :
  fichiers .frd/.zma  ->  courbes brutes  ->  rééchantillonnage sur grille commune
  ->  (augmentation)  ->  Task (= 1 problème de crossover 2 voies Woofer/Mid)

Concepts clés
-------------
- DriverRaw   : un HP = un fichier FRD (SPL dB + phase) + un fichier ZMA (|Z| + phase).
- Driver      : version rééchantillonnée sur la grille geomspace commune, en COMPLEXE
                (H_acoustic, Z_complex), prête pour torch_sim. Miroir de _prepare_driver().
- Task        : {freqs, drivers={'low':(Z,H), 'high':(Z,H)}, target_spl, band_mask, ...}
                C'est l'unité que consomment l'environnement RL et le reward.
- TaskSampler : produit des Task à la volée, avec augmentation, en tenant un split
                train/val basé sur des drivers RÉELS jamais vus en entraînement.

Pourquoi l'augmentation : on a peu de HP réels. On les multiplie en milliers de tâches
plausibles (interpolation intra-classe, décalage de niveau, warp fréquentiel léger,
lissage, bruit de mesure). On augmente UNIQUEMENT à l'intérieur d'une classe
(woofer×woofer) — mélanger un woofer et un tweeter serait non physique.
"""
from __future__ import annotations
import os, glob, hashlib, random
from dataclasses import dataclass, field
import numpy as np
import torch

import config as C


# ============================================================ 1. Lecture brute
def load_curve(filepath):
    """Lit un fichier 3 colonnes (freq, mag, phase). Robuste (= DriverNode.load_data)."""
    data = np.genfromtxt(filepath, usecols=(0, 1, 2), invalid_raise=False, comments="*")
    if data.ndim == 1:
        rows = []
        with open(filepath) as f:
            for line in f:
                p = line.split()
                if len(p) >= 3:
                    try: rows.append([float(p[0]), float(p[1]), float(p[2])])
                    except ValueError: continue
        data = np.array(rows)
    if data.size:
        data = data[~np.isnan(data).any(axis=1)]
    if data.size == 0:
        raise ValueError(f"Fichier illisible / vide : {filepath}")
    return data[:, 0], data[:, 1], data[:, 2]


@dataclass
class DriverRaw:
    name: str
    cls: str
    frd_path: str
    zma_path: str


def _stem(path, exts):
    """Nom de base sans extension ni suffixe d'angle (conserve la casse)."""
    base = os.path.basename(path)
    low = base.lower()
    for e in exts:
        if low.endswith(e):
            base = base[: -len(e)]
            break
    for suf in ("_0deg", "_0 deg", "_0", "@0", " 0deg", "-0deg", "_h0", "_v0"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
    return base.strip()


def _resolve_class_dir(root, name):
    """Trouve le dossier d'une classe de façon tolérante (pluriel/casse/préfixe)."""
    cand = os.path.join(root, name)
    if os.path.isdir(cand):
        return cand
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            dl, nl = d.lower(), name.lower()
            if dl == nl or dl.startswith(nl) or nl.startswith(dl):
                return full
    return None


def _is_onaxis(path):
    p = path.replace("\\", "/").lower()
    return ("0deg" in p) or ("/0/" in p) or ("on axis" in p) or ("onaxis" in p) or ("_0deg" in p)


def _walk_files(base, exts):
    hits = []
    for r, _, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith(exts):
                hits.append(os.path.join(r, fn))
    return hits


def discover_drivers(root=None, classes=None, verbose=False):
    """
    Scanne data/<Classe>/ de façon RÉCURSIVE et apparie chaque FRD avec son ZMA.
    Gère l'arborescence imbriquée type VituixCAD :
        <Classe>/FRD/0deg/nom.frd   +   <Classe>/ZMA/0deg/nom.zma
    Préfère les mesures on-axis (0deg) si plusieurs angles existent.
    Apparie par nom de base (insensible à la casse).
    """
    root = root or C.DATA_ROOT
    classes = classes or C.CLASSES
    out = {cls: [] for cls in classes}
    for cls in classes:
        cdir = _resolve_class_dir(root, cls)
        if cdir is None:
            if verbose: print(f"  [!] dossier introuvable pour '{cls}' sous {root}")
            continue
        frds = _walk_files(cdir, C.FRD_EXTS)
        zmas = _walk_files(cdir, C.ZMA_EXTS)
        fr_on = [f for f in frds if _is_onaxis(f)]
        zm_on = [z for z in zmas if _is_onaxis(z)]
        if fr_on: frds = fr_on            # on ne garde l'on-axis que s'il existe
        if zm_on: zmas = zm_on

        zmap = {}
        for z in zmas:
            zmap.setdefault(_stem(z, C.ZMA_EXTS).lower(), z)
        for f in sorted(frds):
            st = _stem(f, C.FRD_EXTS)
            z = zmap.get(st.lower())
            if z is None and len(zmas) == 1:
                z = zmas[0]
            if z is None:
                if verbose: print(f"  [!] {cls}: ZMA manquant pour '{st}'")
                continue
            out[cls].append(DriverRaw(name=st, cls=cls, frd_path=f, zma_path=z))
        if verbose:
            print(f"  {cls}: {os.path.basename(cdir)}/  FRD={len(frds)} ZMA={len(zmas)} -> {len(out[cls])} paires")
    return out


# ============================================================ 2. Rééchantillonnage
def make_grid(device=None):
    f = np.geomspace(C.F_MIN, C.F_MAX, C.N_FREQ)
    return torch.tensor(f, dtype=torch.float64, device=device or C.DEVICE)


@dataclass
class Driver:
    name: str
    cls: str
    H: torch.Tensor   # complexe [F]  réponse acoustique
    Z: torch.Tensor   # complexe [F]  impédance
    sens_db: float    # sensibilité moyenne en bande passante (dB)


def passband_sens(mag, fr):
    """
    Sensibilité = niveau du PLATEAU de bande passante, robuste quel que soit le type
    de driver (sub, woofer, médium, tweeter, AMT). On lisse la magnitude, on prend la
    référence haute, et on moyenne les points à moins de 3 dB sous ce plateau.
    """
    k = max(3, len(mag) // 40)
    sm = np.convolve(mag, np.ones(k) / k, mode="same")
    ref = np.percentile(sm, 92)               # niveau haut robuste (ignore les bosses)
    band = sm >= (ref - 3.0)
    return float(np.mean(mag[band])) if band.any() else float(np.mean(mag))


def passband_edges(H, fr, drop_db=10.0):
    """(f_lo, f_hi) où le driver est à moins de `drop_db` sous son plateau."""
    mag = 20 * np.log10(np.abs(H.cpu().numpy()) + 1e-12)
    ref = passband_sens(mag, fr.cpu().numpy())
    inb = np.where(mag >= ref - drop_db)[0]
    f = fr.cpu().numpy()
    if len(inb) == 0:
        return f[0], f[-1]
    return float(f[inb[0]]), float(f[inb[-1]])


def is_clean(drv) -> bool:
    """Driver exploitable : impédance plausible, pas de NaN, sensibilité réaliste."""
    Zmin = float(drv.Z.abs().min()); Zmax = float(drv.Z.abs().max())
    if np.isnan(drv.H.abs().cpu().numpy()).any() or np.isnan(drv.Z.abs().cpu().numpy()).any():
        return False
    if not (2.0 <= Zmin <= 60.0):      # ZMA aberrant (ex: 1.1 Ω ou 111 Ω)
        return False
    if Zmax > 400.0:
        return False
    if not (65.0 <= drv.sens_db <= 110.0):
        return False
    return True


def resample_driver(raw: DriverRaw, freqs: torch.Tensor) -> Driver:
    """Interpole sur la grille commune en dB + phase déroulée (= _prepare_driver)."""
    fr = freqs.cpu().numpy()
    f_f, spl_db, spl_ph = load_curve(raw.frd_path)
    f_z, z_mag, z_ph = load_curve(raw.zma_path)

    mag = np.interp(fr, f_f, spl_db)
    ph = np.interp(fr, f_f, np.unwrap(np.deg2rad(spl_ph)))
    H = (10 ** (mag / 20.0)) * np.exp(1j * ph)

    zm = np.interp(fr, f_z, z_mag)
    zp = np.interp(fr, f_z, np.unwrap(np.deg2rad(z_ph)))
    Z = zm * np.exp(1j * zp)

    sens = passband_sens(mag, fr)
    dev = freqs.device
    return Driver(raw.name, raw.cls,
                  torch.tensor(H, dtype=torch.complex64, device=dev),
                  torch.tensor(Z, dtype=torch.complex64, device=dev),
                  sens)


# ============================================================ 3. Augmentation
def _logmag_phase(H):
    mag = 20 * torch.log10(H.abs() + 1e-10)
    ph = torch.angle(H)
    return mag, ph

def _from_logmag_phase(mag, ph):
    return (10 ** (mag / 20.0)).to(torch.complex64) * torch.exp(1j * ph.to(torch.complex64))

def aug_level(drv: Driver, db: float) -> Driver:
    """Décalage de sensibilité (n'affecte que l'acoustique)."""
    return Driver(drv.name, drv.cls, drv.H * (10 ** (db / 20.0)), drv.Z, drv.sens_db + db)

def aug_warp(drv: Driver, freqs: torch.Tensor, ratio: float) -> Driver:
    """Léger étirement fréquentiel (±quelques %) de H et Z (resonances décalées)."""
    fr = freqs.cpu().numpy()
    src = fr * ratio
    def warp(c):
        cc = c.cpu().numpy()
        mag = np.interp(fr, src, np.abs(cc))
        ph = np.interp(fr, src, np.unwrap(np.angle(cc)))
        return torch.tensor(mag * np.exp(1j * ph), dtype=torch.complex64, device=freqs.device)
    return Driver(drv.name, drv.cls, warp(drv.H), warp(drv.Z), drv.sens_db)

def aug_noise(drv: Driver, db_sigma=0.2) -> Driver:
    """Bruit de mesure léger sur la magnitude acoustique."""
    mag, ph = _logmag_phase(drv.H)
    mag = mag + torch.randn_like(mag) * db_sigma
    return Driver(drv.name, drv.cls, _from_logmag_phase(mag, ph), drv.Z, drv.sens_db)

def mix_within_class(d1: Driver, d2: Driver, alpha: float) -> Driver:
    """Interpolation intra-classe en dB + phase déroulée (physiquement raisonnable)."""
    assert d1.cls == d2.cls, "Le mélange ne se fait QU'À l'intérieur d'une classe."
    m1, p1 = _logmag_phase(d1.H); m2, p2 = _logmag_phase(d2.H)
    H = _from_logmag_phase((1 - alpha) * m1 + alpha * m2,
                           (1 - alpha) * p1 + alpha * p2)
    zm1, zp1 = _logmag_phase(d1.Z); zm2, zp2 = _logmag_phase(d2.Z)
    Z = _from_logmag_phase((1 - alpha) * zm1 + alpha * zm2,
                           (1 - alpha) * zp1 + alpha * zp2)
    return Driver(f"{d1.name}~{d2.name}", d1.cls, H, Z,
                  (1 - alpha) * d1.sens_db + alpha * d2.sens_db)

def augment(drv: Driver, freqs: torch.Tensor, rng: random.Random) -> Driver:
    """Pipeline d'augmentation stochastique appliqué à un driver."""
    if rng.random() < 0.7: drv = aug_level(drv, rng.uniform(-2.0, 2.0))
    if rng.random() < 0.5: drv = aug_warp(drv, freqs, rng.uniform(0.95, 1.05))
    if rng.random() < 0.5: drv = aug_noise(drv, rng.uniform(0.05, 0.3))
    return drv


# ============================================================ 4. Tâches
@dataclass
class Task:
    freqs: torch.Tensor
    drivers: dict          # {'low': (Z,H), 'high': (Z,H)}
    labels: list           # ['low', 'high'] dans l'ordre bas->haut
    target_spl: float
    band_mask: torch.Tensor
    meta: dict = field(default_factory=dict)


def crossover_compatible(low: Driver, high: Driver, freqs: torch.Tensor):
    """
    Deux drivers forment un 2-voies SENSÉ si :
      - ils se recouvrent dans une zone de coupure plausible (800–6000 Hz),
      - le tweeter peut atteindre le niveau du woofer (sens_high >= sens_low - 2 dB),
      - l'écart de sensibilité reste réaliste (<= 18 dB d'atténuation par L-pad).
    Évite d'apparier p.ex. un 12" subwoofer avec un driver à compression.
    """
    lo_hi = passband_edges(low.H, freqs)[1]    # le woofer doit monter assez haut
    hi_lo = passband_edges(high.H, freqs)[0]   # le tweeter doit descendre assez bas
    overlap = (lo_hi > 1000.0) and (hi_lo < 5000.0) and (hi_lo < lo_hi)
    level_ok = (high.sens_db >= low.sens_db - 2.0) and (high.sens_db - low.sens_db <= 18.0)
    return bool(overlap and level_ok)


def build_task(low: Driver, high: Driver, freqs: torch.Tensor) -> Task:
    """Assemble une tâche 2 voies. Cible SPL = sensibilité du WOOFER (le tweeter est atténué)."""
    target = low.sens_db - C.TARGET_OFFSET_DB
    fr = freqs
    lo_lo, lo_hi = passband_edges(low.H, freqs)
    hi_lo, hi_hi = passband_edges(high.H, freqs)
    fmin = max(min(lo_lo, hi_lo), C.BAND_MIN)
    fmax = min(max(lo_hi, hi_hi), C.BAND_MAX)
    if fmax <= fmin:
        fmin, fmax = C.BAND_MIN, C.BAND_MAX
    band = (fr >= fmin) & (fr <= fmax)
    return Task(freqs=fr,
                drivers={"low": (low.Z, low.H), "high": (high.Z, high.H)},
                labels=["low", "high"],
                target_spl=target, band_mask=band,
                meta={"low": low.name, "high": high.name, "fmin": fmin, "fmax": fmax,
                      "sens_low": low.sens_db, "sens_high": high.sens_db})


class TaskSampler:
    """
    Source de tâches d'entraînement / validation.
    - charge et rééchantillonne tous les drivers réels (avec cache disque),
    - réserve un sous-ensemble de drivers réels pour la VALIDATION (jamais augmentés),
    - échantillonne des paires (low, high) avec augmentation pour l'entraînement.
    """
    def __init__(self, root=None, device=None, val_frac=0.2, seed=0, use_cache=True,
                 skip_suspect=True):
        self.device = device or C.DEVICE
        self.freqs = make_grid(self.device)
        self.rng = random.Random(seed)
        raws = discover_drivers(root)
        self.real = {cls: [self._load_cached(r, use_cache) for r in raws[cls]] for cls in raws}

        # Hygiène : on écarte les drivers à impédance aberrante (ZMA douteux)
        self.dropped = {}
        if skip_suspect:
            for cls in self.real:
                keep, drop = [], []
                for d in self.real[cls]:
                    (keep if is_clean(d) else drop).append(d)
                self.real[cls] = keep
                self.dropped[cls] = [d.name for d in drop]

        # split train/val par NOM de driver (généralisation honnête)
        self.train, self.val = {}, {}
        for cls, lst in self.real.items():
            lst = list(lst); self.rng.shuffle(lst)
            n_val = max(1, int(len(lst) * val_frac)) if len(lst) > 2 else 0
            self.val[cls] = lst[:n_val]
            self.train[cls] = lst[n_val:] if len(lst) > n_val else lst

        # index par nom (pour reconstruire une tâche depuis un dataset récolté)
        self.by_name = {d.name: d for cls in self.real for d in self.real[cls]}
        self.val_name_set = {d.name for cls in self.val for d in self.val[cls]}

    def is_val_pair(self, low_name, high_name):
        """Une paire est 'validation' si AU MOINS un driver est dans le set réservé."""
        return (low_name in self.val_name_set) or (high_name in self.val_name_set)

    def task_from_names(self, low_name, high_name, augment_data=False):
        low, high = self.by_name[low_name], self.by_name[high_name]
        if augment_data:
            low = augment(low, self.freqs, self.rng)
            high = augment(high, self.freqs, self.rng)
        return build_task(low, high, self.freqs)

    def _load_cached(self, raw: DriverRaw, use_cache):
        if not use_cache:
            return resample_driver(raw, self.freqs)
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        key = hashlib.md5(f"{raw.frd_path}|{raw.zma_path}|{C.N_FREQ}".encode()).hexdigest()[:16]
        path = os.path.join(C.CACHE_DIR, f"drv_{raw.cls}_{key}.pt")
        if os.path.exists(path):
            d = torch.load(path, weights_only=False)
            return Driver(d["name"], d["cls"], d["H"].to(self.device), d["Z"].to(self.device), d["sens"])
        drv = resample_driver(raw, self.freqs)
        torch.save({"name": drv.name, "cls": drv.cls, "H": drv.H.cpu(),
                    "Z": drv.Z.cpu(), "sens": drv.sens_db}, path)
        return drv

    def _pick(self, pool, cls):
        return self.rng.choice(pool[cls])

    def _compatible_pair(self, pool, max_tries=40):
        """Tire un couple (low, high) physiquement sensé (rejet sinon)."""
        last = None
        for _ in range(max_tries):
            low, high = self._pick(pool, C.ROLE_LOW), self._pick(pool, C.ROLE_HIGH)
            last = (low, high)
            if crossover_compatible(low, high, self.freqs):
                return low, high
        return last     # à défaut, on renvoie le dernier (évite une boucle infinie)

    def sample_train(self, augment_data=True) -> Task:
        low, high = self._compatible_pair(self.train)
        if augment_data:
            # 30% : interpolation intra-classe avec un autre driver de la même classe
            if len(self.train[C.ROLE_LOW]) > 1 and self.rng.random() < 0.3:
                low = mix_within_class(low, self._pick(self.train, C.ROLE_LOW), self.rng.random())
            if len(self.train[C.ROLE_HIGH]) > 1 and self.rng.random() < 0.3:
                high = mix_within_class(high, self._pick(self.train, C.ROLE_HIGH), self.rng.random())
            low = augment(low, self.freqs, self.rng)
            high = augment(high, self.freqs, self.rng)
        return build_task(low, high, self.freqs)

    def sample_val(self) -> Task:
        low, high = self._compatible_pair(self.val if self.val.get(C.ROLE_LOW) else self.train)
        return build_task(low, high, self.freqs)

    def stats(self):
        return {cls: {"train": len(self.train.get(cls, [])), "val": len(self.val.get(cls, []))}
                for cls in C.CLASSES}


# ============================================================ 5. Générateur synthétique (tests)
def write_synthetic_dataset(root, n_per_class=4, seed=0, nested=False):
    """Écrit de VRAIS fichiers .frd/.zma synthétiques pour tester le pipeline sans data réelle.
    nested=True reproduit l'arborescence type VituixCAD : <Classe>s/FRD/0deg/ et /ZMA/0deg/."""
    rng = np.random.default_rng(seed)
    f = np.geomspace(20, 20000, 300)
    def butter2(fc, kind):
        s = 1j * (f / fc)
        return 1/(1+1.4142*s+s*s) if kind == "lp" else (s*s)/(1+1.4142*s+s*s)
    def imp(Re, Le, Fs, Rp, Qp):
        w = 2*np.pi*f
        return Re + 1j*w*Le + Rp/(1+1j*Qp*(f/Fs - Fs/f))
    specs = {
        "Woofer": lambda: (90+rng.uniform(-2,2), butter2(rng.uniform(40,55),"hp")*butter2(rng.uniform(1800,3000),"lp"),
                           imp(rng.uniform(5.5,6.5), rng.uniform(0.4,0.7)*1e-3, rng.uniform(38,55), 35, 2.0)),
        "Mid":    lambda: (89+rng.uniform(-2,2), butter2(rng.uniform(250,400),"hp")*butter2(rng.uniform(4000,6000),"lp"),
                           imp(rng.uniform(5.0,6.0), rng.uniform(0.1,0.3)*1e-3, rng.uniform(250,400), 12, 1.2)),
        "Tweeter":lambda: (92+rng.uniform(-2,2), butter2(rng.uniform(600,900),"hp"),
                           imp(rng.uniform(4.5,5.5), rng.uniform(0.03,0.07)*1e-3, rng.uniform(700,1000), 8, 1.0)),
    }
    for cls, gen in specs.items():
        if nested:
            folder = os.path.join(root, cls + "s")  # Woofers, Mids, Tweeters
            frd_dir = os.path.join(folder, "FRD", "0deg")
            zma_dir = os.path.join(folder, "ZMA", "0deg")
        else:
            frd_dir = zma_dir = os.path.join(root, cls)
        os.makedirs(frd_dir, exist_ok=True); os.makedirs(zma_dir, exist_ok=True)
        for i in range(n_per_class):
            sens, H, Z = gen()
            H = H * (10**(sens/20))
            spl = 20*np.log10(np.abs(H)+1e-12); sph = np.degrees(np.angle(H))
            zmag = np.abs(Z); zph = np.degrees(np.angle(Z))
            np.savetxt(os.path.join(frd_dir, f"{cls}_{i:02d}.frd"),
                       np.column_stack([f, spl, sph]), fmt="%.3f", header="Freq SPL Phase", comments="* ")
            np.savetxt(os.path.join(zma_dir, f"{cls}_{i:02d}.zma"),
                       np.column_stack([f, zmag, zph]), fmt="%.4f", header="Freq |Z| Phase", comments="* ")
    return root