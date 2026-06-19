"""
tree_actions.py — Grammaire de construction d'un crossover 2 voies.

On construit DEUX branches (voie basse, voie haute) combinées en Parallel à la racine.
Chaque branche est une séquence d'actions lues DE LA SOURCE VERS LE DRIVER, construite
"à l'envers" (depuis le driver) :
    - SER_x : composant SÉRIE simple             -> Series(Comp_x, reste)
    - SH_x  : composant SHUNT simple à la masse   -> Parallel(Comp_x, reste)
    - SER_LC: TANK LC parallèle EN SÉRIE          -> Series(Parallel(L,C), reste)
              (bouchon dans le chemin : rejette une bande, utile contre un breakup)
    - SH_LC : TRAP LC série VERS LA MASSE         -> Parallel(Series(L,C), reste)
              (= vrai NOTCH : court-circuite une bande étroite à la masse)
    - SH_RC : ZOBEL R-C série vers la masse       -> Parallel(Series(R,C), reste)
              (aplatit la montée d'impédance due à l'inductance de bobine)
    - STOP  : termine la branche (le driver est posé)

Budget : `config.MAX_COMPONENTS_PER_BRANCH` borne le nombre de composants par branche.
Le masquage interdit tout token qui dépasserait le budget et force STOP quand il est
épuisé. Combiné à la pénalité de parcimonie du reward, l'IA n'utilise BEAUCOUP de
composants que si ça améliore vraiment le résultat (drivers difficiles).
"""
from __future__ import annotations
import config as C
from torch_sim import Series, Parallel, Comp, Driver as TDriver

# Description structurelle de chaque token : (forme, [types de composants]).
SPEC = {
    "SER_L":  ("SER",      ["L"]),
    "SER_C":  ("SER",      ["C"]),
    "SER_R":  ("SER",      ["R"]),
    "SH_L":   ("SH",       ["L"]),
    "SH_C":   ("SH",       ["C"]),
    "SH_R":   ("SH",       ["R"]),
    "SER_LC": ("SER_TANK", ["L", "C"]),   # tank LC parallèle en série (bouchon)
    "SH_LC":  ("SH_TRAP",  ["L", "C"]),   # trap LC série à la masse (notch)
    "SH_RC":  ("SH_TRAP",  ["R", "C"]),   # Zobel R-C série à la masse
}
VOCAB = list(SPEC.keys()) + ["STOP"]
VOCAB_ID = {t: i for i, t in enumerate(VOCAB)}
STOP = VOCAB_ID["STOP"]
N_ACTIONS = len(VOCAB)


def _cost(tok):
    """Nombre de composants physiques qu'ajoute un token."""
    return 0 if tok == "STOP" else len(SPEC[tok][1])


MAX_TOKENS_PER_BRANCH = getattr(C, "MAX_COMPONENTS_PER_BRANCH", 6)


def _new_comp(kind, slots):
    slot = len(slots); slots.append((kind, C.INIT_VALUE[kind]))
    return Comp(kind, slot)


def build_branch(tokens, driver_label, slots):
    """tokens (sans STOP), lus source->driver -> sous-arbre terminé par le driver."""
    node = TDriver(driver_label)
    for tok in reversed(tokens):
        form, kinds = SPEC[tok]
        if form == "SER":
            node = Series(_new_comp(kinds[0], slots), node)
        elif form == "SH":
            node = Parallel(_new_comp(kinds[0], slots), node)
        elif form == "SER_TANK":
            tank = Parallel(_new_comp(kinds[0], slots), _new_comp(kinds[1], slots))
            node = Series(tank, node)
        elif form == "SH_TRAP":
            sub = _new_comp(kinds[-1], slots)
            for k in reversed(kinds[:-1]):
                sub = Series(_new_comp(k, slots), sub)
            node = Parallel(sub, node)
    return node


def build_topology(low_tokens, high_tokens):
    """Arbre complet 2 voies + spec des slots. Racine Parallel obligatoire."""
    slots = []
    low = build_branch(low_tokens, "low", slots)
    high = build_branch(high_tokens, "high", slots)
    return Parallel(low, high), slots


def valid_mask(tokens_so_far):
    """
    Masque bool [N_ACTIONS] : interdit tout token qui dépasserait le budget de
    composants de la branche ; force STOP quand le budget est épuisé.
    """
    budget = getattr(C, "MAX_COMPONENTS_PER_BRANCH", 6)
    used = sum(_cost(t) for t in tokens_so_far)
    remaining = budget - used
    if remaining <= 0:
        return [i == STOP for i in range(N_ACTIONS)]
    mask = [True] * N_ACTIONS
    for i, tok in enumerate(VOCAB):
        if tok == "STOP":
            continue
        if _cost(tok) > remaining:
            mask[i] = False
        if tokens_so_far and tok == tokens_so_far[-1] and SPEC[tok][0] == "SH":
            mask[i] = False
    if not any(mask[i] for i in range(N_ACTIONS) if i != STOP):
        return [i == STOP for i in range(N_ACTIONS)]
    return mask


# Templates experts (cold start) — du plus SIMPLE au plus COMPLEXE.
TEMPLATES = [
    (["SER_L"],                          ["SER_C"]),                                  # 1er ordre
    (["SER_L", "SH_C"],                  ["SER_C", "SH_L"]),                          # 2e ordre
    (["SER_L", "SH_C"],                  ["SER_C", "SH_L", "SER_R", "SH_R"]),         # 2e + L-pad
    (["SER_L", "SH_C", "SER_L"],         ["SER_C", "SH_L"]),                          # 3e / 2e
    (["SER_L", "SH_C", "SER_L"],         ["SER_C", "SH_L", "SER_C"]),                 # 3e / 3e
    (["SER_L", "SH_C", "SH_RC"],         ["SER_C", "SH_L", "SER_R", "SH_R"]),         # Zobel woofer + L-pad
    (["SER_L", "SH_C"],                  ["SER_C", "SH_L", "SER_LC"]),                # tweeter + tank série (anti-breakup)
    (["SER_L", "SH_C", "SH_LC"],         ["SER_C", "SH_L", "SER_R", "SH_R"]),         # NOTCH (trap) sur le woofer
]


def template_token_ids(template):
    low, high = template
    return ([VOCAB_ID[t] for t in low] + [STOP],
            [VOCAB_ID[t] for t in high] + [STOP])