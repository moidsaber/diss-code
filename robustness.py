"""
Persistent stratification over an alpha-complex filtration.

Given a finite point set P in R^d we take the alpha complex filtration
X_0 subset X_1 subset ... subset X_n, and at each level compute the local
cohomology cosheaf of that level, L_i = omega_{X_i}, extended by zero to X_n.

For a codimension-1 relation (x >= y) of X_n we record the levels at which
L_i(x >= y) is a quasi-isomorphism.  These form a union of maximal integer
intervals [a_j, b_j], and the robustness of the relation is

    r(x,y) = sum_j (b_j - a_j + 1) / (n - B(x) + 1),

where B(x) is the birth index of x.  Thresholding at epsilon gives an edge set
which is fed to the stratification algorithm in place of the quasi-isomorphism
test.
"""

import itertools
from fractions import Fraction

from stratify import is_quasi_isom
from localcohom import CWComplex
import stratfinal


# ------------------------------------------------------------------ filtration

def alpha_filtration(points, max_levels=None, alpha_max=None):
    """Return (cells, dim, incidence, birth) for the alpha complex of `points`.

    `birth` maps each cell to its index in the filtration; distinct alpha values
    are collapsed to consecutive integers.
    """
    import gudhi

    ac = gudhi.AlphaComplex(points=points)
    st = ac.create_simplex_tree()
    raw = [(tuple(sorted(s)), f) for s, f in st.get_filtration()]

    # Truncate at a geometrically meaningful scale.  The top of an alpha
    # filtration consists of sliver triangles whose circumradii vastly exceed
    # the diameter of the point cloud; including those levels swamps the
    # robustness scores.  A reasonable default is the largest nearest-neighbour
    # distance, so that every point is joined to at least one neighbour.
    if alpha_max is None:
        import numpy as np
        P = np.asarray(points, dtype=float)
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        np.fill_diagonal(D, np.inf)
        alpha_max = float(D.min(1).max()) ** 2
    raw = [(s, f) for s, f in raw if f <= alpha_max]

    vals = sorted({f for _, f in raw})
    if max_levels and len(vals) > max_levels:          # thin out the levels
        step = len(vals) / max_levels
        keep = [vals[min(len(vals) - 1, int(k * step))] for k in range(max_levels)]
        keep[-1] = vals[-1]
        vals = sorted(set(keep))
    index = {v: i for i, v in enumerate(vals)}

    def level(f):
        return min(i for v, i in index.items() if v >= f)

    name = lambda s: '-'.join(map(str, s))
    dim, birth = {}, {}
    for s, f in raw:
        dim[name(s)] = len(s) - 1
        birth[name(s)] = level(f)

    incidence = {}
    for s, _ in raw:
        if len(s) < 2:
            continue
        for i in range(len(s)):
            incidence[(name(s[:i] + s[i + 1:]), name(s))] = (-1) ** i

    return dim, incidence, birth, len(vals) - 1


# ------------------------------------------------------------------ robustness

def level_cosheaf(dim, incidence, birth, i):
    """L_i = local cohomology of the subcomplex X_i, as a cosheaf on X_i."""
    cells_i = {c: d for c, d in dim.items() if birth[c] <= i}
    inc_i = {(a, b): v for (a, b), v in incidence.items()
             if a in cells_i and b in cells_i}
    return CWComplex(cells_i, inc_i).local_cohomology()


def robustness(dim, incidence, birth, n, verbose=False):
    """Robustness score of every codimension-1 relation of X_n."""
    relations = [(b, a) for (a, b) in incidence]          # (coface, face)
    passes = {e: set() for e in relations}

    for i in range(n + 1):
        L = level_cosheaf(dim, incidence, birth, i)
        live = set(L.cells)
        for (x, y) in relations:
            if x in live:                                  # y is too: subcomplex
                if is_quasi_isom(L.extension[(x, y)]):
                    passes[(x, y)].add(i)
            elif y not in live:                            # 0 -> 0
                passes[(x, y)].add(i)
            else:                                          # 0 -> L_i(y)
                c = L.stalk[y]
                if all(c.cohomology_dim(k) == 0 for k in range(-1, max(dim.values()) + 2)):
                    passes[(x, y)].add(i)
        if verbose:
            print(f"   level {i}: {len(live)} cells")

    r = {}
    for (x, y) in relations:
        b = birth[x]
        window = [i for i in range(b, n + 1) if i in passes[(x, y)]]
        r[(x, y)] = Fraction(len(window), n - b + 1)
    return r


# ------------------------------------------------------------------ stratify

def stratify_robust(dim, incidence, r, epsilon):
    """Run the stratification with the quasi-isomorphism test replaced by
    'robustness >= epsilon'."""
    covers = {(b, a) for (a, b) in incidence}
    Q = {e for e, v in r.items() if v >= epsilon}

    into = {}
    for (y, w) in covers:
        into.setdefault(w, set()).add((y, w))

    live, out = set(dim), []
    while live:
        E = set()
        for d in range(max(dim[c] for c in live), 0, -1):
            for (y, x) in covers:
                if y in live and x in live and dim[y] == d:
                    if (y, x) in Q and all(e in E for e in into.get(y, ())
                                           if e[0] in live):
                        E.add((y, x))
        S = {x for x in live
             if all(e in E for e in into.get(x, ()) if e[0] in live)}
        if not S:
            raise RuntimeError("no cell qualified")
        out.append(S)
        live -= S

    class _P:                       # minimal shim for the refinement helpers
        pass
    shim = _P(); shim.covers = covers; shim.cells = list(dim); shim.dim = dim

    P = [c for S in out for c in stratfinal.components(S, covers)]
    changed = True
    while changed:
        changed = False
        cls = {id(B): stratfinal.closure(B, covers) for B in P}
        for A in P:
            for B in P:
                if A is B:
                    continue
                hit = A & cls[id(B)]
                if hit and hit != A:
                    P.remove(A)
                    P += stratfinal.components(hit, covers)
                    P += stratfinal.components(A - hit, covers)
                    changed = True
                    break
            if changed:
                break
    return sorted((sorted(a, key=str) for a in P), key=str)