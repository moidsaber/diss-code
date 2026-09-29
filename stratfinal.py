"""
Stratify(H_X, F) -- connected strata satisfying the frontier condition.

Three stages:

  Layers   an edge survives a round iff it is a quasi-isomorphism and every
           edge above it survives; a cell is peeled when all its surviving-edge
           tests pass.  Peeled cells are removed and the next round runs on what
           is left, so the layers are the differences of a descending filtration
           and satisfy the frontier condition.

  Split    each layer is broken into connected components, since we want strata
           to be connected.

  Refine   splitting can break the frontier condition, so whenever a stratum A
           meets the closure of another stratum B without lying inside it, A is
           cut along that closure and re-split.  Refinement is monotone on a
           finite cell set, so this terminates.
"""

from stratify import is_quasi_isom, conn_components


def closure(C, covers):
    """Down-closure of a set of cells in the face poset."""
    o, changed = set(C), True
    while changed:
        changed = False
        for (y, w) in covers:
            if y in o and w not in o:
                o.add(w)
                changed = True
    return o


def components(A, covers):
    edges = {(y, w) for (y, w) in covers if y in A and w in A}
    return [set(c) for c in conn_components(A, edges)]


def layers(cs):
    """Stage 1: the descending filtration, in edge form."""
    Q = {e for e in cs.covers if is_quasi_isom(cs.extension[e])}
    into = {}
    for (y, w) in cs.covers:
        into.setdefault(w, set()).add((y, w))

    live, out = set(cs.cells), []
    while live:
        E = set()
        for d in range(max(cs.dim[c] for c in live), 0, -1):
            for (y, x) in cs.covers:
                if y in live and x in live and cs.dim[y] == d:
                    if (y, x) in Q and all(e in E for e in into.get(y, ())
                                           if e[0] in live):
                        E.add((y, x))
        S = {x for x in live
             if all(e in E for e in into.get(x, ()) if e[0] in live)}
        if not S:
            raise RuntimeError("no cell qualified")
        out.append(S)
        live -= S
    return out


def stratify(cs):
    """Stages 1-3.  Returns a list of strata, each a sorted list of cells."""
    P = [c for S in layers(cs) for c in components(S, cs.covers)]

    changed = True
    while changed:
        changed = False
        cls = {id(B): closure(B, cs.covers) for B in P}
        for A in P:
            for B in P:
                if A is B:
                    continue
                hit = A & cls[id(B)]
                if hit and hit != A:
                    P.remove(A)
                    P += components(hit, cs.covers)
                    P += components(A - hit, cs.covers)
                    changed = True
                    break
            if changed:
                break

    return sorted((sorted(a, key=str) for a in P), key=str)