"""
Canonical stratification of a regular CW complex by a Ch(Vect_F)-valued cellular
cosheaf, following the algorithm of Nanda (2019) with upward closure.

Conventions
-----------
* Cosheaves: extension maps run F(x >= y) : F(x) -> F(y) for y a face of x.
* Cochain complexes: d^n : C^n -> C^{n+1}.
* Matrices are lists of rows; M[i][j] with shape (rows, cols) = (dim target, dim source).
* Anything indexed outside a complex's support is zero.
* Arithmetic is exact (Fraction), so ranks are exact.
"""

from fractions import Fraction
from itertools import product


# ---------------------------------------------------------------- linear algebra

def zeros(rows, cols):
    return [[Fraction(0)] * cols for _ in range(rows)]


def shape(M):
    if not M:
        return (0, 0)
    return (len(M), len(M[0]) if M[0] else 0)


def rank(M):
    """Exact rank by Gaussian elimination over Q."""
    rows, cols = shape(M)
    if rows == 0 or cols == 0:
        return 0
    A = [row[:] for row in M]
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        pv = A[r][c]
        A[r] = [v / pv for v in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        r += 1
        if r == rows:
            break
    return r


def block(TL, TR, BL, BR, tl_shape, br_shape):
    """Assemble [[TL, TR], [BL, BR]] given the shapes of the diagonal blocks."""
    (t_rows, t_cols), (b_rows, b_cols) = tl_shape, br_shape
    TL = TL if TL else zeros(t_rows, t_cols)
    TR = TR if TR else zeros(t_rows, b_cols)
    BL = BL if BL else zeros(b_rows, t_cols)
    BR = BR if BR else zeros(b_rows, b_cols)
    top = [TL[i] + TR[i] for i in range(t_rows)]
    bot = [BL[i] + BR[i] for i in range(b_rows)]
    return top + bot


# ---------------------------------------------------------------- cochain complexes

class Complex:
    """A bounded cochain complex of finite-dimensional F-vector spaces.

    dims : dict n -> dim C^n   (absent => 0)
    diff : dict n -> matrix of d^n : C^n -> C^{n+1}   (absent => zero)
    """

    def __init__(self, dims=None, diff=None):
        self.dims = {n: d for n, d in (dims or {}).items() if d}
        self.diff = dict(diff or {})
        self._check()

    def _check(self):
        for n, M in self.diff.items():
            if not M:
                continue
            r, c = shape(M)
            assert c == self.dim(n), f"d^{n} has {c} columns, expected {self.dim(n)}"
            assert r == self.dim(n + 1), f"d^{n} has {r} rows, expected {self.dim(n+1)}"
        for n in self.degrees():
            comp = self.compose(n)
            assert all(v == 0 for row in comp for v in row), f"d^{n+1} d^{n} != 0"

    def compose(self, n):
        A, B = self.d(n), self.d(n + 1)
        rows, mid = shape(B)
        _, cols = shape(A)
        if rows == 0 or cols == 0 or mid == 0:
            return zeros(rows, cols)
        return [[sum(B[i][k] * A[k][j] for k in range(mid)) for j in range(cols)]
                for i in range(rows)]

    def dim(self, n):
        return self.dims.get(n, 0)

    def d(self, n):
        M = self.diff.get(n)
        if M:
            return M
        return zeros(self.dim(n + 1), self.dim(n))

    def degrees(self):
        return sorted(self.dims) or [0]

    def support(self):
        ds = self.degrees()
        return (min(ds), max(ds))

    def cohomology_dim(self, n):
        return self.dim(n) - rank(self.d(n)) - rank(self.d(n - 1))

    def __repr__(self):
        return f"Complex({dict(sorted(self.dims.items()))})"


ZERO = Complex()


class CochainMap:
    """A cochain map f : A -> B, given by matrices f^n : A^n -> B^n."""

    def __init__(self, source, target, comps=None):
        self.A, self.B = source, target
        self.comps = dict(comps or {})
        self._check()

    def _check(self):
        for n, M in self.comps.items():
            if not M:
                continue
            r, c = shape(M)
            assert c == self.A.dim(n), f"f^{n} has {c} columns, expected {self.A.dim(n)}"
            assert r == self.B.dim(n), f"f^{n} has {r} rows, expected {self.B.dim(n)}"
        lo = min(self.A.support()[0], self.B.support()[0]) - 1
        hi = max(self.A.support()[1], self.B.support()[1]) + 1
        for n in range(lo, hi + 1):
            assert self._commutes(n), f"f does not commute with d in degree {n}"

    def _commutes(self, n):
        # d_B^n f^n  ==  f^{n+1} d_A^n
        lhs = _mul(self.B.d(n), self.f(n))
        rhs = _mul(self.f(n + 1), self.A.d(n))
        r, c = self.B.dim(n + 1), self.A.dim(n)

        def at(M, i, j):
            # matrices of zero size lose their shape; treat missing entries as 0
            if i < len(M) and j < len(M[i]):
                return M[i][j]
            return 0

        for i in range(r):
            for j in range(c):
                if at(lhs, i, j) != at(rhs, i, j):
                    return False
        return True

    def f(self, n):
        M = self.comps.get(n)
        if M:
            return M
        return zeros(self.B.dim(n), self.A.dim(n))


def _mul(M, N):
    rows, mid = shape(M)
    mid2, cols = shape(N)
    if rows == 0 or cols == 0 or mid == 0:
        return zeros(rows, cols)
    return [[sum(M[i][k] * N[k][j] for k in range(mid)) for j in range(cols)]
            for i in range(rows)]


def identity_map(C):
    comps = {n: [[Fraction(int(i == j)) for j in range(C.dim(n))]
                 for i in range(C.dim(n))] for n in C.degrees()}
    return CochainMap(C, C, comps)


def zero_map(A, B):
    return CochainMap(A, B, {})


# ---------------------------------------------------------------- Subroutine: IsQuasiIsom

def is_quasi_isom(f):
    """TRUE iff the cochain map f is a quasi-isomorphism.

    Tests acyclicity of cone(f), where
        cone(f)^n = A^{n+1} (+) B^n,
        D^n = [[ -d_A^{n+1} , 0     ],
               [  f^{n+1}   , d_B^n ]].
    Over a field, a complex is exact iff
        dim C^n = rank D^n + rank D^{n-1}   for every n.
    """
    A, B = f.A, f.B
    lo = min(A.support()[0], B.support()[0]) - 2
    hi = max(A.support()[1], B.support()[1]) + 2

    C, D = {}, {}
    for n in range(lo, hi + 1):
        C[n] = A.dim(n + 1) + B.dim(n)
        D[n] = block(
            _neg(A.d(n + 1)), None, f.f(n + 1), B.d(n),
            (A.dim(n + 2), A.dim(n + 1)),
            (B.dim(n + 1), B.dim(n)),
        )

    for n in range(lo + 1, hi + 1):
        if rank(D[n]) + rank(D[n - 1]) != C[n]:
            return False
    return True


def _neg(M):
    return [[-v for v in row] for row in M]


# ---------------------------------------------------------------- cellular cosheaves

class Cosheaf:
    """A Ch(Vect_F)-valued cellular cosheaf on a regular CW complex.

    cells      : list of hashable cell labels
    dim        : dict cell -> dimension
    covers     : set of pairs (x, y) meaning x covers y, i.e. x |> y
    stalk      : dict cell -> Complex
    extension  : dict (x, y) -> CochainMap  F(x) -> F(y), for (x, y) in covers
    """

    def __init__(self, cells, dim, covers, stalk, extension):
        self.cells = list(cells)
        self.dim = dict(dim)
        self.covers = set(covers)
        self.stalk = dict(stalk)
        self.extension = dict(extension)
        for (x, y) in self.covers:
            assert self.dim[x] == self.dim[y] + 1, f"({x},{y}) is not codimension 1"
            assert (x, y) in self.extension, f"no extension map for ({x},{y})"

    def incoming(self, y):
        """The edges (x, y) with x |> y: the incoming star of y."""
        return {(x, yy) for (x, yy) in self.covers if yy == y}



# ---------------------------------------------------------------- Subroutine: ConnComponents

def conn_components(vertices, edges):
    """Partition of `vertices` into connected components of `edges`."""
    parent = {v: v for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for (v, w) in edges:
        rv, rw = find(v), find(w)
        if rv != rw:
            parent[rv] = rw

    blocks = {}
    for v in vertices:
        blocks.setdefault(find(v), set()).add(v)
    return sorted((sorted(b, key=str) for b in blocks.values()), key=str)