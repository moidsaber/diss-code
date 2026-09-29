"""Robustness over an alpha-complex filtration: a circle with a whisker."""
import math
from fractions import Fraction
from robustness import alpha_filtration, robustness, stratify_robust

pts  = [[math.cos(2*math.pi*k/12), math.sin(2*math.pi*k/12)] for k in range(12)]
pts += [[1.0 + 0.35*j, 0.0] for j in (1, 2)]          # spike off the circle

dim, inc, birth, n = alpha_filtration(pts, max_levels=8)
print(f"{len(dim)} cells, {n+1} filtration levels")

r = robustness(dim, inc, birth, n)
print("robustness values:", sorted({str(v) for v in r.values()}))

for eps in [Fraction(0), Fraction(1,4), Fraction(1,2), Fraction(3,4), Fraction(1)]:
    S = stratify_robust(dim, inc, r, eps)
    print(f"  eps = {eps}:  {len(S)} strata, sizes {sorted((len(s) for s in S), reverse=True)[:6]}")