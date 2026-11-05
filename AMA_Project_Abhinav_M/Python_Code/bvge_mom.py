import numpy as np
from scipy.integrate import quad, dblquad

L = lambda u: -np.log1p(-np.minimum(u, 1 - 1e-16))      # x = L(u)/lam,  u = 1 - exp(-lam x)

def EG(g, a1, a2, a3, lam):
    """E[g(X1,X2)] over the three parts of the BVGE law, in u = 1-exp(-lam x) coordinates."""
    s = a1 + a2 + a3
    f1 = lambda v, u: g(L(u*v)/lam, L(u)/lam) * (a1+a3)*a2 * u**(s-1) * v**(a1+a3-1)   # x1 < x2
    f2 = lambda v, u: g(L(u)/lam, L(u*v)/lam) * a1*(a2+a3) * u**(s-1) * v**(a2+a3-1)   # x1 > x2
    f0 = lambda u: g(L(u)/lam, L(u)/lam) * a3 * u**(s-1)                                # x1 = x2
    kw = dict(epsabs=1e-10, epsrel=1e-10)
    return dblquad(f1, 0, 1, 0, 1, **kw)[0] + dblquad(f2, 0, 1, 0, 1, **kw)[0] + quad(f0, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]

def cm(r, v, aA, aB, a3, lam):
    """E[X_A^r | X_B = y], v = 1-exp(-lam*y). A=target (shape aA), B=conditioning (shape aB)."""
    y = L(v) / lam
    I1 = aB/(aB+a3) * (aA+a3) * v**aA * quad(lambda s: (L(v*s)/lam)**r * s**(aA+a3-1), 0, 1, epsabs=1e-11, epsrel=1e-11)[0]
    I2 = aA * quad(lambda u: (L(u)/lam)**r * u**(aA-1), v, 1, epsabs=1e-11, epsrel=1e-11)[0]
    return I1 + I2 + a3/(aB+a3) * v**aA * y**r