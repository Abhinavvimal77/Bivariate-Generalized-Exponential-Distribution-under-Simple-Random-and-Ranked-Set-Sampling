import numpy as np
from scipy import integrate

def _A(x, lam):
    return np.clip(-np.expm1(-lam * np.asarray(x, float)), 0.0, 1.0)

def cdf(x1, x2, a1, a2, a3, lam):
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    return _A(x1, lam) ** a1 * _A(x2, lam) ** a2 * _A(np.minimum(x1, x2), lam) ** a3

def cdf_marg1(x, a1, a3, lam): return _A(x, lam) ** (a1 + a3)
def cdf_marg2(x, a2, a3, lam): return _A(x, lam) ** (a2 + a3)

def survival(x1, x2, a1, a2, a3, lam):
    return 1 - cdf_marg1(x1, a1, a3, lam) - cdf_marg2(x2, a2, a3, lam) + cdf(x1, x2, a1, a2, a3, lam)

def pdf_lt(x1, x2, a1, a2, a3, lam):          # region x1 < x2
    with np.errstate(all="ignore"):
        return ((a1 + a3) * a2 * lam**2 * _A(x1, lam) ** (a1 + a3 - 1) * _A(x2, lam) ** (a2 - 1)
                * np.exp(-lam * (np.asarray(x1) + np.asarray(x2))))

def pdf_gt(x1, x2, a1, a2, a3, lam):          # region x1 > x2
    with np.errstate(all="ignore"):
        return ((a2 + a3) * a1 * lam**2 * _A(x1, lam) ** (a1 - 1) * _A(x2, lam) ** (a2 + a3 - 1)
                * np.exp(-lam * (np.asarray(x1) + np.asarray(x2))))

def pdf_diag(x, a1, a2, a3, lam):             # singular part, density on x1 = x2 = x
    with np.errstate(all="ignore"):
        return a3 * lam * _A(x, lam) ** (a1 + a2 + a3 - 1) * np.exp(-lam * np.asarray(x))

def pdf_ac(x1, x2, a1, a2, a3, lam):          # absolutely continuous part (x1 != x2)
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float)
    return np.where(x1 < x2, pdf_lt(x1, x2, a1, a2, a3, lam), pdf_gt(x1, x2, a1, a2, a3, lam))

def p_tie(a1, a2, a3): return a3 / (a1 + a2 + a3)

def rge(n, a, lam, rng):                      # GE(a, lam) variates
    return -np.log1p(-rng.random(n) ** (1.0 / a)) / lam

def simulate(n, a1, a2, a3, lam, rng):        # exact: X1=max(U1,U3), X2=max(U2,U3)
    u1, u2, u3 = rge(n, a1, lam, rng), rge(n, a2, lam, rng), rge(n, a3, lam, rng)
    return np.maximum(u1, u3), np.maximum(u2, u3)

def mass_check(a1, a2, a3, lam):              # integrate the 3 parts in u = 1 - exp(-lam x) coordinates
    s = a1 + a2 + a3
    xo = lambda u: -np.log1p(-u) / lam
    jc = lambda u: 1.0 / (lam * (1.0 - u))
    f1 = lambda v, u: float(pdf_lt(xo(u), xo(v), a1, a2, a3, lam)) * jc(u) * jc(v)
    f2 = lambda v, u: float(pdf_gt(xo(u), xo(v), a1, a2, a3, lam)) * jc(u) * jc(v)
    m1 = integrate.dblquad(f1, 0, 1, lambda u: u, lambda u: 1)[0]
    m2 = integrate.dblquad(f2, 0, 1, lambda u: 0, lambda u: u)[0]
    m0 = integrate.quad(lambda u: float(pdf_diag(xo(u), a1, a2, a3, lam)) * jc(u), 0, 1)[0]
    return dict(mass_x1_lt_x2=m1, theory_lt=a2 / s, mass_x1_gt_x2=m2, theory_gt=a1 / s,
                mass_diag=m0, theory_diag=a3 / s, total=m1 + m2 + m0)