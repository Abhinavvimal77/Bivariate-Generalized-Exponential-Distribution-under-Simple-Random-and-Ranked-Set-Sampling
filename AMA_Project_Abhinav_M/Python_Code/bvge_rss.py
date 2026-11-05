import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def sim_pairs(shape, th, rng):
    a = np.array(th[:3], float).reshape((3,) + (1,)*len(shape)); V = rng.random((3,) + shape)
    U = -np.log1p(-V**(1/a))/th[3]; return np.maximum(U[0], U[2]), np.maximum(U[1], U[2])
def sim_srs(n, th, rng):
    x1, x2 = sim_pairs((n,), th, rng); return (x1, x2, np.ones(n))
def sim_rss(m, c, th, rng):
    X1, X2 = sim_pairs((c, m, m), th, rng); o = np.argsort(X1, axis=2)
    X1s, X2s = np.take_along_axis(X1, o, 2), np.take_along_axis(X2, o, 2); ar = np.arange(m)
    return X1s[:, ar, ar].ravel(), X2s[:, ar, ar].ravel(), np.tile(ar+1.0, (c, 1)).ravel()

def loglik(t, d, m):
    a1, a2, a3, lam = t; x1, x2, r = d
    lA1 = np.log(-np.expm1(-lam*x1)); lA2 = np.log(-np.expm1(-lam*x2)); b = np.log(lam)
    lt, gt = x1 < x2, x1 > x2; eq = ~(lt | gt); lf = np.empty_like(x1)
    lf[eq] = np.log(a3) + b + (a1+a2+a3-1)*lA1[eq] - lam*x1[eq]
    lf[lt] = np.log(a1+a3) + np.log(a2) + 2*b + (a1+a3-1)*lA1[lt] + (a2-1)*lA2[lt] - lam*(x1[lt]+x2[lt])
    lf[gt] = np.log(a2+a3) + np.log(a1) + 2*b + (a1-1)*lA1[gt] + (a2+a3-1)*lA2[gt] - lam*(x1[gt]+x2[gt])
    if m > 1:
        lF = (a1+a3)*lA1; l1F = np.log1p(-np.minimum(np.exp(lF), 1-1e-15))
        lf = lf + (r-1)*lF + (m-r)*l1F + gammaln(m+1) - gammaln(r) - gammaln(m-r+2)
    return lf.sum()

def fit(d, m, start=None):
    lam0 = 1/np.mean(np.concatenate([d[0], d[1]])); x1, x2 = d[0], d[1]
    lo = np.array([np.log(1e-3)]*3 + [np.log(lam0)-6]); hi = np.array([np.log(100)]*3 + [np.log(lam0)+6])
    pl, pg, pt = np.mean(x1 < x2), np.mean(x1 > x2), np.mean(x1 == x2)
    starts = [np.array([1, 1, 1, lam0]), np.array([max(2*pg, .1), max(2*pl, .1), max(2*pt, .1), lam0*1.5])]
    if start is not None: starts = [np.asarray(start, float)]
    def f(u):
        v = loglik(np.exp(u), d, m); return -v if np.isfinite(v) else 1e12
    best = None
    for s0 in starts:
        r_ = minimize(f, np.log(s0), method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r_.fun < best.fun: best = r_
    hit = bool(np.any((best.x < lo+1e-4) | (best.x > hi-1e-4)))
    return np.exp(best.x), -best.fun, hit

def hess(t, d, m):
    k = len(t); H = np.zeros((k, k)); h = 1e-3*np.abs(t); f = lambda v: -loglik(v, d, m); f0 = f(t)
    for i in range(k):
        ei = np.zeros(k); ei[i] = h[i]; H[i, i] = (f(t+ei) + f(t-ei) - 2*f0)/h[i]**2
        for j in range(i):
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(t+ei+ej) - f(t+ei-ej) - f(t-ei+ej) + f(t-ei-ej))/(4*h[i]*h[j])
    return H