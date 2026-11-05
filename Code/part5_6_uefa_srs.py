import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

OUT = "outputs/part5_6"; os.makedirs(OUT, exist_ok=True)
B_BOOT, B_GOF = 500, 300            # reduce for a quick test run (e.g. 100, 100)
rng = np.random.default_rng(2024)

# ---------------- data: UEFA Champions League (Kundu & Gupta 2009, Table 1) ----------------
uefa = [(26,20),(63,18),(19,19),(66,85),(40,40),(49,49),(8,8),(69,71),(39,39),(82,48),
        (72,72),(66,62),(25,9),(41,3),(16,75),(18,18),(22,14),(42,42),(36,52),
        (34,34),(53,39),(54,7),(51,28),(76,64),(64,15),(26,48),(16,16),(44,13),(25,14),
        (55,11),(49,49),(24,24),(44,30),(42,3),(27,47),(28,28),(2,2)]
d = np.array(uefa, float); x1, x2 = d[:, 0], d[:, 1]; n = len(d)
names = ["alpha1", "alpha2", "alpha3", "lambda"]
mpool = np.mean(np.r_[x1, x2])

# ---------------- model ----------------
def A_(x, lam): return -np.expm1(-lam * np.asarray(x, float))

def ll_bvge(th, x1, x2):
    a1, a2, a3, lam = th
    if min(th) <= 0: return -np.inf
    lt, gt, eq = x1 < x2, x1 > x2, x1 == x2; ll = 0.0
    with np.errstate(all="ignore"):
        if lt.any():
            u, v = x1[lt], x2[lt]
            ll += np.sum(np.log(a1+a3) + np.log(a2) + 2*np.log(lam) + (a1+a3-1)*np.log(A_(u, lam))
                         + (a2-1)*np.log(A_(v, lam)) - lam*(u+v))
        if gt.any():
            u, v = x1[gt], x2[gt]
            ll += np.sum(np.log(a2+a3) + np.log(a1) + 2*np.log(lam) + (a1-1)*np.log(A_(u, lam))
                         + (a2+a3-1)*np.log(A_(v, lam)) - lam*(u+v))
        if eq.any():
            w_ = x1[eq]
            ll += np.sum(np.log(a3) + np.log(lam) + (a1+a2+a3-1)*np.log(A_(w_, lam)) - lam*w_)
    return ll

def ll_mo(th, x1, x2):                  # Marshall-Olkin bivariate exponential
    l1, l2, l12 = th
    if min(th) <= 0: return -np.inf
    lt, gt, eq = x1 < x2, x1 > x2, x1 == x2
    ll = np.sum(np.log(l1) + np.log(l2+l12) - l1*x1[lt] - (l2+l12)*x2[lt])
    ll += np.sum(np.log(l2) + np.log(l1+l12) - (l1+l12)*x1[gt] - l2*x2[gt])
    ll += np.sum(np.log(l12) - (l1+l2+l12)*x1[eq])
    return ll

def _bounds(k, m):
    return [(-6, 4)] * 3 + [(np.log(1e-3/m), np.log(1e3/m))] if k == 4 else [(np.log(1e-4/m), np.log(1e2/m))] * 3

def fit_generic(llf, k, x1, x2, nstart=30, seed=0, init=None):
    r_ = np.random.default_rng(seed); m = np.mean(np.r_[x1, x2]); bnd = _bounds(k, m)
    def f(p):
        v = -llf(np.exp(p), x1, x2); return v if np.isfinite(v) else 1e10
    if k == 4:
        starts = [np.r_[np.log(r_.uniform(.3, 3, 3)), np.log(r_.uniform(.3, 3)/m)] for _ in range(nstart)]
    else:
        starts = [np.log(r_.uniform(.3, 3, 3)/m) for _ in range(nstart)]
    if init is not None: starts = [np.log(np.asarray(init, float))] + starts[:max(nstart-1, 0)]
    best = None
    for s0 in starts:
        r = minimize(f, s0, method="L-BFGS-B", bounds=bnd)
        if best is None or r.fun < best.fun: best = r
    r2 = minimize(f, best.x, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=5000))
    if r2.fun < best.fun: best = r2
    return np.exp(best.x), -best.fun

fit_bvge = lambda a, b, **kw: fit_generic(ll_bvge, 4, a, b, **kw)
fit_mo   = lambda a, b, **kw: fit_generic(ll_mo, 3, a, b, **kw)

def fit_em(x1, x2, tol=1e-9, maxit=3000):      # EM of Kundu & Gupta (E-step exact; M-step numerical)
    lt, gt, eq = x1 < x2, x1 > x2, x1 == x2; n0, n1, n2 = eq.sum(), lt.sum(), gt.sum()
    y = x1[eq]; xa1, xa2, xb1, xb2 = x1[lt], x2[lt], x1[gt], x2[gt]
    m = np.mean(np.r_[x1, x2]); th = np.array([1., 1., 1., 1/m]); bnd = _bounds(4, m); hist = []
    ssum = y.sum() + xa1.sum() + xa2.sum() + xb1.sum() + xb2.sum()
    for it in range(maxit):
        a1, a2, a3, lam = th
        u1, u2 = a1/(a1+a3), a3/(a1+a3); w1, w2 = a2/(a2+a3), a3/(a2+a3)
        def Q(p):
            b1, b2, b3, l = np.exp(p); LA = lambda z: np.log(A_(z, l))
            return (n0*np.log(b3) + np.log(l)*(n0 + 2*(n1+n2)) + (b1+b2+b3-1)*LA(y).sum() - l*ssum
                    + np.log(b1)*(u1*n1+n2) + np.log(b2)*(w1*n2+n1) + np.log(b3)*(n1*u2+n2*w2)
                    + (b1+b3-1)*LA(xa1).sum() + (b2+b3-1)*LA(xb2).sum()
                    + (b2-1)*LA(xa2).sum() + (b1-1)*LA(xb1).sum())
        r = minimize(lambda p: -Q(p), np.log(th), method="L-BFGS-B", bounds=bnd)
        th = np.exp(r.x); hist.append(ll_bvge(th, x1, x2))
        if it > 0 and abs(hist[-1] - hist[-2]) < tol: break
    return th, hist[-1], len(hist)

def ge_fit(x):                                 # GE(alpha, lambda) MLE (profile in lambda)
    x = np.asarray(x, float); nn = len(x); m = x.mean()
    def nll(l_):
        lam = np.exp(l_); t = np.log1p(-np.exp(-lam*x)); a = -nn/t.sum()
        return -(nn*np.log(a) + nn*np.log(lam) - lam*x.sum() - nn - t.sum())
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(nll, bounds=(np.log(1e-3/m), np.log(1e2/m)), method="bounded")
    lam = np.exp(r.x); a = -nn/np.log1p(-np.exp(-lam*x)).sum()
    return a, lam, -r.fun

def rge(nn, a, lam, rg): return -np.log1p(-rg.random(nn) ** (1.0/a)) / lam
def sim_bvge(nn, th, rg):
    a1, a2, a3, lam = th
    u1, u2, u3 = rge(nn, a1, lam, rg), rge(nn, a2, lam, rg), rge(nn, a3, lam, rg)
    return np.maximum(np.maximum(u1, u3), .05), np.maximum(np.maximum(u2, u3), .05)
ge_cdf = lambda t, a, lam: A_(t, lam) ** a
ge_q   = lambda p, a, lam: -np.log1p(-p ** (1.0/a)) / lam

def num_hess(f, x, rel=1e-3):
    k = len(x); H = np.zeros((k, k)); h = rel*np.abs(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*h[i]*h[j])
    return H

# =============================== PART 5: estimation ===============================
th_hat, ll_hat = fit_bvge(x1, x2, nstart=40, seed=1)
th_em, ll_em, it_em = fit_em(x1, x2)
th_mo, ll_mo_hat = fit_mo(x1, x2, nstart=30, seed=1)
paper = np.array([1.445, 0.468, 1.170, 0.0390])
n_ac, n_t = int((x1 != x2).sum()), int((x1 == x2).sum())

print(f"n = {n} | ties = {n_t} | abs. continuous pairs = {n_ac}")
est = pd.DataFrame({"direct ML": th_hat, "EM": th_em, "paper (K&G 2009)": paper}, index=names)
print("\n=== BVGE estimates ===\n", est.round(5))
print(f"logL direct = {ll_hat:.4f} | logL EM = {ll_em:.4f} ({it_em} EM iterations)")
print(f"logL at the paper's estimates (checks our data entry) = {ll_bvge(paper, x1, x2):.4f}")
print("\nlogL under data scaling x/c (paper reports -20.59) :")
for c in (1, 10, 100):
    sh = (2*n_ac + n_t) * np.log(c)
    print(f"  c={c:>3}: ours = {ll_hat + sh:9.4f} | at paper estimates = {ll_bvge(paper, x1, x2) + sh:9.4f}")
print("\n=== Marshall-Olkin estimates (lambda1, lambda2, lambda12) ===", np.round(th_mo, 5))

k_b, k_m = 4, 3
crit = pd.DataFrame({"model": ["BVGE", "Marshall-Olkin"], "k": [k_b, k_m], "logL": [ll_hat, ll_mo_hat]})
crit["AIC"] = 2*crit.k - 2*crit.logL; crit["BIC"] = crit.k*np.log(n) - 2*crit.logL
crit["HQIC"] = 2*crit.k*np.log(np.log(n)) - 2*crit.logL
print("\n=== Information criteria ===\n", crit.round(4).to_string(index=False))
est.to_csv(f"{OUT}/table_estimates.csv"); crit.to_csv(f"{OUT}/table_criteria.csv", index=False)

# ---- goodness of fit (KS), asymptotic and parametric-bootstrap p-values
def gof_stats(th, a, b):
    a1, a2, a3, lam = th; s = a1 + a2 + a3
    return (stats.kstest(a, lambda t: ge_cdf(t, a1+a3, lam)).statistic,
            stats.kstest(b, lambda t: ge_cdf(t, a2+a3, lam)).statistic,
            stats.kstest(np.maximum(a, b), lambda t: ge_cdf(t, s, lam)).statistic)
D_obs = np.array(gof_stats(th_hat, x1, x2))
Db = []
for _ in range(B_GOF):
    s1, s2 = sim_bvge(n, th_hat, rng)
    thb, _ = fit_bvge(s1, s2, nstart=3, seed=int(rng.integers(1e9)), init=th_hat)
    Db.append(gof_stats(thb, s1, s2))
Db = np.array(Db); p_boot = (Db >= D_obs).mean(axis=0)
a1h, a2h, a3h, lamh = th_hat; sh_ = a1h + a2h + a3h
asy = [stats.kstest(x1, lambda t: ge_cdf(t, a1h+a3h, lamh)).pvalue,
       stats.kstest(x2, lambda t: ge_cdf(t, a2h+a3h, lamh)).pvalue,
       stats.kstest(np.maximum(x1, x2), lambda t: ge_cdf(t, sh_, lamh)).pvalue]
gof = pd.DataFrame({"test": ["X1 ~ GE(a1+a3, lam)", "X2 ~ GE(a2+a3, lam)", "max(X1,X2) ~ GE(a1+a2+a3, lam)"],
                    "KS_D": D_obs, "p_asymptotic": asy, "p_param_bootstrap": p_boot})
print("\n=== Goodness of fit (joint-model marginals) ===\n", gof.round(4).to_string(index=False))
g1, g2 = ge_fit(x1), ge_fit(x2); sep = []
for nm, (a, l, ll_), v in (("X1", g1, x1), ("X2", g2, x2)):
    ks = stats.kstest(v, lambda t: ge_cdf(t, a, l)); sep.append((nm, a, l, ll_, ks.statistic, ks.pvalue))
sep = pd.DataFrame(sep, columns=["var", "alpha", "lambda", "logL", "KS_D", "p"])
print("\n=== Separate GE fits (paper: X1 (3.121,0.0449), X2 (1.678,0.0413)) ===\n", sep.round(4).to_string(index=False))
gof.to_csv(f"{OUT}/table_gof.csv", index=False); sep.to_csv(f"{OUT}/table_separate_GE.csv", index=False)

# =============================== PART 6: information, CIs, graphs ===============================
I_obs = -num_hess(lambda t: ll_bvge(t, x1, x2), th_hat)
ev = np.linalg.eigvalsh(I_obs); print("\neigenvalues of observed information:", np.round(ev, 4), "(all > 0 needed)")
cov = np.linalg.inv(I_obs); se = np.sqrt(np.diag(cov)); corr = cov / np.outer(se, se)
pd.DataFrame(I_obs, index=names, columns=names).to_csv(f"{OUT}/table_observed_information.csv")
pd.DataFrame(cov, index=names, columns=names).to_csv(f"{OUT}/table_covariance.csv")
print("\n=== Observed information ===\n", pd.DataFrame(I_obs, index=names, columns=names).round(3))
print("\n=== Correlation of estimators ===\n", pd.DataFrame(corr, index=names, columns=names).round(3))

z = 1.96
ci_w = np.c_[th_hat - z*se, th_hat + z*se]
ci_l = np.c_[th_hat*np.exp(-z*se/th_hat), th_hat*np.exp(z*se/th_hat)]

# profile likelihood
def profile(k, grid):
    idx = [i for i in range(4) if i != k]; out = []
    for v in grid:
        def f(p):
            t = th_hat.copy(); t[k] = v; t[idx] = np.exp(p)
            val = -ll_bvge(t, x1, x2); return val if np.isfinite(val) else 1e10
        best = 1e10
        for j in range(3):
            p0 = np.log(th_hat[idx]) + (0 if j == 0 else rng.normal(0, .3, 3))
            r = minimize(f, p0, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=3000))
            best = min(best, r.fun)
        out.append(-best)
    return np.array(out)
def prof_ci(grid, dev, e, c=3.841):
    lo = hi = np.nan
    L = np.where((grid < e) & (dev > c))[0]
    if len(L): i = L.max(); lo = np.interp(c, [dev[i+1], dev[i]], [grid[i+1], grid[i]])
    R = np.where((grid > e) & (dev > c))[0]
    if len(R): j = R.min(); hi = np.interp(c, [dev[j-1], dev[j]], [grid[j-1], grid[j]])
    return lo, hi
grids, devs, ci_p = [], [], []
for k in range(4):
    g_ = th_hat[k] * np.exp(np.linspace(-2.5, 2.5, 41)); lp = profile(k, g_); dv = 2*(ll_hat - lp)
    grids.append(g_); devs.append(np.maximum(dv, 0)); ci_p.append(prof_ci(g_, devs[-1], th_hat[k]))
ci_p = np.array(ci_p)

# nonparametric bootstrap of pairs
bt = []
for _ in range(B_BOOT):
    ii = rng.integers(0, n, n); thb, _ = fit_bvge(x1[ii], x2[ii], nstart=3, seed=int(rng.integers(1e9)), init=th_hat)
    bt.append(thb)
bt = np.array(bt); ci_b = np.c_[np.percentile(bt, 2.5, axis=0), np.percentile(bt, 97.5, axis=0)]
paper_ci = np.array([[0.657, 2.233], [0.167, 0.769], [0.651, 1.689], [0.028, 0.050]])
ci_tab = pd.DataFrame({"MLE": th_hat, "SE": se, "boot_SE": bt.std(axis=0, ddof=1),
                       "Wald_lo": ci_w[:, 0], "Wald_hi": ci_w[:, 1], "LogWald_lo": ci_l[:, 0], "LogWald_hi": ci_l[:, 1],
                       "Profile_lo": ci_p[:, 0], "Profile_hi": ci_p[:, 1], "Boot_lo": ci_b[:, 0], "Boot_hi": ci_b[:, 1],
                       "Paper_lo": paper_ci[:, 0], "Paper_hi": paper_ci[:, 1]}, index=names)
print("\n=== 95% confidence intervals ===\n", ci_tab.round(4).to_string())
ci_tab.to_csv(f"{OUT}/table_confidence_intervals.csv")
pd.DataFrame(bt, columns=names).to_csv(f"{OUT}/bootstrap_estimates.csv", index=False)

# ---------------- figures ----------------
big = sim_bvge(3000, th_hat, rng)
tie = x1 == x2; lim = [0, max(x1.max(), x2.max()) * 1.05]
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
ax[0, 0].scatter(big[0], big[1], s=4, c="lightgrey", label="simulated from fitted BVGE")
ax[0, 0].scatter(x1[~tie], x2[~tie], s=22, label="observed, X1 != X2"); ax[0, 0].scatter(x1[tie], x2[tie], s=28, c="red", label="observed ties")
ax[0, 0].plot(lim, lim, "k--", lw=.7); ax[0, 0].set_xlim(lim); ax[0, 0].set_ylim(lim); ax[0, 0].legend(fontsize=7)
ax[0, 0].set_xlabel("X1 (first kick goal)"); ax[0, 0].set_ylabel("X2 (first home goal)"); ax[0, 0].set_title("UEFA data and fitted model")
tt = np.linspace(0.01, 100, 400)
for a_, v_, lab, k_ in ((ax[0, 1], x1, "X1", a1h+a3h), (ax[1, 0], x2, "X2", a2h+a3h), (ax[1, 1], np.maximum(x1, x2), "max(X1,X2)", sh_)):
    a_.step(np.sort(v_), np.arange(1, n+1)/n, where="post", label="empirical"); a_.plot(tt, ge_cdf(tt, k_, lamh), label=f"GE({k_:.2f}, {lamh:.4f})")
    a_.set_title(f"CDF of {lab}"); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_fit_and_cdfs.png", dpi=300)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2)); pp = (np.arange(1, n+1) - .5) / n
for a_, v_, lab, k_ in zip(ax, (x1, x2, np.maximum(x1, x2)), ("X1", "X2", "max(X1,X2)"), (a1h+a3h, a2h+a3h, sh_)):
    a_.scatter(ge_q(pp, k_, lamh), np.sort(v_), s=14); m_ = max(np.sort(v_).max(), ge_q(pp, k_, lamh).max())
    a_.plot([0, m_], [0, m_], "k--", lw=.7); a_.set_xlabel("fitted quantile"); a_.set_ylabel("empirical quantile"); a_.set_title(f"QQ plot: {lab}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_qq.png", dpi=300)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for k, a_ in enumerate(ax.ravel()):
    a_.plot(grids[k], devs[k]); a_.axhline(3.841, c="r", ls=":"); a_.axvline(th_hat[k], c="k", ls="--", lw=.7)
    a_.set_ylim(0, 12); a_.set_xscale("log"); a_.set_xlabel(names[k]); a_.set_ylabel("profile deviance"); a_.set_title(f"Profile deviance: {names[k]}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_profile_deviance.png", dpi=300)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for k, a_ in enumerate(ax.ravel()):
    a_.hist(bt[:, k], bins=30, color="lightgrey", edgecolor="k"); a_.axvline(th_hat[k], c="r"); a_.axvline(ci_b[k, 0], c="b", ls="--"); a_.axvline(ci_b[k, 1], c="b", ls="--")
    a_.set_title(f"Bootstrap distribution: {names[k]}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bootstrap_hist.png", dpi=300)
pd.plotting.scatter_matrix(pd.DataFrame(bt, columns=names), figsize=(9, 9), alpha=.4, diagonal="hist")
plt.savefig(f"{OUT}/fig_bootstrap_scatter_matrix.png", dpi=300)
print("\nsaved tables and figures to", OUT)