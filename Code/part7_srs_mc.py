import os, sys, time, numpy as np, pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

R, NS, SEED = 1000, (30, 50, 70, 100), 2026
OUT = "outputs/part7"
SETS = {"UEFA": (1.4452, 0.4681, 1.1704, 0.03899), "S1": (1.5, 1.0, 0.5, 1.0),
        "S2": (1.0, 1.0, 2.0, 1.0), "S3": (2.0, 0.7, 0.3, 0.5)}
PN = ["alpha1", "alpha2", "alpha3", "lambda"]; Z = 1.959964

def sim(n, th, rng):
    a = np.array(th[:3]); V = rng.random((3, n)); U = -np.log1p(-V**(1/a[:, None]))/th[3]
    return np.maximum(U[0], U[2]), np.maximum(U[1], U[2])

def mk(x1, x2):
    lt, gt, eq = x1 < x2, x1 > x2, x1 == x2
    return x1[lt], x2[lt], x1[gt], x2[gt], x1[eq]

def loglik(t, d):
    a1, a2, a3, lam = t; x1l, x2l, x1g, x2g, xt = d; s = a1+a2+a3
    lA = lambda x: np.log(-np.expm1(-lam*x))
    ll = len(xt)*(np.log(a3)+np.log(lam)) + (s-1)*lA(xt).sum() - lam*xt.sum()
    ll += len(x1l)*(np.log(a1+a3)+np.log(a2)+2*np.log(lam)) + (a1+a3-1)*lA(x1l).sum() + (a2-1)*lA(x2l).sum() - lam*(x1l.sum()+x2l.sum())
    ll += len(x1g)*(np.log(a2+a3)+np.log(a1)+2*np.log(lam)) + (a1-1)*lA(x1g).sum() + (a2+a3-1)*lA(x2g).sum() - lam*(x1g.sum()+x2g.sum())
    return ll

def fit(d, lam0):
    n = sum(len(v) for v in (d[0], d[2], d[4])); pl, pg, pt = len(d[0])/n, len(d[2])/n, len(d[4])/n
    lo = np.array([np.log(1e-3)]*3 + [np.log(lam0)-6]); hi = np.array([np.log(100)]*3 + [np.log(lam0)+6])
    starts = [np.array([1, 1, 1, lam0]), np.array([max(2*pg, .1), max(2*pl, .1), max(2*pt, .1), lam0*1.5])]
    best = None
    def f(u):
        v = loglik(np.exp(u), d); return -v if np.isfinite(v) else 1e12
    for s0 in starts:
        r = minimize(f, np.log(s0), method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun: best = r
    hit = bool(np.any((best.x < lo+1e-4) | (best.x > hi-1e-4)))
    return np.exp(best.x), -best.fun, hit

def hess(t, d):
    k = len(t); H = np.zeros((k, k)); h = 1e-3*np.abs(t); f = lambda v: -loglik(v, d); f0 = f(t)
    for i in range(k):
        ei = np.zeros(k); ei[i] = h[i]
        H[i, i] = (f(t+ei) + f(t-ei) - 2*f0)/h[i]**2
        for j in range(i):
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(t+ei+ej) - f(t+ei-ej) - f(t-ei+ej) + f(t-ei-ej))/(4*h[i]*h[j])
    return H

def ci_block(t, d):
    H = hess(t, d)
    try:
        if not np.all(np.linalg.eigvalsh(H) > 0): raise np.linalg.LinAlgError
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        return None
    return se

def chunk(args):
    nm, n, seed, reps = args; th = SETS[nm]; rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        x1, x2 = sim(n, th, rng); d = mk(x1, x2); lam0 = 1/np.mean(np.concatenate([x1, x2]))
        t, ll, hit = fit(d, lam0); row = np.full(28, np.nan); row[:4] = t; row[24] = hit; row[26] = len(d[4]); row[27] = ll
        se = None if hit else ci_block(t, d)
        if se is not None:
            row[4:8] = se; row[8:12] = t - Z*se; row[12:16] = t + Z*se
            row[16:20] = t*np.exp(-Z*se/t); row[20:24] = t*np.exp(Z*se/t); row[25] = 1
        else: row[25] = 0
        out.append(row)
    return np.array(out)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1: R = int(sys.argv[1])
    nw = max(1, (os.cpu_count() or 2) - 1); ss = np.random.SeedSequence(SEED); raw = []; t0 = time.time()
    with ProcessPoolExecutor(nw) as ex:
        for nm in SETS:
            for n in NS:
                parts = [R//nw + (1 if i < R % nw else 0) for i in range(nw)]; seeds = ss.spawn(nw)
                res = list(ex.map(chunk, [(nm, n, s, p) for s, p in zip(seeds, parts) if p > 0]))
                A = np.vstack(res); df = pd.DataFrame(A, columns=[f"est_{p}" for p in PN] + [f"se_{p}" for p in PN] +
                     [f"wlo_{p}" for p in PN] + [f"whi_{p}" for p in PN] + [f"llo_{p}" for p in PN] + [f"lhi_{p}" for p in PN] +
                     ["bound_hit", "ci_ok", "n_ties", "logL"]); df.insert(0, "n", n); df.insert(0, "set", nm); raw.append(df)
                print(f"done {nm} n={n}  ({time.time()-t0:.0f}s)", flush=True)
    raw = pd.concat(raw, ignore_index=True); raw.to_csv(f"{OUT}/mc_raw.csv", index=False)

    rows = []
    for (nm, n), g in raw.groupby(["set", "n"], sort=False):
        th = SETS[nm]; ok = g[(g.bound_hit == 0)]; cik = g[g.ci_ok == 1]
        for i, p in enumerate(PN):
            e = ok[f"est_{p}"]; tv = th[i]
            rows.append(dict(set=nm, n=n, param=p, true=tv, mean=e.mean(), bias=e.mean()-tv, rel_bias_pct=100*(e.mean()-tv)/tv,
                MSE=np.mean((e-tv)**2), RMSE=np.sqrt(np.mean((e-tv)**2)),
                cov_Wald=np.mean((cik[f"wlo_{p}"] <= tv) & (tv <= cik[f"whi_{p}"])), AL_Wald=np.mean(cik[f"whi_{p}"]-cik[f"wlo_{p}"]),
                cov_logWald=np.mean((cik[f"llo_{p}"] <= tv) & (tv <= cik[f"lhi_{p}"])), AL_logWald=np.mean(cik[f"lhi_{p}"]-cik[f"llo_{p}"]),
                bound_hit_pct=100*g.bound_hit.mean(), ci_fail_pct=100*(1-g.ci_ok.mean()), mean_ties=g.n_ties.mean()))
    S = pd.DataFrame(rows); S.to_csv(f"{OUT}/table_mc_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for nm in SETS:
        print(f"\n=== SRS Monte Carlo, set {nm}, true = {SETS[nm]} ===")
        print(S[S.set == nm].drop(columns="set").to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for i, p in enumerate(PN):
        for nm in SETS:
            d = S[(S.set == nm) & (S.param == p)]
            ax[0, i].plot(d.n, d.MSE/d.true**2, "o-", label=nm); ax[1, i].plot(d.n, d.cov_logWald, "o-", label=nm)
        ax[0, i].set_yscale("log"); ax[0, i].set_title(f"{p}: relative MSE (MSE/true^2)"); ax[1, i].axhline(.95, c="k", ls="--")
        ax[1, i].set_title(f"{p}: coverage (log-Wald 95%)"); ax[1, i].set_xlabel("n"); ax[0, i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_mse_coverage.png", dpi=300)
    fig, ax = plt.subplots(len(SETS), 4, figsize=(16, 3.2*len(SETS)))
    for r_, nm in enumerate(SETS):
        g = raw[(raw.set == nm) & (raw.n == NS[-1]) & (raw.bound_hit == 0)]
        for i, p in enumerate(PN):
            ax[r_, i].hist(g[f"est_{p}"], bins=30, density=True, color="C0", alpha=.7); ax[r_, i].axvline(SETS[nm][i], c="r")
            ax[r_, i].set_title(f"{nm}: {p} (n={NS[-1]})", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_hist_n100.png", dpi=300)
    print("\nsaved to", OUT)