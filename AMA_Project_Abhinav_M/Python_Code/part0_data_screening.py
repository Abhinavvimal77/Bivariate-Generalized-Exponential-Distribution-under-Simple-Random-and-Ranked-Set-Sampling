import os, re, itertools, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

DATA_FILE   = r"C:\abhi\AMA_ACADEMIC_PROJECT\Code\main.csv"   # <-- CHANGE: your cleaned C-MAPSS file (xlsx or csv)
CHOSEN      = None               # after screening, e.g. ("s_11","s_4",5,2.0) = (sensor X1, sensor X2, window, threshold)
BASE_CYCLES = 20
WINDOWS     = [5, 10]
THRESHOLDS  = [1.5, 2.0, 2.5, 3.0]
OUT = "outputs/part0"; os.makedirs(OUT, exist_ok=True)

# ---- 0. UEFA data (Kundu & Gupta 2009, Table 1) saved for validation later
uefa = [(26,20),(63,18),(19,19),(66,85),(40,40),(49,49),(8,8),(69,71),(39,39),(82,48),
        (72,72),(66,62),(25,9),(41,3),(16,75),(18,18),(22,14),(42,42),(36,52),
        (34,34),(53,39),(54,7),(51,28),(76,64),(64,15),(26,48),(16,16),(44,13),(25,14),
        (55,11),(49,49),(24,24),(44,30),(42,3),(27,47),(28,28),(2,2)]
pd.DataFrame(uefa, columns=["X1","X2"]).to_csv(f"{OUT}/uefa_data.csv", index=False)
print("UEFA: n =", len(uefa), "| ties =", sum(a == b for a, b in uefa))

# ---- 1. load C-MAPSS
df = pd.read_excel(DATA_FILE) if DATA_FILE.lower().endswith(("xlsx", "xls")) else pd.read_csv(DATA_FILE)
df.columns = [str(c).strip() for c in df.columns]
assert {"unit_nr", "time_cycles"} <= set(df.columns), f"columns found: {list(df.columns)}"
df = df.dropna().sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)
sensors = [c for c in df.columns if re.fullmatch(r"s_?\d+", c, flags=re.I) and df[c].nunique() > 10]
units = df["unit_nr"].unique()
g = {u: d for u, d in df.groupby("unit_nr")}
print("engines:", len(units), "| usable sensors:", sensors)

# ---- 2. baseline sd and degradation direction per sensor
sd_base, dirn = {}, {}
for s in sensors:
    v_, d_ = [], []
    for u in units:
        v = g[u][s].to_numpy()
        v_.append(v[:BASE_CYCLES].var(ddof=1))
        d_.append(v[-BASE_CYCLES:].mean() - v[:BASE_CYCLES].mean())
    sd_base[s] = np.sqrt(np.mean(v_)); dirn[s] = np.sign(np.mean(d_))
sensors = [s for s in sensors if dirn[s] != 0 and sd_base[s] > 0]

# ---- 3. alarm cycles: first cycle (after baseline) where rolled, direction-adjusted z >= threshold
alarms = {}
for s in sensors:
    for w in WINDOWS:
        zs = []
        for u in units:
            d = g[u]; v = d[s].to_numpy(); t = d["time_cycles"].to_numpy()
            z = dirn[s] * (v - v[:BASE_CYCLES].mean()) / sd_base[s]
            zr = pd.Series(z).rolling(w, min_periods=w).mean().to_numpy()
            zs.append((zr, t))
        for c in THRESHOLDS:
            arr = np.full(len(units), np.nan)
            for k, (zr, t) in enumerate(zs):
                idx = np.where(np.nan_to_num(zr[BASE_CYCLES:], nan=-np.inf) >= c)[0]
                if len(idx): arr[k] = t[BASE_CYCLES + idx[0]]
            alarms[(s, w, c)] = arr

def ge_fit(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    def nll(ll):
        lam = np.exp(ll); t = np.log1p(-np.exp(-lam * x)); a = -n / t.sum()
        return -(n * np.log(a) + n * np.log(lam) - lam * x.sum() - n - t.sum())
    r = minimize_scalar(nll, bounds=(np.log(1e-3 / m), np.log(1e2 / m)), method="bounded")
    lam = np.exp(r.x); a = -n / np.log1p(-np.exp(-lam * x)).sum()
    p = stats.kstest(x, lambda q: (1 - np.exp(-lam * q)) ** a).pvalue
    return a, lam, p

# ---- 4. screen all pairs
rows = []
for (w, c) in itertools.product(WINDOWS, THRESHOLDS):
    for s1, s2 in itertools.combinations(sensors, 2):
        a1_, a2_ = alarms[(s1, w, c)], alarms[(s2, w, c)]
        m = ~np.isnan(a1_) & ~np.isnan(a2_)
        if m.sum() < 30: continue
        x1, x2 = a1_[m], a2_[m]
        rows.append(dict(window=w, thr=c, X1=s1, X2=s2, n=int(m.sum()),
                         cross_rate=m.mean(), tie_pct=100 * np.mean(x1 == x2),
                         pearson=stats.pearsonr(x1, x2)[0], spearman=stats.spearmanr(x1, x2)[0],
                         ks_p1=ge_fit(x1)[2], ks_p2=ge_fit(x2)[2],
                         skew1=stats.skew(x1), skew2=stats.skew(x2)))
scr = pd.DataFrame(rows)
scr.to_csv(f"{OUT}/screening_all_pairs.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
ok = scr[(scr.cross_rate >= 0.9) & (scr.n >= 80) & (scr.tie_pct.between(5, 60)) &
         (scr.ks_p1 >= 0.05) & (scr.ks_p2 >= 0.05)].sort_values("spearman", ascending=False)
print("\n=== pairs passing all filters (top 20 by Spearman) ===")
print(ok.head(20).round(3).to_string(index=False) if len(ok) else "NONE passed")
print("\n=== top 15 by tie percentage (ignoring filters) ===")
print(scr[scr.n >= 80].sort_values("tie_pct", ascending=False).head(15).round(3).to_string(index=False))

# ---- 5. freeze chosen pair
if CHOSEN:
    s1, s2, w, c = CHOSEN
    a1_, a2_ = alarms[(s1, w, c)], alarms[(s2, w, c)]
    m = ~np.isnan(a1_) & ~np.isnan(a2_)
    fin = pd.DataFrame({"unit_nr": units[m], "X1": a1_[m], "X2": a2_[m]})
    fin.to_csv(f"{OUT}/data_final.csv", index=False)
    def summ(v):
        return dict(Min=v.min(), Q1=np.percentile(v, 25), Median=np.median(v), Mean=v.mean(),
                    Q3=np.percentile(v, 75), Max=v.max(), Skewness=stats.skew(v),
                    Kurtosis=stats.kurtosis(v, fisher=False), StDev=v.std(ddof=1), Range=np.ptp(v))
    tab = pd.DataFrame({"X1": summ(fin.X1.values), "X2": summ(fin.X2.values)}).round(3)
    tab.to_csv(f"{OUT}/summary_table.csv"); print("\n", tab)
    tie = fin.X1 == fin.X2
    print(f"\nn={len(fin)} | ties={tie.sum()} ({100*tie.mean():.1f}%) | Pearson={stats.pearsonr(fin.X1,fin.X2)[0]:.3f}"
          f" | Spearman={stats.spearmanr(fin.X1,fin.X2)[0]:.3f} | Kendall={stats.kendalltau(fin.X1,fin.X2)[0]:.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].scatter(fin.X1[~tie], fin.X2[~tie], s=14, alpha=.7, label="X1 != X2")
    ax[0].scatter(fin.X1[tie], fin.X2[tie], s=22, c="red", label="ties")
    lim = [0, max(fin.X1.max(), fin.X2.max()) * 1.05]; ax[0].plot(lim, lim, "k--", lw=.8)
    ax[0].set_xlabel(f"X1 ({s1})"); ax[0].set_ylabel(f"X2 ({s2})"); ax[0].legend()
    ax[1].hist(fin.X1, bins=20); ax[1].set_title("X1")
    ax[2].hist(fin.X2, bins=20); ax[2].set_title("X2")
    fig.savefig(f"{OUT}/final_data_plots.png", dpi=300, bbox_inches="tight")
# ===================== PART 0b: BVGE vs Marshall-Olkin on shortlisted pairs, with inspection blocks =====================
from scipy.optimize import minimize
BLOCKS = [1, 5, 10]          # inspection interval in cycles (1 = no blocking)

def A_(x, lam): return -np.expm1(-lam * x)

def ll_bvge(p, x1, x2):
    a1, a2, a3, lam = np.exp(p); lt, gt, eq = x1 < x2, x1 > x2, x1 == x2; ll = 0.0
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

def ll_mo(p, x1, x2):
    l1, l2, l12 = np.exp(p); lt, gt, eq = x1 < x2, x1 > x2, x1 == x2
    ll = np.sum(np.log(l1) + np.log(l2+l12) - l1*x1[lt] - (l2+l12)*x2[lt])
    ll += np.sum(np.log(l2) + np.log(l1+l12) - (l1+l12)*x1[gt] - l2*x2[gt])
    ll += np.sum(np.log(l12) - (l1+l2+l12)*x1[eq])
    return ll

def fit_model(llf, npar, x1, x2, nstart=25, seed=1):
    rng = np.random.default_rng(seed); best = None
    def f(p):
        v = -llf(p, x1, x2); return v if np.isfinite(v) else 1e10
    for _ in range(nstart):
        r = minimize(f, np.log(rng.uniform(.3, 3, npar)), method="L-BFGS-B", bounds=[(-6, 5)] * npar)
        if best is None or r.fun < best.fun: best = r
    return np.exp(best.x), -best.fun

def rge(n, a, lam, rng): return -np.log1p(-rng.random(n) ** (1.0 / a)) / lam

rng0 = np.random.default_rng(7)
short = ok.head(25)[["window", "thr", "X1", "X2"]].values.tolist()
res = []
for (w, c, s1, s2) in short:
    a1_, a2_ = alarms[(s1, w, c)], alarms[(s2, w, c)]
    m = ~np.isnan(a1_) & ~np.isnan(a2_)
    for k in BLOCKS:
        x1, x2 = np.ceil(a1_[m] / k), np.ceil(a2_[m] / k)
        sc = np.mean(np.r_[x1, x2]); x1s, x2s = x1 / sc, x2 / sc       # rescale to mean ~1 (comparisons stay valid)
        n = len(x1s)
        pb, llb = fit_model(ll_bvge, 4, x1s, x2s); pm, llm = fit_model(ll_mo, 3, x1s, x2s)
        a1, a2, a3, lam = pb; s = a1 + a2 + a3
        u1, u2, u3 = rge(100000, a1, lam, rng0), rge(100000, a2, lam, rng0), rge(100000, a3, lam, rng0)
        y1, y2 = np.maximum(u1, u3), np.maximum(u2, u3)
        ksmax = stats.kstest(np.maximum(x1s, x2s), lambda t: A_(t, lam) ** s).pvalue
        res.append(dict(window=w, thr=c, X1=s1, X2=s2, block=k, n=n,
                        tie_obs=100*np.mean(x1 == x2), tie_BVGE=100*a3/s,
                        spear_obs=stats.spearmanr(x1, x2)[0], spear_BVGE=stats.spearmanr(y1, y2)[0],
                        pear_obs=stats.pearsonr(x1, x2)[0], pear_BVGE=stats.pearsonr(y1, y2)[0],
                        logL_BVGE=llb, logL_MO=llm,
                        AIC_BVGE=8-2*llb, AIC_MO=6-2*llm, BIC_BVGE=4*np.log(n)-2*llb, BIC_MO=3*np.log(n)-2*llm,
                        ks_max_p=ksmax, a1=a1, a2=a2, a3=a3, lam=lam))
r0b = pd.DataFrame(res); r0b["dAIC"] = r0b.AIC_BVGE - r0b.AIC_MO     # negative = BVGE better
r0b.to_csv(f"{OUT}/part0b_bvge_vs_mo.csv", index=False)
cols = ["window", "thr", "X1", "X2", "block", "n", "tie_obs", "tie_BVGE", "spear_obs", "spear_BVGE",
        "logL_BVGE", "logL_MO", "dAIC", "ks_max_p"]
print("\n=== PART 0b: sorted by dAIC (BVGE - MO), negative = BVGE better ===")
print(r0b[r0b.n >= 80].sort_values("dAIC")[cols].head(30).round(3).to_string(index=False))
print("\n=== PART 0b: tie_obs by block size (mean over shortlisted pairs) ===")
print(r0b.groupby("block")[["tie_obs", "tie_BVGE", "spear_obs", "spear_BVGE"]].mean().round(3))