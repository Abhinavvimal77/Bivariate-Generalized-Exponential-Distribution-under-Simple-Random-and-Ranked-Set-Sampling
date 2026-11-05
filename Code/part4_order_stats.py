import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma, comb
from scipy.stats import beta as Beta
from scipy.integrate import quad
import bvge_mom as M

OUT = "outputs/part4"; os.makedirs(OUT, exist_ok=True)
SETS = {"S1": (1.5, 1.0, 0.5, 1.0), "S2": (1.0, 1.0, 2.0, 1.0), "S3": (2.0, 0.7, 0.3, 0.5)}
f_est = "outputs/part5_6/table_estimates.csv"
if os.path.exists(f_est):
    e = pd.read_csv(f_est, index_col=0)["direct ML"].values; SETS["UEFA"] = tuple(e[:4])
NS = (5, 10); rng = np.random.default_rng(11); pd.set_option("display.width", 220)

def gm(k, lam): return (digamma(k+1)-digamma(1))/lam, (polygamma(1, 1)-polygamma(1, k+1))/lam**2
def sim(n, a1, a2, a3, lam, size):
    V = rng.random((3, size, n)); U = -np.log1p(-V**(1/np.array([a1, a2, a3])[:, None, None]))/lam
    return np.maximum(U[0], U[2]), np.maximum(U[1], U[2])
def integ(h, r, n):  # integral of h(t) * Beta(r, n-r+1) density over (0,1)
    return quad(lambda t: h(t)*Beta.pdf(t, r, n-r+1), 0, 1, epsabs=1e-9, epsrel=1e-9, limit=200)[0]

allrows = []; chk = []
for nm, (a1, a2, a3, lam) in SETS.items():
    k = a1 + a3; w = lambda t: np.minimum(t**(1/k), 1-1e-16); x = lambda t: M.L(w(t))/lam
    cond = lambda t, r_: M.cm(r_, w(t), a2, a1, a3, lam)            # E[X2^r | X1 = x(t)]
    EX1, _ = gm(k, lam); EX2, VX2 = gm(a2+a3, lam); EX2sq = VX2+EX2**2
    E12 = M.EG(lambda u, v: u*v, a1, a2, a3, lam)
    for n in NS:
        rows = []
        for r in range(1, n+1):
            EX = integ(x, r, n); EXs = integ(lambda t: x(t)**2, r, n)
            EY = integ(lambda t: cond(t, 1), r, n); EYs = integ(lambda t: cond(t, 2), r, n)
            EXY = integ(lambda t: x(t)*cond(t, 1), r, n)
            VX, VY = EXs-EX**2, EYs-EY**2; C = EXY-EX*EY
            rows.append(dict(set=nm, n=n, r=r, E_X_rn=EX, Var_X_rn=VX, E_Y_rn=EY, Var_Y_rn=VY, Cov=C, Corr=C/np.sqrt(VX*VY), E_XY=EXY, EY2=EYs))
        d = pd.DataFrame(rows)
        # MC check
        x1, x2 = sim(n, a1, a2, a3, lam, 200_000); o = np.argsort(x1, axis=1)
        xs, ys = np.take_along_axis(x1, o, 1), np.take_along_axis(x2, o, 1)
        d["MC_E_X"], d["MC_E_Y"], d["MC_Var_Y"] = xs.mean(0), ys.mean(0), ys.var(0)
        d["MC_Corr"] = [np.corrcoef(xs[:, i], ys[:, i])[0, 1] for i in range(n)]
        allrows.append(d)
        chk.append(dict(set=nm, n=n,
            sum_EX_vs_nEX1=abs(d.E_X_rn.sum()/(n*EX1)-1), sum_EY_vs_nEX2=abs(d.E_Y_rn.sum()/(n*EX2)-1),
            sum_EY2_vs_nEX2sq=abs(d.EY2.sum()/(n*EX2sq)-1), sum_EXY_vs_nE12=abs(d.E_XY.sum()/(n*E12)-1),
            max_rel_MC_E_X=np.max(np.abs(d.MC_E_X/d.E_X_rn-1)), max_rel_MC_E_Y=np.max(np.abs(d.MC_E_Y/d.E_Y_rn-1)),
            max_abs_MC_Corr=np.max(np.abs(d.MC_Corr-d.Corr))))
        print(f"\n=== {nm}, n={n}: order statistic X_(r:n) of X1 and concomitant Y_[r:n] of X2 ===")
        print(d[["r", "E_X_rn", "Var_X_rn", "E_Y_rn", "Var_Y_rn", "Cov", "Corr", "MC_E_Y", "MC_Corr"]].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
T = pd.concat(allrows); T.to_csv(f"{OUT}/table_order_stats_concomitants.csv", index=False)
C = pd.DataFrame(chk); C.to_csv(f"{OUT}/table_checks.csv", index=False)
print("\n=== identity and Monte Carlo checks (relative errors; sums should be ~1e-8) ===\n", C.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# componentwise extremes of the pair: min(X1,X2), max(X1,X2)
rows = []
for nm, (a1, a2, a3, lam) in SETS.items():
    s = a1+a2+a3; Emax = (digamma(s+1)-digamma(1))/lam
    Emin = quad(lambda u: (1-u**(a1+a3)-u**(a2+a3)+u**s)/(lam*(1-u)), 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
    x1, x2 = sim(1, a1, a2, a3, lam, 1_000_000); x1, x2 = x1[:, 0], x2[:, 0]
    rows.append(dict(set=nm, E_max=Emax, MC_max=np.maximum(x1, x2).mean(), E_min=Emin, MC_min=np.minimum(x1, x2).mean(),
                     sum_check=Emax+Emin-(gm(a1+a3, lam)[0]+gm(a2+a3, lam)[0])))
X = pd.DataFrame(rows); X.to_csv(f"{OUT}/table_min_max.csv", index=False)
print("\n=== E[max(X1,X2)], E[min(X1,X2)] (sum_check should be ~0) ===\n", X.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

# figures (n=10)
nS = len(SETS); fig, ax = plt.subplots(2, nS, figsize=(4.2*nS, 7), squeeze=False)
for j, nm in enumerate(SETS):
    d = T[(T.set == nm) & (T.n == 10)]
    ax[0, j].plot(d.r, d.E_X_rn, "o-", label="E X_(r:10)"); ax[0, j].plot(d.r, d.E_Y_rn, "s-", label="E Y_[r:10]")
    ax[0, j].plot(d.r, d.MC_E_Y, "k.", label="MC (Y)"); ax[0, j].set_title(f"{nm}: means by rank"); ax[0, j].legend(fontsize=7)
    ax[1, j].plot(d.r, d.Corr, "o-", label="theory"); ax[1, j].plot(d.r, d.MC_Corr, "k.", label="MC")
    ax[1, j].set_title(f"{nm}: corr(X_(r:10), Y_[r:10])"); ax[1, j].set_xlabel("rank r"); ax[1, j].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_order_stats.png", dpi=300)
print("\nsaved to", OUT)