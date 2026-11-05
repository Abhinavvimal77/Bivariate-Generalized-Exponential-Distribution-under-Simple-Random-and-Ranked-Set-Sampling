import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma
from scipy.integrate import quad
import bvge_mom as M

OUT = "outputs/part3"; os.makedirs(OUT, exist_ok=True)
SETS = {"S1": (1.5, 1.0, 0.5, 1.0), "S2": (1.0, 1.0, 2.0, 1.0), "S3": (2.0, 0.7, 0.3, 0.5)}
f_est = "outputs/part5_6/table_estimates.csv"
if os.path.exists(f_est):
    e = pd.read_csv(f_est, index_col=0)["direct ML"].values; SETS["UEFA"] = tuple(e[:4])
rng = np.random.default_rng(7); A = lambda x, lam: 1 - np.exp(-lam*x)
pd.set_option("display.width", 220)

def sim(n, a1, a2, a3, lam):
    V = rng.random((3, n)); U = -np.log1p(-V**(1/np.array([a1, a2, a3])[:, None]))/lam
    return np.maximum(U[0], U[2]), np.maximum(U[1], U[2])
def gm(k, lam):  # GE(k,lam): mean, var
    return (digamma(k+1)-digamma(1))/lam, (polygamma(1, 1)-polygamma(1, k+1))/lam**2
def q_m(u, k, lam): return -np.log1p(-np.asarray(u, float)**(1/k))/lam
def covcor(a1, a2, a3, lam):
    m1, v1 = gm(a1+a3, lam); m2, v2 = gm(a2+a3, lam); e12 = M.EG(lambda x, y: x*y, a1, a2, a3, lam)
    c = e12 - m1*m2; return m1, m2, v1, v2, e12, c, c/np.sqrt(v1*v2)

# ---------- A. raw joint moments E[X1^r X2^s] vs Monte Carlo ----------
RS = [(1,0), (0,1), (2,0), (0,2), (1,1), (2,1), (1,2), (2,2)]; rows = []
for nm, (a1, a2, a3, lam) in SETS.items():
    x1, x2 = sim(1_000_000, a1, a2, a3, lam)
    for r, s in RS:
        th = M.EG(lambda x, y: x**r*y**s, a1, a2, a3, lam); mc = np.mean(x1**r*x2**s)
        rows.append(dict(set=nm, r=r, s=s, theory=th, MC=mc, rel_err=abs(mc/th-1)))
tA = pd.DataFrame(rows); tA.to_csv(f"{OUT}/table_joint_moments.csv", index=False)
print("=== E[X1^r X2^s]: numerical integration vs Monte Carlo (1e6) ===\n", tA.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

# ---------- B. mean, variance, covariance, correlation ----------
rows = []
for nm, (a1, a2, a3, lam) in SETS.items():
    m1, m2, v1, v2, e12, c, rho = covcor(a1, a2, a3, lam); x1, x2 = sim(1_000_000, a1, a2, a3, lam)
    rows.append(dict(set=nm, EX1=m1, EX2=m2, VarX1=v1, VarX2=v2, E_X1X2=e12, Cov=c, Corr=rho,
                     Corr_MC=np.corrcoef(x1, x2)[0, 1], P_tie=a3/(a1+a2+a3)))
tB = pd.DataFrame(rows); tB.to_csv(f"{OUT}/table_cov_corr.csv", index=False)
print("\n=== moments, covariance, correlation ===\n", tB.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

# ---------- C. correlation versus alpha3 ----------
A3 = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]; CFG = {"a1=a2=1": (1, 1), "a1=1.5,a2=0.5": (1.5, 0.5), "a1=0.5,a2=2": (0.5, 2)}
rows = []
for lab, (a1, a2) in CFG.items():
    for a3 in A3:
        _, _, _, _, _, c, rho = covcor(a1, a2, a3, 1.0)
        rows.append(dict(config=lab, alpha3=a3, Cov=c, Corr=rho, P_tie=a3/(a1+a2+a3)))
tC = pd.DataFrame(rows); tC.to_csv(f"{OUT}/table_corr_vs_alpha3.csv", index=False)
print("\n=== correlation vs alpha3 (lambda=1) ===\n", tC.pivot(index="alpha3", columns="config", values="Corr").round(4).to_string())
print("\n=== tie probability vs alpha3 ===\n", tC.pivot(index="alpha3", columns="config", values="P_tie").round(4).to_string())

# ---------- D. conditional moments ----------
PS = [0.1, 0.25, 0.5, 0.75, 0.9]; rows = []; chk = []
for nm, (a1, a2, a3, lam) in SETS.items():
    for tgt, (aA, aB) in (("X1|X2", (a1, a2)), ("X2|X1", (a2, a1))):
        for p in PS:
            y = q_m(p, aB+a3, lam); v = A(y, lam)
            m = M.cm(1, v, aA, aB, a3, lam); m2 = M.cm(2, v, aA, aB, a3, lam)
            rows.append(dict(set=nm, target=tgt, p=p, cond_on=y, E=m, Var=m2-m**2, atom=a3/(aB+a3)*v**aA))
        # total-expectation checks: E[E(X_A | X_B)] = E[X_A];  E[E(X_A^2|X_B)] = E[X_A^2]
        mA, vA = gm(aA+a3, lam)
        for r, tv in ((1, mA), (2, vA+mA**2)):
            val = quad(lambda w: M.cm(r, w, aA, aB, a3, lam)*(aB+a3)*w**(aB+a3-1), 0, 1, epsabs=1e-9, epsrel=1e-9)[0]
            chk.append(dict(set=nm, target=tgt, moment=r, tot_exp=val, truth=tv, rel_err=abs(val/tv-1)))
        chk.append(dict(set=nm, target=tgt, moment=0, tot_exp=M.cm(0, A(q_m(.5, aB+a3, lam), lam), aA, aB, a3, lam), truth=1.0, rel_err=np.nan))
tD, tchk = pd.DataFrame(rows), pd.DataFrame(chk)
tD.to_csv(f"{OUT}/table_conditional_moments.csv", index=False); tchk.to_csv(f"{OUT}/table_cond_checks.csv", index=False)
print("\n=== conditional mean/variance at quantiles of the conditioning variable ===\n", tD.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
print("\n=== total-expectation checks (cond. moments averaged over the conditioning law) ===\n", tchk.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(6, 4.4))
for lab in CFG: d = tC[tC.config == lab]; ax.plot(d.alpha3, d.Corr, "o-", label=lab)
ax.set_xscale("log"); ax.set_xlabel("alpha3"); ax.set_ylabel("corr(X1,X2)"); ax.legend(); ax.set_title("Correlation vs alpha3 (lambda=1)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_corr_vs_alpha3.png", dpi=300)

nS = len(SETS); fig, ax = plt.subplots(2, nS, figsize=(4.2*nS, 7), squeeze=False)
for j, (nm, (a1, a2, a3, lam)) in enumerate(SETS.items()):
    ys = q_m(np.linspace(0.02, 0.98, 40), a2+a3, lam); vs = A(ys, lam)
    em = [M.cm(1, v, a1, a2, a3, lam) for v in vs]; ev = [M.cm(2, v, a1, a2, a3, lam)-M.cm(1, v, a1, a2, a3, lam)**2 for v in vs]
    ax[0, j].plot(ys, em); ax[0, j].plot(ys, ys, "k--", lw=.7); ax[0, j].set_title(f"{nm}: E[X1 | X2=y]"); ax[0, j].set_xlabel("y")
    ax[1, j].plot(ys, ev); ax[1, j].set_title(f"{nm}: Var[X1 | X2=y]"); ax[1, j].set_xlabel("y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_conditional_moments.png", dpi=300)
print("\nsaved to", OUT)