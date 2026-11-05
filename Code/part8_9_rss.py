import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import beta as Beta, kstest, shapiro, skew, kurtosis, probplot, norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "outputs/part8_9"; os.makedirs(OUT, exist_ok=True)
PN = ["alpha1", "alpha2", "alpha3", "lambda"]; Z = 1.959964
TRUE = np.array([1.44523, 0.46814, 1.17035, 0.03899])
f_est = "outputs/part5_6/table_estimates.csv"
if os.path.exists(f_est): TRUE = pd.read_csv(f_est, index_col=0)["direct ML"].values[:4].astype(float)
M_SET, C_CYC = 5, 8                 # set size m, cycles c  -> n = 40
B_GOF, R_AS, C_EXP, N_AS_C = 200, 400, 20000, 20     # bootstrap GOF reps, asymptotic MC reps, expected-info cycles, cycles for MC (n=100)
if len(sys.argv) > 1: B_GOF, R_AS, C_EXP = 20, 40, 2000       # quick test:  python part8_9_rss.py q
rng = np.random.default_rng(8); pd.set_option("display.width", 220)

# ---------------- simulation ----------------
def sim_pairs(shape, th, rng):
    a = np.array(th[:3], float).reshape((3,) + (1,)*len(shape)); V = rng.random((3,) + shape)
    U = -np.log1p(-V**(1/a))/th[3]; return np.maximum(U[0], U[2]), np.maximum(U[1], U[2])
def sim_srs(n, th, rng):
    x1, x2 = sim_pairs((n,), th, rng); return (x1, x2, np.ones(n))
def sim_rss(m, c, th, rng):
    X1, X2 = sim_pairs((c, m, m), th, rng); o = np.argsort(X1, axis=2)
    X1s, X2s = np.take_along_axis(X1, o, 2), np.take_along_axis(X2, o, 2); ar = np.arange(m)
    return X1s[:, ar, ar].ravel(), X2s[:, ar, ar].ravel(), np.tile(ar+1.0, (c, 1)).ravel()

# ---------------- likelihoods (m = 1 gives SRS) ----------------
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

def loglik_mo(t, d, m):
    l1, l2, l12 = t; x1, x2, r = d; lt, gt = x1 < x2, x1 > x2; eq = ~(lt | gt); lf = np.empty_like(x1)
    lf[eq] = np.log(l12) - (l1+l2+l12)*x1[eq]
    lf[lt] = np.log(l1) + np.log(l2+l12) - l1*x1[lt] - (l2+l12)*x2[lt]
    lf[gt] = np.log(l2) + np.log(l1+l12) - (l1+l12)*x1[gt] - l2*x2[gt]
    if m > 1:
        rho = l1 + l12; lF = np.log(-np.expm1(-rho*x1)); l1F = -rho*x1
        lf = lf + (r-1)*lF + (m-r)*l1F + gammaln(m+1) - gammaln(r) - gammaln(m-r+2)
    return lf.sum()

def fit(d, m, fixed=None, start=None):
    lam0 = 1/np.mean(np.concatenate([d[0], d[1]])); x1, x2 = d[0], d[1]
    lo = np.array([np.log(1e-3)]*3 + [np.log(lam0)-6]); hi = np.array([np.log(100)]*3 + [np.log(lam0)+6])
    pl, pg, pt = np.mean(x1 < x2), np.mean(x1 > x2), np.mean(x1 == x2)
    starts = [np.array([1, 1, 1, lam0]), np.array([max(2*pg, .1), max(2*pl, .1), max(2*pt, .1), lam0*1.5])]
    if start is not None: starts = [np.asarray(start, float)]
    fixed = fixed or {}; free = [i for i in range(4) if i not in fixed]
    def full(u):
        t = np.empty(4); t[free] = np.exp(u)
        for i, v in fixed.items(): t[i] = v
        return t
    def f(u):
        v = loglik(full(u), d, m); return -v if np.isfinite(v) else 1e12
    best = None
    for s0 in starts:
        s = np.array(s0, float)
        for i, v in fixed.items(): s[i] = v
        r_ = minimize(f, np.log(s[free]), method="L-BFGS-B", bounds=[(lo[i], hi[i]) for i in free])
        if best is None or r_.fun < best.fun: best = r_
    hit = bool(np.any((best.x < lo[free]+1e-4) | (best.x > hi[free]-1e-4)))
    return full(best.x), -best.fun, hit

def fit_mo(d, m):
    l0 = 1/np.mean(np.concatenate([d[0], d[1]])); best = None
    for s in ([.3, .3, .3], [.1, .5, .5], [.5, .1, .5]):
        r_ = minimize(lambda u: -loglik_mo(np.exp(u), d, m) if np.isfinite(loglik_mo(np.exp(u), d, m)) else 1e12,
                      np.log(np.array(s)*l0), method="L-BFGS-B", bounds=[(np.log(l0)-8, np.log(l0)+4)]*3)
        if best is None or r_.fun < best.fun: best = r_
    return np.exp(best.x), -best.fun

def hess(t, d, m, ll=loglik):
    k = len(t); H = np.zeros((k, k)); h = 1e-3*np.abs(t); f = lambda v: -ll(v, d, m); f0 = f(t)
    for i in range(k):
        ei = np.zeros(k); ei[i] = h[i]; H[i, i] = (f(t+ei) + f(t-ei) - 2*f0)/h[i]**2
        for j in range(i):
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(t+ei+ej) - f(t+ei-ej) - f(t-ei+ej) + f(t-ei-ej))/(4*h[i]*h[j])
    return H

# ---------------- PIT-based goodness of fit ----------------
def pit(t, d, m, rng):
    a1, a2, a3, lam = t; x1, x2, r = d; A1 = -np.expm1(-lam*x1); A2 = -np.expm1(-lam*x2)
    u1 = Beta.cdf(np.minimum(A1**(a1+a3), 1-1e-15), r, m-r+1)           # X1_(r:m) PIT
    u2 = np.empty_like(x1); lt, gt = x2 < x1, x2 > x1; eq = ~(lt | gt)  # randomized PIT of X2 | X1
    u2[lt] = a1/(a1+a3)*A2[lt]**(a2+a3)*A1[lt]**(-a3); u2[gt] = A2[gt]**a2
    Fl, Fr = a1/(a1+a3)*A1[eq]**a2, A1[eq]**a2; u2[eq] = Fl + rng.random(eq.sum())*(Fr-Fl)
    return u1, u2
def ks_u(u): return kstest(u, "uniform")

def gof(d, m, th, design, B, rng):
    u1, u2 = pit(th, d, m, rng); k1, k2 = ks_u(u1), ks_u(u2); n = len(d[0]); D = np.zeros((B, 2))
    for b in range(B):
        db = sim_rss(m, n//m, th, rng) if design == "rss" else sim_srs(n, th, rng)
        tb, _, _ = fit(db, m); v1, v2 = pit(tb, db, m, rng); D[b] = ks_u(v1).statistic, ks_u(v2).statistic
    return dict(D1=k1.statistic, p1_asym=k1.pvalue, p1_boot=(1+np.sum(D[:, 0] >= k1.statistic))/(B+1),
                D2=k2.statistic, p2_asym=k2.pvalue, p2_boot=(1+np.sum(D[:, 1] >= k2.statistic))/(B+1)), (u1, u2)

# ---------------- profile likelihood ----------------
def profile(d, m, t_hat, ll_hat, i, span, npts=25):
    vals = t_hat[i]*np.exp(np.linspace(-span, span, npts)); dev = []
    for v in vals:
        _, ll, _ = fit(d, m, fixed={i: v}, start=t_hat); dev.append(max(2*(ll_hat-ll), 0))
    return vals, np.array(dev)
def crossing(vals, dev, t0, q=3.841459):
    lo = hi = np.nan; Lm, Rm = vals <= t0, vals >= t0
    if dev[Lm].max() >= q: lo = np.interp(q, dev[Lm][::-1], vals[Lm][::-1])
    if dev[Rm].max() >= q: hi = np.interp(q, dev[Rm], vals[Rm])
    return lo, hi

# =========================== 1. data ===========================
n = M_SET*C_CYC; print(f"True parameters (UEFA-fitted BVGE): {np.round(TRUE, 5)}   RSS: m={M_SET}, c={C_CYC}, n={n}")
dR, dS = sim_rss(M_SET, C_CYC, TRUE, rng), sim_srs(n, TRUE, rng)
pd.DataFrame(dict(x1=dR[0], x2=dR[1], rank=dR[2])).to_csv(f"{OUT}/rss_sample.csv", index=False)
pd.DataFrame(dict(x1=dS[0], x2=dS[1])).to_csv(f"{OUT}/srs_sample.csv", index=False)
print(f"ties: RSS {np.sum(dR[0]==dR[1])}/{n}   SRS {np.sum(dS[0]==dS[1])}/{n}")

# =========================== 2. fits, IC ===========================
res = {}
for nm, d, m in (("RSS", dR, M_SET), ("SRS", dS, 1)):
    t, ll, hit = fit(d, m); tm, llm = fit_mo(d, m); H = hess(t, d, m); ev = np.linalg.eigvalsh(H); V = np.linalg.inv(H)
    res[nm] = dict(d=d, m=m, t=t, ll=ll, tm=tm, llm=llm, H=H, ev=ev, V=V, se=np.sqrt(np.diag(V)), hit=hit)
rows = []
for nm in res:
    r = res[nm]
    for mod, k, ll in (("BVGE", 4, r["ll"]), ("Marshall-Olkin", 3, r["llm"])):
        rows.append(dict(design=nm, model=mod, k=k, logL=ll, AIC=-2*ll+2*k, BIC=-2*ll+k*np.log(n), HQIC=-2*ll+2*k*np.log(np.log(n))))
IC = pd.DataFrame(rows); IC.to_csv(f"{OUT}/table_IC.csv", index=False)
EST = pd.DataFrame({"true": TRUE, "RSS_MLE": res["RSS"]["t"], "RSS_SE": res["RSS"]["se"], "SRS_MLE": res["SRS"]["t"], "SRS_SE": res["SRS"]["se"]}, index=PN)
EST["SE_ratio_SRS/RSS"] = EST.SRS_SE/EST.RSS_SE; EST.to_csv(f"{OUT}/table_estimates.csv")
print("\n=== BVGE estimates (RSS vs SRS, same n) ===\n", EST.round(5).to_string())
print("MO estimates (l1,l2,l12): RSS", np.round(res["RSS"]["tm"], 5), "| SRS", np.round(res["SRS"]["tm"], 5))
print("\n=== information criteria (RSS logL includes rank terms, so RSS and SRS logL are not comparable; compare models within a design) ===\n", IC.round(3).to_string(index=False))

# =========================== 3. GOF ===========================
rows = []; PIT = {}
for nm in res:
    r = res[nm]; g, PIT[nm] = gof(r["d"], r["m"], r["t"], nm.lower(), B_GOF, rng); rows.append(dict(design=nm, **g))
G = pd.DataFrame(rows); G.to_csv(f"{OUT}/table_gof.csv", index=False)
print("\n=== PIT-based KS tests (U1: order-statistic PIT of X1; U2: randomized PIT of X2|X1) ===\n", G.round(4).to_string(index=False))

# =========================== 4. observed information, CIs ===========================
for nm in res:
    r = res[nm]; print(f"\n=== {nm}: observed information (eigenvalues {np.round(r['ev'], 3)}) ===")
    print(pd.DataFrame(r["H"], PN, PN).round(3).to_string())
    sd = np.sqrt(np.diag(r["V"])); print(f"--- {nm}: correlation of estimators ---\n", pd.DataFrame(r["V"]/np.outer(sd, sd), PN, PN).round(3).to_string())
rows = []; PROF = {}
for nm in res:
    r = res[nm]; t, se = r["t"], r["se"]
    for i, p in enumerate(PN):
        vals, dev = profile(r["d"], r["m"], t, r["ll"], i, 0.7 if i == 3 else 1.5); PROF[(nm, i)] = (vals, dev); plo, phi = crossing(vals, dev, t[i])
        rows.append(dict(design=nm, param=p, MLE=t[i], SE=se[i], Wald_lo=t[i]-Z*se[i], Wald_hi=t[i]+Z*se[i],
                         logWald_lo=t[i]*np.exp(-Z*se[i]/t[i]), logWald_hi=t[i]*np.exp(Z*se[i]/t[i]), Prof_lo=plo, Prof_hi=phi,
                         Wald_len=2*Z*se[i], true=TRUE[i]))
CI = pd.DataFrame(rows); CI.to_csv(f"{OUT}/table_CI.csv", index=False)
print("\n=== 95% confidence intervals ===\n", CI.round(4).to_string(index=False))

# =========================== 5. expected information / asymptotics ===========================
dE_R, dE_S = sim_rss(M_SET, C_EXP, TRUE, rng), sim_srs(M_SET*C_EXP, TRUE, rng); nE = M_SET*C_EXP
IR, IS = hess(TRUE, dE_R, M_SET)/nE, hess(TRUE, dE_S, 1)/nE; VR, VS = np.linalg.inv(IR), np.linalg.inv(IS)
AS = pd.DataFrame({"I_RSS_diag": np.diag(IR), "I_SRS_diag": np.diag(IS), "AVar_RSS(per obs)": np.diag(VR), "AVar_SRS(per obs)": np.diag(VS),
                   "ARE_i=Var_SRS/Var_RSS": np.diag(VS)/np.diag(VR)}, index=PN)
are_det = (np.linalg.det(IR)/np.linalg.det(IS))**(1/4); AS.to_csv(f"{OUT}/table_expected_info.csv")
print(f"\n=== expected information per observation (m={M_SET}, simulated n={nE}) ===\n", AS.round(5).to_string())
print(f"determinant-based ARE (det I_RSS/det I_SRS)^(1/4) = {are_det:.4f}")
print("expected-information SE at n=%d: RSS" % n, np.round(np.sqrt(np.diag(VR)/n), 5), "| SRS", np.round(np.sqrt(np.diag(VS)/n), 5))

# =========================== 6. Monte Carlo check of asymptotic normality (RSS, n=100) ===========================
n_as = M_SET*N_AS_C; E = []
for _ in range(R_AS):
    d = sim_rss(M_SET, N_AS_C, TRUE, rng); t, _, hit = fit(d, M_SET)
    if not hit: E.append(t)
E = np.array(E); se_as = np.sqrt(np.diag(VR)/n_as); Zs = (E-TRUE)/se_as; rows = []
for i, p in enumerate(PN):
    rows.append(dict(param=p, true=TRUE[i], mean=E[:, i].mean(), bias=E[:, i].mean()-TRUE[i], emp_SD=E[:, i].std(ddof=1), asym_SE=se_as[i],
                     SD_over_SE=E[:, i].std(ddof=1)/se_as[i], skew=skew(E[:, i]), ex_kurt=kurtosis(E[:, i]),
                     shapiro_p=shapiro(E[:, i])[1], shapiro_p_log=shapiro(np.log(E[:, i]))[1]))
ASM = pd.DataFrame(rows); ASM.to_csv(f"{OUT}/table_asymptotic_check.csv", index=False)
Cemp = np.corrcoef(E.T); Casy = VR/np.sqrt(np.outer(np.diag(VR), np.diag(VR)))
print(f"\n=== asymptotic check: {len(E)} RSS replicates, m={M_SET}, n={n_as} ===\n", ASM.round(4).to_string(index=False))
print("empirical correlation of estimators:\n", pd.DataFrame(Cemp, PN, PN).round(3).to_string())
print("asymptotic correlation (expected information):\n", pd.DataFrame(Casy, PN, PN).round(3).to_string())

# =========================== figures ===========================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for a_, (nm, d) in zip(ax, (("RSS", dR), ("SRS", dS))):
    a_.scatter(d[0], d[1], s=14); mx = max(d[0].max(), d[1].max()); a_.plot([0, mx], [0, mx], "k--", lw=.7); a_.set_title(f"{nm} sample (n={n})"); a_.set_xlabel("x1"); a_.set_ylabel("x2")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_samples.png", dpi=300)
fig, ax = plt.subplots(2, 2, figsize=(9, 8))
for i, nm in enumerate(("RSS", "SRS")):
    for j in range(2):
        u = np.sort(PIT[nm][j]); ax[i, j].plot((np.arange(1, n+1)-.5)/n, u, "o", ms=4); ax[i, j].plot([0, 1], [0, 1], "k--")
        ax[i, j].set_title(f"{nm}: PIT QQ, U{j+1} ({'X1 order stat.' if j == 0 else 'X2 | X1'})", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pit_qq.png", dpi=300)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for i_, nm in enumerate(("RSS", "SRS")):
    for i, p in enumerate(PN):
        v, dv = PROF[(nm, i)]; ax[i_, i].plot(v, dv); ax[i_, i].axhline(3.841, c="r", ls="--"); ax[i_, i].axvline(res[nm]["t"][i], c="k", lw=.6)
        ax[i_, i].axvline(TRUE[i], c="g", ls=":"); ax[i_, i].set_ylim(0, 10); ax[i_, i].set_title(f"{nm}: profile deviance, {p}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_profile_deviance.png", dpi=300)
fig, ax = plt.subplots(2, 4, figsize=(16, 7)); xs = np.linspace(-4, 4, 200)
for i, p in enumerate(PN):
    probplot(Zs[:, i], dist="norm", plot=ax[0, i]); ax[0, i].set_title(f"QQ of (est - true)/asym.SE: {p}", fontsize=9)
    ax[1, i].hist(Zs[:, i], bins=25, density=True, alpha=.7); ax[1, i].plot(xs, norm.pdf(xs), "r"); ax[1, i].set_title(f"{p}: standardized estimator")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_asymptotic_normality.png", dpi=300)
print("\nsaved to", OUT)