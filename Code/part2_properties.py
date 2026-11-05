import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
import bvge_core as B

OUT = "outputs/part2"; os.makedirs(OUT, exist_ok=True)
SETS = {"S1": (1.5, 1.0, 0.5, 1.0, 6), "S2": (1.0, 1.0, 2.0, 1.0, 6), "S3": (2.0, 0.7, 0.3, 0.5, 10)}
f_est = "outputs/part5_6/table_estimates.csv"          # created by part5_6_uefa_srs.py
if os.path.exists(f_est):
    e = pd.read_csv(f_est, index_col=0)["direct ML"].values
    SETS["UEFA"] = (e[0], e[1], e[2], e[3], 100.0); print("UEFA fitted parameters added:", np.round(e, 5))
ncol = len(SETS); A = B._A

# ---------- marginals GE(k, lam) ----------
def pdf_m(x, k, lam): return k * lam * np.exp(-lam * x) * A(x, lam) ** (k - 1)
def cdf_m(x, k, lam): return A(x, lam) ** k
def haz_m(x, k, lam): return pdf_m(x, k, lam) / (1 - cdf_m(x, k, lam))
def q_m(u, k, lam):   return -np.log1p(-np.asarray(u, float) ** (1.0 / k)) / lam

# ---------- conditional law of X1 given X2 = y (atom at x1 = y); X2 | X1 follows by swapping a1 <-> a2 ----------
def cond_cdf_x1(x, y, a1, a2, a3, lam):
    x = np.asarray(x, float)
    return np.where(x < y, a2/(a2+a3) * A(x, lam)**(a1+a3) * A(y, lam)**(-a3), A(x, lam)**a1)
def cond_pdf_x1(x, y, a1, a2, a3, lam):                 # density of the absolutely continuous part
    x = np.asarray(x, float)
    return np.where(x < y, a2/(a2+a3)*(a1+a3)*lam*np.exp(-lam*x)*A(x, lam)**(a1+a3-1)*A(y, lam)**(-a3),
                    a1*lam*np.exp(-lam*x)*A(x, lam)**(a1-1))
def cond_atom_x1(y, a1, a2, a3, lam): return a3/(a2+a3) * A(y, lam)**a1
def cond_q_x1(u, y, a1, a2, a3, lam):
    u = np.asarray(u, float); Ay = A(y, lam); Fl = a2/(a2+a3)*Ay**a1; Fr = Ay**a1
    q1 = -np.log1p(-np.clip(u*(a2+a3)/a2*Ay**a3, 0, 1) ** (1.0/(a1+a3))) / lam
    q3 = -np.log1p(-np.clip(u, 0, 1) ** (1.0/a1)) / lam
    return np.where(u <= Fl, q1, np.where(u <= Fr, y, q3))

# ---------- hazards (off the diagonal) ----------
def jk_hazard(x1, x2, a1, a2, a3, lam):                 # Johnson-Kotz: h_i = -d ln S / d x_i
    x1 = np.asarray(x1, float); x2 = np.asarray(x2, float); A1, A2 = A(x1, lam), A(x2, lam)
    e1, e2 = np.exp(-lam*x1), np.exp(-lam*x2); lt = x1 < x2
    dF1 = np.where(lt, (a1+a3)*lam*e1*A1**(a1+a3-1)*A2**a2, a1*lam*e1*A1**(a1-1)*A2**(a2+a3))
    dF2 = np.where(lt, a2*lam*e2*A1**(a1+a3)*A2**(a2-1), (a2+a3)*lam*e2*A1**a1*A2**(a2+a3-1))
    S = B.survival(x1, x2, a1, a2, a3, lam)
    return (pdf_m(x1, a1+a3, lam) - dF1)/S, (pdf_m(x2, a2+a3, lam) - dF2)/S
def haz_joint(x1, x2, a1, a2, a3, lam): return B.pdf_ac(x1, x2, a1, a2, a3, lam) / B.survival(x1, x2, a1, a2, a3, lam)

# =============================== verification ===============================
rng = np.random.default_rng(1); ver = {}; PS = [0.1, 0.25, 0.5, 0.75, 0.9]
for nm, (a1, a2, a3, lam, xmax) in SETS.items():
    r = {}; P = rng.uniform(0.05*xmax, 0.8*xmax, (3000, 2)); P = P[np.abs(P[:, 0]-P[:, 1]) > 0.02*xmax]
    u1, u2 = P[:, 0], P[:, 1]; f = B.pdf_ac(u1, u2, a1, a2, a3, lam)
    ra = f / pdf_m(u2, a2+a3, lam); mk = ra > 1e-6
    r["max rel err f(x1|x2) formula vs f/f_X2"] = np.max(np.abs(cond_pdf_x1(u1, u2, a1, a2, a3, lam)[mk] - ra[mk]) / ra[mk])
    rb = f / pdf_m(u1, a1+a3, lam); mk = rb > 1e-6
    r["max rel err f(x2|x1) formula vs f/f_X1"] = np.max(np.abs(cond_pdf_x1(u2, u1, a2, a1, a3, lam)[mk] - rb[mk]) / rb[mk])
    tot = []
    for y in q_m(PS, a2+a3, lam):
        g = lambda t: float(cond_pdf_x1(t, y, a1, a2, a3, lam))
        tot.append(quad(g, 0, y, limit=200)[0] + quad(g, y, np.inf, limit=200)[0] + cond_atom_x1(y, a1, a2, a3, lam))
    r["max|total conditional mass - 1| (X1|X2)"] = np.max(np.abs(np.array(tot) - 1))
    uu = np.linspace(0.001, 0.999, 999); err = 0
    for y in q_m(PS, a2+a3, lam):
        Ay = A(y, lam); Fl, Fr = a2/(a2+a3)*Ay**a1, Ay**a1; m_ = (uu <= Fl) | (uu > Fr)
        err = max(err, np.max(np.abs(cond_cdf_x1(cond_q_x1(uu, y, a1, a2, a3, lam), y, a1, a2, a3, lam)[m_] - uu[m_])))
    r["max|F(Q(u|y)|y) - u| outside the atom"] = err
    r["max|F(Q(u)) - u| marginal"] = np.max(np.abs(cdf_m(q_m(uu, a1+a3, lam), a1+a3, lam) - uu))
    h = 1e-4*xmax; h1, h2 = jk_hazard(u1, u2, a1, a2, a3, lam)
    fd1 = -(np.log(B.survival(u1+h, u2, a1, a2, a3, lam)) - np.log(B.survival(u1-h, u2, a1, a2, a3, lam))) / (2*h)
    fd2 = -(np.log(B.survival(u1, u2+h, a1, a2, a3, lam)) - np.log(B.survival(u1, u2-h, a1, a2, a3, lam))) / (2*h)
    m1, m2 = np.abs(fd1) > 1e-8, np.abs(fd2) > 1e-8
    r["max rel err h1 vs -dlnS/dx1"] = np.max(np.abs(h1[m1]-fd1[m1])/np.abs(fd1[m1]))
    r["max rel err h2 vs -dlnS/dx2"] = np.max(np.abs(h2[m2]-fd2[m2])/np.abs(fd2[m2]))
    r["min h1, h2 (>= 0)"] = min(np.nanmin(h1), np.nanmin(h2)); ver[nm] = r
vt = pd.DataFrame(ver); vt.to_csv(f"{OUT}/table_verification.csv")
pd.set_option("display.width", 220); print(vt.to_string(float_format=lambda v: f"{v:.4g}"))

# ---------- tables: quantiles and atom masses ----------
qrows, arows = [], []
for nm, (a1, a2, a3, lam, xmax) in SETS.items():
    for lab, k in (("X1", a1+a3), ("X2", a2+a3), ("max(X1,X2)", a1+a2+a3)):
        qrows.append(dict(set=nm, variable=lab, **{f"p={p}": q_m(p, k, lam) for p in PS}))
    for p, y in zip(PS, q_m(PS, a2+a3, lam)):
        arows.append(dict(set=nm, p_of_X2=p, x2=y, atom_mass_P_X1_eq_x2=cond_atom_x1(y, a1, a2, a3, lam)))
qt, at = pd.DataFrame(qrows), pd.DataFrame(arows)
qt.to_csv(f"{OUT}/table_quantiles.csv", index=False); at.to_csv(f"{OUT}/table_atom_masses.csv", index=False)
print("\n=== marginal and max quantiles ===\n", qt.round(4).to_string(index=False))
print("\n=== P(X1 = X2 | X2 = x2) at quantiles of X2 ===\n", at.round(4).to_string(index=False))

# =============================== figures ===============================
fig, ax = plt.subplots(3, ncol, figsize=(4.2*ncol, 9), squeeze=False)
for j, (nm, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items()):
    t = np.linspace(xmax/400, xmax, 400); hz = []
    with np.errstate(all="ignore"):
        for k, lab, col in ((a1+a3, "X1", "C0"), (a2+a3, "X2", "C1")):
            ax[0, j].plot(t, pdf_m(t, k, lam), col, label=lab); ax[1, j].plot(t, cdf_m(t, k, lam), col, label=lab)
            hz.append(haz_m(t, k, lam)); ax[2, j].plot(t, hz[-1], col, label=lab)
        ax[2, j].set_ylim(0, np.nanpercentile(np.concatenate(hz), 95) * 1.5)
    for i, ttl in enumerate(("marginal pdf", "marginal CDF", "marginal hazard")):
        ax[i, j].set_title(f"{nm}: {ttl}"); ax[i, j].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_marginals.png", dpi=300)

fig = plt.figure(figsize=(4.6*ncol, 4.6))
for j, (nm, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items(), 1):
    g_ = np.linspace(xmax/80, xmax, 70); X1, X2 = np.meshgrid(g_, g_)
    with np.errstate(all="ignore"): Z = haz_joint(X1, X2, a1, a2, a3, lam)
    cap = np.nanpercentile(Z, 95); Z = np.minimum(np.nan_to_num(Z, nan=cap, posinf=cap), cap)
    ax3 = fig.add_subplot(1, ncol, j, projection="3d"); ax3.plot_surface(X1, X2, Z, cmap="viridis", linewidth=0)
    ax3.set_title(f"{nm}: bivariate hazard f/S (off-diagonal)", fontsize=9); ax3.set_xlabel("x1"); ax3.set_ylabel("x2")
fig.savefig(f"{OUT}/fig_joint_hazard.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(2, ncol, figsize=(4.4*ncol, 7), squeeze=False)
for j, (nm, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items()):
    t = np.linspace(xmax/600, xmax, 600)
    for p, col in zip((0.25, 0.5, 0.75), ("C0", "C1", "C2")):
        y = q_m(p, a2+a3, lam); at_ = cond_atom_x1(y, a1, a2, a3, lam)
        with np.errstate(all="ignore"):
            d = cond_pdf_x1(t, y, a1, a2, a3, lam); ax[0, j].plot(t[t < y], d[t < y], col); ax[0, j].plot(t[t > y], d[t > y], col, label=f"X2={y:.2f}, atom={at_:.2f}")
            ax[1, j].plot(t, cond_cdf_x1(t, y, a1, a2, a3, lam), col, label=f"X2={y:.2f}")
        ax[0, j].axvline(y, c=col, ls=":", lw=.8); ax[1, j].axvline(y, c=col, ls=":", lw=.8)
    ax[0, j].set_ylim(0, np.nanpercentile(cond_pdf_x1(t, q_m(0.5, a2+a3, lam), a1, a2, a3, lam), 99)*1.6)
    ax[0, j].set_title(f"{nm}: f(x1 | X2=y), a.c. part (+ atom at x1=y)", fontsize=9); ax[1, j].set_title(f"{nm}: F(x1 | X2=y)", fontsize=9)
    ax[0, j].legend(fontsize=7); ax[1, j].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_conditional.png", dpi=300)

fig, ax = plt.subplots(1, ncol, figsize=(4.4*ncol, 4.2), squeeze=False)
for j, (nm, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items()):
    ys = np.linspace(xmax/200, xmax, 300)
    for u in (0.1, 0.25, 0.5, 0.75, 0.9): ax[0, j].plot(ys, cond_q_x1(u, ys, a1, a2, a3, lam), label=f"u={u}")
    ax[0, j].plot(ys, ys, "k--", lw=.7); ax[0, j].set_title(f"{nm}: conditional quantiles of X1 given X2=y", fontsize=9)
    ax[0, j].set_xlabel("y"); ax[0, j].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_conditional_quantiles.png", dpi=300)
print("\nsaved to", OUT)