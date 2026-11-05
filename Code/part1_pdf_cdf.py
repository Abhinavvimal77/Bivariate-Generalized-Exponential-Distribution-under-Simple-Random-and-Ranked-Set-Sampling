import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import bvge_core as B

OUT = "outputs/part1"; os.makedirs(OUT, exist_ok=True)
# illustrative sets (a1, a2, a3, lam, plotting range); we replace/add the fitted set after Part 5
SETS = {"S1": (1.5, 1.0, 0.5, 1.0, 6), "S2": (1.0, 1.0, 2.0, 1.0, 6), "S3": (2.0, 0.7, 0.3, 0.5, 10)}
pd.DataFrame(SETS, index=["alpha1", "alpha2", "alpha3", "lambda", "xmax"]).T.to_csv(f"{OUT}/table1_param_sets.csv")

rng = np.random.default_rng(2024)
res = {}
for name, (a1, a2, a3, lam, xmax) in SETS.items():
    s = a1 + a2 + a3
    r = {}
    mc = B.mass_check(a1, a2, a3, lam); r.update(mc)
    g = np.linspace(0.02, xmax, 150); X1, X2 = np.meshgrid(g, g)
    r["min pdf_ac on grid (>=0)"] = np.nanmin(B.pdf_ac(X1, X2, a1, a2, a3, lam))
    r["min pdf_diag on grid (>=0)"] = np.nanmin(B.pdf_diag(g, a1, a2, a3, lam))
    xs = np.linspace(0.05, xmax, 60)
    r["max|F(x,inf)-F1(x)|"] = np.max(np.abs(B.cdf(xs, 1e3, a1, a2, a3, lam) - B.cdf_marg1(xs, a1, a3, lam)))
    r["max|F(inf,y)-F2(y)|"] = np.max(np.abs(B.cdf(1e3, xs, a1, a2, a3, lam) - B.cdf_marg2(xs, a2, a3, lam)))
    r["F(0,y) and F(x,0) max"] = max(np.max(B.cdf(0, xs, a1, a2, a3, lam)), np.max(B.cdf(xs, 0, a1, a2, a3, lam)))
    r["F(inf,inf)"] = float(B.cdf(1e3, 1e3, a1, a2, a3, lam))
    # mixed partial of F vs pdf_ac (off-diagonal)
    h = 1e-3 / lam
    P = rng.uniform(0.2, xmax, (4000, 2)); P = P[np.abs(P[:, 0] - P[:, 1]) > 0.05]
    x1, x2 = P[:, 0], P[:, 1]
    F = lambda a, b: B.cdf(a, b, a1, a2, a3, lam)
    fd = (F(x1+h, x2+h) - F(x1+h, x2-h) - F(x1-h, x2+h) + F(x1-h, x2-h)) / (4*h*h)
    ex = B.pdf_ac(x1, x2, a1, a2, a3, lam); mk = ex > 1e-3
    r["max rel|d2F/dx1dx2 - f_ac|"] = np.max(np.abs(fd[mk] - ex[mk]) / ex[mk])
    # rectangle inequality
    a = rng.uniform(0, xmax, (100000, 2)); b = a + rng.uniform(0, xmax/2, (100000, 2))
    rect = F(b[:, 0], b[:, 1]) - F(a[:, 0], b[:, 1]) - F(b[:, 0], a[:, 1]) + F(a[:, 0], a[:, 1])
    r["min rectangle prob (>=0)"] = rect.min()
    # abs. continuous part of the CDF (Thm 2.3) must be a valid CDF
    p = (a1 + a2) / s; zz = np.minimum(X1, X2)
    Fa = (F(X1, X2) - (a3 / s) * B._A(zz, lam) ** s) / p
    r["min F_a on grid (>=0)"] = Fa.min()
    # Monte Carlo from the U1,U2,U3 construction
    N = 400000; y1, y2 = B.simulate(N, a1, a2, a3, lam, rng)
    r["P(X1=X2): theory"] = B.p_tie(a1, a2, a3); r["P(X1=X2): simulated"] = np.mean(y1 == y2)
    qa, qb = np.quantile(y1, [.2, .4, .6, .8]), np.quantile(y2, [.2, .4, .6, .8])
    r["max|F_emp - F| (4x4 grid)"] = max(abs(np.mean((y1 <= u) & (y2 <= v)) - F(u, v)) for u in qa for v in qb)
    r["KS p-value X1 ~ GE(a1+a3)"] = stats.kstest(y1, lambda t: B.cdf_marg1(t, a1, a3, lam)).pvalue
    r["KS p-value X2 ~ GE(a2+a3)"] = stats.kstest(y2, lambda t: B.cdf_marg2(t, a2, a3, lam)).pvalue
    res[name] = r
tab = pd.DataFrame(res); tab.to_csv(f"{OUT}/table2_verification.csv")
pd.set_option("display.width", 200); print(tab.to_string(float_format=lambda v: f"{v:.6g}"))

# ---------- figures
def grid(xmax, n, lo=0.0):
    g_ = np.linspace(lo, xmax, n); return np.meshgrid(g_, g_)

fig = plt.figure(figsize=(15, 4.6))
for k, (name, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items(), 1):
    X1, X2 = grid(xmax, 60); ax = fig.add_subplot(1, 3, k, projection="3d")
    ax.plot_surface(X1, X2, B.cdf(X1, X2, a1, a2, a3, lam), cmap="viridis", linewidth=0)
    ax.set_title(f"{name}: ({a1}, {a2}, {a3}, {lam})", fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("F(x1,x2)")
fig.savefig(f"{OUT}/fig1_cdf_surfaces.png", dpi=300, bbox_inches="tight")

fig = plt.figure(figsize=(15, 4.6))
for k, (name, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items(), 1):
    X1, X2 = grid(xmax, 70, lo=xmax / 80); Z = B.pdf_ac(X1, X2, a1, a2, a3, lam)
    Z = np.minimum(Z, np.nanpercentile(Z, 98)); ax = fig.add_subplot(1, 3, k, projection="3d")
    ax.plot_surface(X1, X2, Z, cmap="viridis", linewidth=0)
    ax.set_title(f"{name}: abs. continuous pdf (capped at 98th pct)", fontsize=9)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f_ac")
fig.savefig(f"{OUT}/fig2_pdf_surfaces.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for k, (name, (a1, a2, a3, lam, xmax)) in enumerate(SETS.items()):
    X1, X2 = grid(xmax, 200, lo=xmax / 200); Z = B.pdf_ac(X1, X2, a1, a2, a3, lam)
    Z = np.minimum(Z, np.nanpercentile(Z, 98))
    cs = ax[k].contourf(X1, X2, Z, 25, cmap="magma"); plt.colorbar(cs, ax=ax[k])
    ax[k].plot([0, xmax], [0, xmax], "w--", lw=1)
    ax[k].set_title(f"{name}: P(X1=X2)={B.p_tie(a1, a2, a3):.3f} on the dashed diagonal", fontsize=9)
    ax[k].set_xlabel("x1"); ax[k].set_ylabel("x2")
fig.savefig(f"{OUT}/fig3_pdf_contours.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for name, (a1, a2, a3, lam, xmax) in SETS.items():
    x = np.linspace(xmax / 400, xmax, 400); ax[0].plot(x, B.pdf_diag(x, a1, a2, a3, lam), label=name)
ax[0].set_title("Singular part: density along x1 = x2"); ax[0].set_xlabel("x"); ax[0].legend()
a1, a2, a3, lam, xmax = SETS["S1"]; y1, y2 = B.simulate(600, a1, a2, a3, lam, rng); t = y1 == y2
ax[1].scatter(y1[~t], y2[~t], s=10, alpha=.6); ax[1].scatter(y1[t], y2[t], s=14, c="red", label="ties")
ax[1].plot([0, y1.max()], [0, y1.max()], "k--", lw=.8); ax[1].legend(); ax[1].set_title("Simulated sample, S1 (n=600)")
fig.savefig(f"{OUT}/fig4_singular_and_sample.png", dpi=300, bbox_inches="tight")
print("saved to", OUT)