import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bvge_core as B

OUT = "outputs/part2"; os.makedirs(OUT, exist_ok=True)
A = B._A

def q_m(u, k, lam): return -np.log1p(-np.asarray(u, float)**(1/k))/lam
def h_m(x, k, lam):                       # marginal GE(k, lam) hazard
    return k*lam*np.exp(-lam*x)*A(x, lam)**(k-1)/(1 - A(x, lam)**k)
def haz_joint(x1, x2, a1, a2, a3, lam):   # f / S off the diagonal
    return B.pdf_ac(x1, x2, a1, a2, a3, lam)/B.survival(x1, x2, a1, a2, a3, lam)

# (alpha1, alpha2, alpha3, lambda)
CASES = {
    "A_left":  (0.3, 0.4, 0.2, 1.0),   # k1=0.5, k2=0.6  -> both marginal hazards decreasing
    "A_right": (2.0, 3.0, 1.5, 1.0),   # k1=3.5, k2=4.5  -> both increasing
    "B_left":  (0.3, 2.0, 0.2, 1.0),   # k1=0.5, k2=2.2  -> X1 decreasing, X2 increasing (mixed)
    "B_right": (0.7, 0.7, 0.3, 1.0),   # k1=k2=1         -> constant marginal hazard (= lambda)
}

def surface(ax, P):
    a1, a2, a3, lam = P; k1, k2 = a1+a3, a2+a3
    xmax = max(q_m(0.95, k1, lam), q_m(0.95, k2, lam)); g = np.linspace(xmax/60, xmax, 60); X1, X2 = np.meshgrid(g, g)
    with np.errstate(all="ignore"): Z = haz_joint(X1, X2, a1, a2, a3, lam)
    cap = np.nanpercentile(Z, 95); Z = np.minimum(np.nan_to_num(Z, nan=cap, posinf=cap), cap)
    ax.plot_surface(X1, X2, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$h(x_1,x_2)$")
    ax.set_title(rf"$\alpha_1={a1},\ \alpha_2={a2},\ \alpha_3={a3},\ \lambda={lam}$", fontsize=10)

for fig_name, (L, R) in {"A": ("A_left", "A_right"), "B": ("B_left", "B_right")}.items():
    fig = plt.figure(figsize=(12, 5.2))
    for j, key in enumerate((L, R), 1):
        ax = fig.add_subplot(1, 2, j, projection="3d"); surface(ax, CASES[key])
    fig.savefig(f"{OUT}/fig_hazard_{fig_name}.png", dpi=300, bbox_inches="tight")

# marginal hazard shapes for the four cases
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8)); rows = []
for j, (key, (a1, a2, a3, lam)) in enumerate(CASES.items()):
    k1, k2 = a1+a3, a2+a3; xmax = max(q_m(0.95, k1, lam), q_m(0.95, k2, lam)); x = np.linspace(xmax/300, xmax, 400)
    for k, lab, col in ((k1, r"$h_1$ (X1)", "C0"), (k2, r"$h_2$ (X2)", "C1")):
        hx = h_m(x, k, lam); ax[j].plot(x, hx, col, label=lab)
        d = np.diff(hx); shape = "decreasing" if np.all(d < 0) else "increasing" if np.all(d > 0) else "constant" if np.allclose(d, 0, atol=1e-9) else "non-monotone"
        rows.append(dict(case=key, margin=lab, k=k, shape=shape, h_at_small_x=hx[0], h_at_large_x=hx[-1]))
    ax[j].set_title(f"{key}: k1={k1:.2f}, k2={k2:.2f}", fontsize=10); ax[j].set_xlabel("x"); ax[j].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_hazard_marginal_shapes.png", dpi=300)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/table_hazard_shapes.csv", index=False)
print(T.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print("saved to", OUT)