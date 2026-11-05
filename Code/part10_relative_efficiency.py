import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bvge_rss as B

OUT = "outputs/part10"; os.makedirs(OUT, exist_ok=True)
PN = ["alpha1", "alpha2", "alpha3", "lambda"]
SETS = {"UEFA": (1.4452, 0.4681, 1.1704, 0.03899), "S1": (1.5, 1.0, 0.5, 1.0), "S2": (1.0, 1.0, 2.0, 1.0), "S3": (2.0, 0.7, 0.3, 0.5)}
MS = list(range(1, 11)); N_OBS = 2520*80          # divisible by every m = 1..10
rng = np.random.default_rng(10); pd.set_option("display.width", 220)

def info(th, m):
    d = B.sim_srs(N_OBS, th, rng) if m == 1 else B.sim_rss(m, N_OBS//m, th, rng)
    return B.hess(np.array(th), d, m)/N_OBS

rows = []
for nm, th in SETS.items():
    I1 = info(th, 1); V1 = np.linalg.inv(I1)           # SRS baseline
    for m in MS:
        I = info(th, m); V = np.linalg.inv(I); are = np.diag(V1)/np.diag(V); ad = (np.linalg.det(I)/np.linalg.det(I1))**0.25
        rows.append(dict(set=nm, m=m, **{f"ARE_{p}": a for p, a in zip(PN, are)}, ARE_det=ad))
    print(f"done {nm}", flush=True)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/table_ARE_by_m.csv", index=False)
for nm in SETS:
    print(f"\n=== ARE of RSS vs SRS (equal n), set {nm}, true = {SETS[nm]} (m=1 is an independent SRS replicate: noise check, ~1) ===")
    print(T[T.set == nm].drop(columns="set").to_string(index=False, float_format=lambda v: f"{v:.4f}"))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a_, nm in zip(ax.ravel(), SETS):
    d = T[T.set == nm]
    for p in PN: a_.plot(d.m, d[f"ARE_{p}"], "o-", label=p)
    a_.plot(d.m, d.ARE_det, "k s--", lw=2, label="det-based"); a_.axhline(1, c="grey", lw=.6); a_.set_title(f"{nm}: asymptotic ARE vs set size m")
    a_.set_xlabel("m"); a_.set_ylabel("ARE = Var_SRS / Var_RSS"); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ARE_vs_m.png", dpi=300)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for nm in SETS: d = T[T.set == nm]; ax.plot(d.m, d.ARE_det, "o-", label=nm)
ax.axhline(1, c="grey", lw=.6); ax.set_xlabel("set size m"); ax.set_ylabel("determinant-based ARE"); ax.legend(); ax.set_title("Overall efficiency gain of RSS")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ARE_det.png", dpi=300)
print("\nsaved to", OUT)