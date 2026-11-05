import os, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "outputs/part10"; T = pd.read_csv(f"{OUT}/table_ARE_by_m.csv"); PN = ["alpha1", "alpha2", "alpha3", "lambda"]
SETS = list(T["set"].unique())
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a_, nm in zip(ax.ravel(), SETS):
    d = T[T["set"] == nm]
    for p in PN: a_.plot(d.m, d[f"ARE_{p}"], "o-", label=p)
    a_.plot(d.m, d.ARE_det, "ks--", lw=2, label="det-based"); a_.axhline(1, c="grey", lw=.6)
    a_.set_title(f"{nm}: asymptotic ARE vs set size m"); a_.set_xlabel("m"); a_.set_ylabel("ARE = Var_SRS / Var_RSS"); a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ARE_vs_m.png", dpi=300)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for nm in SETS: d = T[T["set"] == nm]; ax.plot(d.m, d.ARE_det, "o-", label=nm)
ax.axhline(1, c="grey", lw=.6); ax.set_xlabel("set size m"); ax.set_ylabel("determinant-based ARE"); ax.legend(); ax.set_title("Overall efficiency gain of RSS")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ARE_det.png", dpi=300); print("saved")
