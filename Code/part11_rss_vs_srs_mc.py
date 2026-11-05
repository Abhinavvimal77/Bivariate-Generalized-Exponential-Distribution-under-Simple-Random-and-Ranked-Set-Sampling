import os, sys, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import bvge_rss as B

OUT = "outputs/part11"; R, NS, MS, SEED = 1000, (30, 50, 70, 100), (2, 5, 10), 2027
SETS = {"UEFA": (1.4452, 0.4681, 1.1704, 0.03899), "S1": (1.5, 1.0, 0.5, 1.0), "S2": (1.0, 1.0, 2.0, 1.0), "S3": (2.0, 0.7, 0.3, 0.5)}
PN = ["alpha1", "alpha2", "alpha3", "lambda"]; Z = 1.959964

def chunk(args):
    nm, m, n, seed, reps = args; th = np.array(SETS[nm]); rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        d = B.sim_rss(m, n//m, th, rng); t, ll, hit = B.fit(d, m); row = np.full(26, np.nan); row[:4] = t; row[24] = hit; row[25] = 0
        if not hit:
            try:
                H = B.hess(t, d, m)
                if np.all(np.linalg.eigvalsh(H) > 0):
                    se = np.sqrt(np.diag(np.linalg.inv(H))); row[4:8] = se; row[8:12] = t-Z*se; row[12:16] = t+Z*se
                    row[16:20] = t*np.exp(-Z*se/t); row[20:24] = t*np.exp(Z*se/t); row[25] = 1
            except np.linalg.LinAlgError: pass
        out.append(row)
    return np.array(out)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1: R = int(sys.argv[1])
    srs = pd.read_csv("outputs/part7/mc_raw.csv"); nw = max(1, (os.cpu_count() or 2)-1); ss = np.random.SeedSequence(SEED); raw = []; t0 = time.time()
    cols = [f"est_{p}" for p in PN] + [f"se_{p}" for p in PN] + [f"wlo_{p}" for p in PN] + [f"whi_{p}" for p in PN] + [f"llo_{p}" for p in PN] + [f"lhi_{p}" for p in PN] + ["bound_hit", "ci_ok"]
    with ProcessPoolExecutor(nw) as ex:
        for nm in SETS:
            for m in MS:
                for n in NS:
                    parts = [R//nw + (1 if i < R % nw else 0) for i in range(nw)]; seeds = ss.spawn(nw)
                    A = np.vstack(list(ex.map(chunk, [(nm, m, n, s, p) for s, p in zip(seeds, parts) if p > 0])))
                    df = pd.DataFrame(A, columns=cols); df.insert(0, "n", n); df.insert(0, "m", m); df.insert(0, "set", nm); raw.append(df)
                print(f"done {nm} m={m}  ({time.time()-t0:.0f}s)", flush=True)
    raw = pd.concat(raw, ignore_index=True); raw.to_csv(f"{OUT}/mc_raw_rss.csv", index=False)

    rows, jrows = [], []
    for (nm, m, n), g in raw.groupby(["set", "m", "n"], sort=False):
        th = SETS[nm]; S = srs[(srs.set == nm) & (srs.n == n) & (srs.bound_hit == 0)]; G = g[g.bound_hit == 0]; ck = g[g.ci_ok == 1]
        ES, ER = S[[f"est_{p}" for p in PN]].values, G[[f"est_{p}" for p in PN]].values
        for i, p in enumerate(PN):
            ms, mr = np.mean((ES[:, i]-th[i])**2), np.mean((ER[:, i]-th[i])**2)
            rows.append(dict(set=nm, m=m, n=n, param=p, true=th[i], bias_SRS=ES[:, i].mean()-th[i], bias_RSS=ER[:, i].mean()-th[i],
                MSE_SRS=ms, MSE_RSS=mr, RE=ms/mr, cov_Wald_RSS=np.mean((ck[f"wlo_{p}"] <= th[i]) & (th[i] <= ck[f"whi_{p}"])),
                cov_logWald_RSS=np.mean((ck[f"llo_{p}"] <= th[i]) & (th[i] <= ck[f"lhi_{p}"])),
                AL_logWald_RSS=np.mean(ck[f"lhi_{p}"]-ck[f"llo_{p}"]), bound_hit_pct_RSS=100*g.bound_hit.mean()))
        jrows.append(dict(set=nm, m=m, n=n, RE_joint=(np.linalg.det(np.cov(ES.T))/np.linalg.det(np.cov(ER.T)))**0.25))
    T, J = pd.DataFrame(rows), pd.DataFrame(jrows); T.to_csv(f"{OUT}/table_rss_vs_srs.csv", index=False); J.to_csv(f"{OUT}/table_joint_RE.csv", index=False)
    pd.set_option("display.width", 250)
    for nm in SETS:
        print(f"\n=== set {nm}, true = {SETS[nm]}: RE = MSE_SRS / MSE_RSS ===")
        print(T[T.set == nm].pivot_table(index="n", columns=["param", "m"], values="RE").round(3).to_string())
        print("joint RE (det of estimator covariance, 4th root):\n", J[J.set == nm].pivot(index="n", columns="m", values="RE_joint").round(3).to_string())
        print("RSS log-Wald coverage:\n", T[T.set == nm].pivot_table(index="n", columns=["param", "m"], values="cov_logWald_RSS").round(3).to_string())
        print("detail:\n", T[(T.set == nm)].drop(columns="set").to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(SETS), 4, figsize=(16, 3.3*len(SETS)))
    for r_, nm in enumerate(SETS):
        for i, p in enumerate(PN):
            for m in MS:
                d = T[(T.set == nm) & (T.param == p) & (T.m == m)]; ax[r_, i].plot(d.n, d.RE, "o-", label=f"m={m}")
            ax[r_, i].axhline(1, c="grey", lw=.6); ax[r_, i].set_title(f"{nm}: RE of {p}", fontsize=9); ax[r_, i].set_xlabel("n")
            if r_ == 0 and i == 0: ax[r_, i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_RE_vs_n.png", dpi=300)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for i, p in enumerate(PN):
        for nm in SETS:
            d = T[(T.set == nm) & (T.param == p) & (T.m == 5)]
            ax[i].plot(d.n, d.MSE_SRS/d.true**2, "o--", c=None, label=f"{nm} SRS"); ax[i].plot(d.n, d.MSE_RSS/d.true**2, "s-", label=f"{nm} RSS")
        ax[i].set_yscale("log"); ax[i].set_title(f"{p}: relative MSE, SRS vs RSS (m=5)", fontsize=9); ax[i].set_xlabel("n")
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/fig_mse_srs_vs_rss.png", dpi=300)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for nm in SETS:
        d = J[(J.set == nm) & (J.m == 5)]; ax.plot(d.n, d.RE_joint, "o-", label=nm)
    ax.axhline(1, c="grey", lw=.6); ax.set_xlabel("n"); ax.set_ylabel("joint RE"); ax.set_title("Joint relative efficiency, RSS (m=5) vs SRS"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_joint_RE.png", dpi=300)
    print("\nsaved to", OUT)