import argparse
import itertools
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def native_contacts_q(traj, native):
    """Fraction of native heavy-atom contacts (Best-Hummer-Eaton definition)."""
    beta, lam, cutoff = 50.0, 1.8, 0.45          # 1/nm, -, nm
    heavy = native.topology.select_atom_indices("heavy")
    pairs = np.array([(i, j) for i, j in itertools.combinations(heavy, 2)
                      if abs(native.topology.atom(i).residue.index
                             - native.topology.atom(j).residue.index) > 3])
    r0 = md.compute_distances(native[0], pairs)[0]
    pairs, r0 = pairs[r0 < cutoff], r0[r0 < cutoff]
    r = md.compute_distances(traj, pairs)
    return np.mean(1.0 / (1 + np.exp(beta * (r - lam * r0))), axis=1)


def smooth(y, w=10):
    return pd.Series(y).rolling(w, center=True, min_periods=1).mean().values


def analyse_run(path):
    ref = md.load(os.path.join(path, "protein_start.pdb"))
    traj = md.load(os.path.join(path, "protein.xtc"), top=ref)
    ca = traj.topology.select("name CA")
    traj.superpose(ref, atom_indices=ca)
    dssp = md.compute_dssp(traj, simplified=True)
    df = pd.DataFrame({
        "time_ns": traj.time / 1000.0,
        "rmsd_nm": md.rmsd(traj, ref, atom_indices=ca),
        "rg_nm": md.compute_rg(traj),
        "Q": native_contacts_q(traj, ref),
        "helix_frac": (dssp == "H").mean(axis=1),
    })
    # RMSF around the average structure
    xyz = traj.xyz[:, ca, :]
    rmsf = np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))
    residues = [traj.topology.atom(i).residue for i in ca]
    rmsf_df = pd.DataFrame({"resid": [r.resSeq for r in residues],
                            "resname": [r.name for r in residues],
                            "rmsf_nm": rmsf})
    pair_idx = np.array(list(itertools.combinations(ca, 2)))
    dists = md.compute_distances(traj, pair_idx)
    return df, rmsf_df, dists


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--labels", nargs="+", help="legend labels (default: dir names)")
    ap.add_argument("--out", default="results", help="output directory")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    os.makedirs(args.out, exist_ok=True)

    results = {lab: analyse_run(run) for lab, run in zip(labels, args.runs)}

    # ---- time series ---------------------------------------------------------
    panels = [("rmsd_nm", "Cα RMSD (nm)"), ("rg_nm", "Radius of gyration (nm)"),
              ("Q", "Native contacts Q"), ("helix_frac", "Helical fraction")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, (col, ylabel) in zip(axes.flat, panels):
        for lab, (df, _, _) in results.items():
            line, = ax.plot(df.time_ns, df[col], alpha=0.25, lw=0.8)
            ax.plot(df.time_ns, smooth(df[col]), color=line.get_color(), lw=1.8, label=lab)
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("Time (ns)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "timeseries.png"), dpi=200)

    # ---- RMSF ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for lab, (_, rmsf_df, _) in results.items():
        ax.plot(rmsf_df.resid, rmsf_df.rmsf_nm, marker="o", ms=4, label=lab)
    first = next(iter(results.values()))[1]
    ax.set_xticks(first.resid)
    ax.set_xticklabels([f"{n}{i}" for n, i in zip(first.resname, first.resid)],
                       rotation=90, fontsize=8)
    ax.set_ylabel("Cα RMSF (nm)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rmsf.png"), dpi=200)

    # ---- PCA on C-alpha distance features (all runs together) ---------------
    X = np.vstack([d for (_, _, d) in results.values()])
    pcs = PCA(n_components=2).fit(X)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab, (_, _, d) in results.items():
        p = pcs.transform(d)
        ax.scatter(p[:, 0], p[:, 1], s=6, alpha=0.5, label=lab)
    ev = pcs.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
    ax.legend(frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pca.png"), dpi=200)

    # ---- summary table (second half of each run = "equilibrated") ----------
    rows = []
    for lab, (df, _, _) in results.items():
        tail = df[df.time_ns >= df.time_ns.max() / 2]
        row = {"run": lab, "frames": len(df), "length_ns": round(df.time_ns.max(), 2)}
        for col, _ in panels:
            row[f"{col}_mean"] = tail[col].mean()
            row[f"{col}_std"] = tail[col].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"\nFigures and summary.csv written to {args.out}/")


if __name__ == "__main__":
    main()
