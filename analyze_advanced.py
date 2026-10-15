import argparse
import itertools
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyze import native_contacts_q


def native_contacts_q_excluding(traj, native, exclude_resseq):
    """Q (Best-Hummer-Eaton) computed without native contacts that involve the given residues."""
    beta, lam, cutoff = 50.0, 1.8, 0.45
    top = native.topology
    heavy = top.select_atom_indices("heavy")
    pairs = np.array([(i, j) for i, j in itertools.combinations(heavy, 2)
                      if abs(top.atom(i).residue.index - top.atom(j).residue.index) > 3
                      and top.atom(i).residue.resSeq not in exclude_resseq
                      and top.atom(j).residue.resSeq not in exclude_resseq])
    r0 = md.compute_distances(native[0], pairs)[0]
    pairs, r0 = pairs[r0 < cutoff], r0[r0 < cutoff]
    r = md.compute_distances(traj, pairs)
    return np.mean(1.0 / (1 + np.exp(beta * (r - lam * r0))), axis=1)

KB = 0.0083144626                       # Boltzmann constant, kJ mol^-1 K^-1
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]   # fixed order per group
CONTACT_CUTOFF = 0.45                   # nm, closest heavy-atom residue contact
METRICS = {"rmsd_nm": "Cα RMSD (nm)", "rg_nm": "R$_g$ (nm)", "Q": "Q",
           "helix_frac": "Helical fraction", "trp6_sasa_nm2": "Trp6 SASA (nm²)",
           "sb_d9_r16_nm": "Asp9–Arg16 distance (nm)"}
SALT_BRIDGE_CUTOFF = 0.40               # nm, carboxylate O - guanidinium N

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})


# --------------------------------------------------------------------------- #
# Per-run features
# --------------------------------------------------------------------------- #
def residue_label(res):
    return f"{res.code}{res.resSeq}"


def load_run(path, skip_ns):
    ref = md.load(os.path.join(path, "protein_start.pdb"))
    traj = md.load(os.path.join(path, "protein.xtc"), top=ref)
    t_ns = (traj.time - traj.time[0]) / 1000.0
    traj = traj[t_ns >= skip_ns]
    t_ns = t_ns[t_ns >= skip_ns]

    ca = traj.topology.select("name CA")
    traj.superpose(ref, atom_indices=ca)
    dssp = md.compute_dssp(traj, simplified=True)       # 'H' = alpha, 3-10, pi helix
    sasa = md.shrake_rupley(traj, mode="residue", n_sphere_points=240)
    trp = [r.index for r in traj.topology.residues if r.name == "TRP"][0]
    cdist, cpairs = md.compute_contacts(traj, contacts="all", scheme="closest-heavy")
    pair_idx = np.array(list(itertools.combinations(ca, 2)))

    # Asp9-Arg16 salt bridge: minimum O(Asp9 side chain) - N(Arg16 side chain) distance
    o = traj.topology.select("resSeq 9 and resname ASP and name OD1 OD2")
    n = traj.topology.select("resSeq 16 and resname ARG and name NE NH1 NH2")
    sb = md.compute_distances(traj, np.array([(i, j) for i in o for j in n])).min(axis=1)

    xyz = traj.xyz[:, ca, :]
    rmsf = np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))

    df = pd.DataFrame({
        "time_ns": t_ns,
        "rmsd_nm": md.rmsd(traj, ref, atom_indices=ca),
        "rg_nm": md.compute_rg(traj),
        "Q": native_contacts_q(traj, ref),
        "helix_frac": (dssp == "H").mean(axis=1),
        "trp6_sasa_nm2": sasa[:, trp],
        "sb_d9_r16_nm": sb,
        "Q_excl_d9_r16": native_contacts_q_excluding(traj, ref, {9, 16}),
    })
    return {"path": path, "traj": traj, "df": df, "rmsf": rmsf,
            "ca_dist": md.compute_distances(traj, pair_idx),
            "ca_pairs": pair_idx, "contacts": cdist < CONTACT_CUTOFF,
            "contact_pairs": cpairs, "ca": ca}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def block_means(x, n_blocks):
    n = len(x) // n_blocks
    return np.array([x[i * n:(i + 1) * n].mean() for i in range(n_blocks)])


def block_sem(x, n_blocks):
    b = block_means(x, n_blocks)
    return b.std(ddof=1) / np.sqrt(n_blocks)


def sem_vs_block_length(x, dt_ns):
    """Standard error of the mean as a function of block length (Flyvbjerg-Petersen style)."""
    lengths, sems = [], []
    for nb in range(len(x) // 4, 3, -1):
        size = len(x) // nb
        if lengths and size == lengths[-1]:
            continue
        lengths.append(size)
        sems.append(block_sem(x, nb))
    return np.array(lengths) * dt_ns, np.array(sems)


def group_summary(runs, n_blocks):
    """Mean, SEM and s.d. per metric. SEM from replicas if >= 2, else from blocks."""
    out = {}
    for m in METRICS:
        per_run = [r["df"][m].values for r in runs]
        pooled = np.concatenate(per_run)
        if len(runs) >= 2:
            means = np.array([x.mean() for x in per_run])
            sem = means.std(ddof=1) / np.sqrt(len(means))
        else:
            sem = block_sem(per_run[0], n_blocks)
        out[m] = {"mean": pooled.mean(), "sem": sem, "sd": pooled.std(ddof=1)}
    q = np.concatenate([r["df"]["Q"].values for r in runs])
    sbf = [(r["df"]["sb_d9_r16_nm"].values < SALT_BRIDGE_CUTOFF).astype(float) for r in runs]
    out["salt_bridge_formed"] = {"mean": float(np.concatenate(sbf).mean()), "sem": float(
        np.std(np.concatenate([block_means(x, n_blocks) for x in sbf]), ddof=1)
        / np.sqrt(n_blocks * len(runs)))}
    out["p_Q_below_0.9"] = {"mean": float((q < 0.9).mean()), "sem": float(
        np.std([block_means((r["df"]["Q"].values < 0.9).astype(float), n_blocks)
                for r in runs], ddof=1) / np.sqrt(n_blocks * len(runs)))}
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def free_energy(x, y, T, bins, xr, yr):
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr], density=True)
    with np.errstate(divide="ignore"):
        F = -KB * T * np.log(H)
    F -= np.nanmin(F[np.isfinite(F)])
    F[~np.isfinite(F)] = np.nan
    return F.T, xe, ye


def plot_fes(groups, xkey, ykey, xlabel, ylabel, fname, out, data=None, fmax=12.0):
    xs = {g: (data[g][0] if data else np.concatenate([r["df"][xkey] for r in runs]))
          for g, runs in groups.items()}
    ys = {g: (data[g][1] if data else np.concatenate([r["df"][ykey] for r in runs]))
          for g, runs in groups.items()}
    allx, ally = np.concatenate(list(xs.values())), np.concatenate(list(ys.values()))
    pad_x, pad_y = 0.02 * np.ptp(allx), 0.02 * np.ptp(ally)
    xr = (allx.min() - pad_x, allx.max() + pad_x)
    yr = (ally.min() - pad_y, ally.max() + pad_y)

    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 0.8, 3.4),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    levels = np.linspace(0, fmax, 13)
    for ax, (T, runs) in zip(axes, groups.items()):
        F, xe, ye = free_energy(xs[T], ys[T], T, 30, xr, yr)
        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
        cf = ax.contourf(xc, yc, np.clip(F, 0, fmax), levels=levels, cmap="Blues_r",
                         extend="max")
        ax.contour(xc, yc, F, levels=levels[::3], colors="k", linewidths=0.4)
        ax.set_title(f"{T:.0f} K")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(cf, ax=axes, label="F (kJ mol$^{-1}$)", shrink=0.9)
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", action="append", required=True,
                    metavar=("TEMP_K", "RUN_DIR"),
                    help="temperature in K followed by one or more run directories")
    ap.add_argument("--nmr", default="1L2Y.pdb", help="experimental NMR ensemble")
    ap.add_argument("--skip_ns", type=float, default=2.0,
                    help="initial relaxation discarded from every run (ns)")
    ap.add_argument("--blocks", type=int, default=10, help="blocks for error estimates")
    ap.add_argument("--q_excursion", type=float, default=0.92,
                    help="Q threshold that defines a non-native excursion")
    ap.add_argument("--out", default="results_advanced")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    groups = {}
    for spec in args.group:
        T = float(spec[0])
        groups[T] = [load_run(p, args.skip_ns) for p in spec[1:]]
        print(f"{T:.0f} K: {len(spec) - 1} run(s), "
              f"{sum(len(r['df']) for r in groups[T])} frames after {args.skip_ns} ns skip")
    temps = list(groups)
    color = {T: COLORS[i] for i, T in enumerate(temps)}
    first = groups[temps[0]][0]
    top = first["traj"].topology
    labels = [residue_label(r) for r in top.residues]
    report = {"skip_ns": args.skip_ns}

    # ---- 5. Block-averaged summary + Welch t-test --------------------------
    summ = {T: group_summary(runs, args.blocks) for T, runs in groups.items()}
    rows = []
    for T in temps:
        row = {"T_K": T, "n_runs": len(groups[T]),
               "ns_analysed": round(sum(np.ptp(r["df"].time_ns.values) for r in groups[T]), 1)}
        for m, v in summ[T].items():
            row.update({f"{m}_mean": v["mean"], f"{m}_sem": v["sem"]})
            if "sd" in v:
                row[f"{m}_sd"] = v["sd"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary_advanced.csv"), index=False)
    report["summary"] = {f"{T:.0f}": summ[T] for T in temps}

    if len(temps) >= 2:
        a, b = temps[0], temps[1]
        tests = {}
        for m in METRICS:
            ba = np.concatenate([block_means(r["df"][m].values, args.blocks) for r in groups[a]])
            bb = np.concatenate([block_means(r["df"][m].values, args.blocks) for r in groups[b]])
            t, p = stats.ttest_ind(bb, ba, equal_var=False)
            tests[m] = {"diff": float(bb.mean() - ba.mean()), "p": float(p)}
        report["welch_block_means"] = tests

    # block-length convergence for Q and RMSD
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, m in zip(axes, ["Q", "rmsd_nm"]):
        for T in temps:
            for k, r in enumerate(groups[T]):
                dt = np.median(np.diff(r["df"].time_ns))
                L, s = sem_vs_block_length(r["df"][m].values, dt)
                ax.plot(L, s, color=color[T], lw=1.5, alpha=1 if k == 0 else 0.5,
                        label=f"{T:.0f} K" if k == 0 else None)
        ax.set_xlabel("Block length (ns)")
        ax.set_ylabel(f"SEM of {METRICS[m]}")
    axes[0].legend(frameon=False)
    fig.savefig(os.path.join(args.out, "block_convergence.png"))
    plt.close(fig)

    # ---- 1. Free-energy landscapes ------------------------------------------
    plot_fes(groups, "Q", "rg_nm", "Native contacts Q", "R$_g$ (nm)", "fes_Q_Rg.png", args.out)
    X_all = np.vstack([r["ca_dist"] for T in temps for r in groups[T]])
    pca = PCA(n_components=2).fit(X_all)
    pc = {T: np.vstack([pca.transform(r["ca_dist"]) for r in groups[T]]) for T in temps}
    ev = pca.explained_variance_ratio_ * 100
    plot_fes(groups, None, None, f"PC1 ({ev[0]:.0f}%)", f"PC2 ({ev[1]:.0f}%)",
             "fes_pca.png", args.out, data={T: (pc[T][:, 0], pc[T][:, 1]) for T in temps})
    report["pca_explained_pct"] = ev.round(1).tolist()

    # ---- 2. Validation against the NMR ensemble ----------------------------
    if os.path.exists(args.nmr):
        nmr = md.load(args.nmr)
        nmr_ca = nmr.atom_slice(nmr.topology.select("name CA"))
        nmr_ca.superpose(nmr_ca, 0)
        nmr_rmsd = md.rmsd(nmr_ca, nmr_ca, 0)[1:]                 # models 2-38 vs model 1
        mean_xyz = nmr_ca.xyz.mean(axis=0)
        nmr_rmsf = np.sqrt(((nmr_ca.xyz - mean_xyz) ** 2).sum(axis=2).mean(axis=0))

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
        x = np.arange(len(labels))
        axes[0].plot(x, nmr_rmsf, "k--", marker="s", ms=3, lw=1.2, label=f"NMR ensemble ({nmr.n_frames} models)")
        val = {"nmr_rmsd_to_model1_nm": {"min": float(nmr_rmsd.min()),
                                         "median": float(np.median(nmr_rmsd)),
                                         "max": float(nmr_rmsd.max())}}
        for T in temps:
            md_rmsf = np.mean([r["rmsf"] for r in groups[T]], axis=0)
            r_corr = stats.pearsonr(md_rmsf, nmr_rmsf)[0]
            axes[0].plot(x, md_rmsf, marker="o", ms=3, lw=1.5, color=color[T],
                         label=f"MD {T:.0f} K (r = {r_corr:.2f})")
            traj_ca = np.vstack([r["traj"].atom_slice(r["ca"]).xyz for r in groups[T]])
            md_ca = md.Trajectory(traj_ca, nmr_ca.topology)
            md_rmsd = md.rmsd(md_ca, nmr_ca, 0)
            axes[1].hist(md_rmsd, bins=40, density=True, histtype="step", lw=1.5,
                         color=color[T], label=f"MD {T:.0f} K")
            val[f"{T:.0f}"] = {"rmsf_pearson_r": float(r_corr),
                               "md_rmsd_to_model1_median_nm": float(np.median(md_rmsd)),
                               "frac_within_nmr_range": float((md_rmsd <= nmr_rmsd.max()).mean())}
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=7)
        axes[0].set_ylabel("Cα RMSF (nm)")
        axes[0].legend(frameon=False, fontsize=7)
        ymax = axes[1].get_ylim()[1]
        axes[1].plot(nmr_rmsd, np.full_like(nmr_rmsd, ymax * 0.03), "k|", ms=10,
                     label="NMR models vs model 1")
        axes[1].set_xlabel("Cα RMSD to NMR model 1 (nm)")
        axes[1].set_ylabel("Probability density")
        axes[1].legend(frameon=False, fontsize=7)
        fig.savefig(os.path.join(args.out, "nmr_validation.png"))
        plt.close(fig)
        report["nmr_validation"] = val
    else:
        print(f"NMR file {args.nmr} not found - skipping validation")

    # ---- 3. Interpretable classifier ---------------------------------------
    if len(temps) == 2:
        runs_flat = [(i, k, r) for i, T in enumerate(temps) for k, r in enumerate(groups[T])]
        X = np.vstack([r["ca_dist"] for _, _, r in runs_flat])
        y = np.concatenate([np.full(len(r["df"]), i) for i, _, r in runs_flat])
        n_rep = min(len(groups[T]) for T in temps)
        if n_rep >= 2:          # leave one replica of each temperature out
            fold = np.concatenate([np.full(len(r["df"]), k % n_rep) for _, k, r in runs_flat])
            n_folds, cv_kind = n_rep, "leave-one-replica-out"
        else:                   # contiguous time blocks (avoids leaking correlated frames)
            n_folds, cv_kind = 5, "5 contiguous time blocks per run"
            fold = np.concatenate([np.minimum(np.arange(len(r["df"])) * 5 // len(r["df"]), 4)
                                   for _, _, r in runs_flat])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
        accs = []
        for k in range(n_folds):
            tr, te = fold != k, fold == k
            model.fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], model.predict(X[te])))
        model.fit(X, y)
        coef = model[-1].coef_[0]
        order = np.argsort(-np.abs(coef))[:12]
        pair_res = [(top.atom(i).residue, top.atom(j).residue) for i, j in first["ca_pairs"]]
        names = [f"{residue_label(a)}–{residue_label(b)}" for a, b in pair_res]
        d_mean = {i: X[y == i].mean(axis=0) for i in (0, 1)}

        fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
        yy = np.arange(len(order))[::-1]
        ax.barh(yy, coef[order], color=[color[temps[1]] if c > 0 else color[temps[0]]
                                        for c in coef[order]], height=0.6)
        ax.set_yticks(yy)
        ax.set_yticklabels([names[i] for i in order])
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_xlabel(f"Standardized coefficient  (+ = longer at {temps[1]:.0f} K)")
        ax.set_title(f"Balanced accuracy {np.mean(accs):.2f} ± {np.std(accs):.2f} ({cv_kind})",
                     fontsize=8)
        fig.savefig(os.path.join(args.out, "classifier_features.png"))
        plt.close(fig)
        report["classifier"] = {
            "cv": cv_kind, "balanced_accuracy_folds": [round(a, 3) for a in accs],
            "balanced_accuracy_mean": float(np.mean(accs)),
            "top_features": [{"pair": names[i], "coef": float(coef[i]),
                              f"mean_dist_{temps[0]:.0f}K_nm": float(d_mean[0][i]),
                              f"mean_dist_{temps[1]:.0f}K_nm": float(d_mean[1][i])}
                             for i in order[:6]]}

    # ---- 4. Trp6 SASA, contact maps, excursion contacts ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2, 1]})
    for T in temps:
        for k, r in enumerate(groups[T]):
            s = pd.Series(r["df"].trp6_sasa_nm2).rolling(20, center=True, min_periods=1).mean()
            axes[0].plot(r["df"].time_ns, s, color=color[T], lw=1,
                         alpha=1 if k == 0 else 0.5, label=f"{T:.0f} K" if k == 0 else None)
        allv = np.concatenate([r["df"].trp6_sasa_nm2 for r in groups[T]])
        axes[1].hist(allv, bins=40, density=True, histtype="step", lw=1.5, color=color[T],
                     label=f"{T:.0f} K")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Trp6 SASA (nm²)")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Trp6 SASA (nm²)")
    axes[1].set_ylabel("Probability density")
    fig.savefig(os.path.join(args.out, "trp6_sasa.png"))
    plt.close(fig)

    n_res = top.n_residues
    cpairs = first["contact_pairs"]
    freq = {}
    for T in temps:
        f = np.concatenate([r["contacts"] for r in groups[T]]).mean(axis=0)
        M = np.full((n_res, n_res), np.nan)
        M[cpairs[:, 0], cpairs[:, 1]] = f
        M[cpairs[:, 1], cpairs[:, 0]] = f
        freq[T] = M
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    im0 = axes[0].imshow(freq[temps[0]], cmap="Blues", vmin=0, vmax=1, origin="lower")
    axes[0].set_title(f"Contact frequency, {temps[0]:.0f} K")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    if len(temps) >= 2:
        D = freq[temps[1]] - freq[temps[0]]
        lim = max(0.05, np.nanmax(np.abs(D)))
        im1 = axes[1].imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        axes[1].set_title(f"Difference, {temps[1]:.0f} K − {temps[0]:.0f} K")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_xticks(range(n_res))
        ax.set_yticks(range(n_res))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    fig.savefig(os.path.join(args.out, "contact_maps.png"))
    plt.close(fig)

    pair_names = [f"{labels[i]}–{labels[j]}" for i, j in cpairs]
    if len(temps) >= 2:
        D = freq[temps[1]][cpairs[:, 0], cpairs[:, 1]] - freq[temps[0]][cpairs[:, 0], cpairs[:, 1]]
        idx = np.argsort(D)[:5]
        report["largest_contact_loss_at_high_T"] = [
            {"pair": pair_names[i], "freq_low_T": float(freq[temps[0]][tuple(cpairs[i])]),
             "freq_high_T": float(freq[temps[1]][tuple(cpairs[i])])} for i in idx]

    exc = {}
    for T in temps:
        q = np.concatenate([r["df"].Q.values for r in groups[T]])
        C = np.concatenate([r["contacts"] for r in groups[T]])
        low = q < args.q_excursion
        if low.sum() >= 5:
            drop = C[~low].mean(axis=0) - C[low].mean(axis=0)
            idx = np.argsort(-drop)[:5]
            # excursion events: contiguous low-Q stretches per run
            events = []
            for r in groups[T]:
                lo = (r["df"].Q.values < args.q_excursion).astype(int)
                starts = np.flatnonzero(np.diff(np.r_[0, lo]) == 1)
                ends = np.flatnonzero(np.diff(np.r_[lo, 0]) == -1)
                events += [{"run": r["path"], "start_ns": float(r["df"].time_ns.iloc[s]),
                            "end_ns": float(r["df"].time_ns.iloc[e]),
                            "min_Q": float(r["df"].Q.values[s:e + 1].min())}
                           for s, e in zip(starts, ends) if e - s >= 4]
            exc[f"{T:.0f}"] = {
                "fraction_frames": float(low.mean()),
                "events_ge_5_frames": sorted(events, key=lambda e: e["min_Q"])[:6],
                "contacts_broken": [{"pair": pair_names[i],
                                     "freq_native_like": float(C[~low, i].mean()),
                                     "freq_excursion": float(C[low, i].mean())} for i in idx]}
        else:
            exc[f"{T:.0f}"] = {"fraction_frames": float(low.mean())}
    report["excursions_Q_below"] = {"threshold": args.q_excursion, **exc}

    # Asp9-Arg16 salt bridge vs native contacts
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True, constrained_layout=True)
    sb_rep = {}
    for T in temps:
        for k, r in enumerate(groups[T]):
            d = r["df"]
            sm = lambda v: pd.Series(v).rolling(10, center=True, min_periods=1).mean()
            kw = dict(color=color[T], lw=1, alpha=1 if k == 0 else 0.5,
                      label=f"{T:.0f} K" if k == 0 else None)
            axes[0].plot(d.time_ns, sm(d.Q), **kw)
            axes[1].plot(d.time_ns, sm(d.Q_excl_d9_r16), **kw)
            axes[2].plot(d.time_ns, sm(d.sb_d9_r16_nm), **kw)
        q = np.concatenate([r["df"].Q.values for r in groups[T]])
        sbd = np.concatenate([r["df"].sb_d9_r16_nm.values for r in groups[T]])
        low = q < args.q_excursion
        broken = sbd >= SALT_BRIDGE_CUTOFF
        qx = np.concatenate([r["df"].Q_excl_d9_r16.values for r in groups[T]])
        sb_rep[f"{T:.0f}"] = {
            "corr_Qexcl_vs_distance": float(stats.pearsonr(qx, sbd)[0]),
            "frac_Qexcl_below_threshold": float((qx < args.q_excursion).mean()),
            "Qexcl_mean": float(qx.mean()),
            "p_broken": float(broken.mean()),
            "p_broken_given_excursion": float(broken[low].mean()) if low.any() else None,
            "p_broken_given_native": float(broken[~low].mean()),
            "corr_Q_vs_distance": float(stats.pearsonr(q, sbd)[0])}
    axes[0].axhline(args.q_excursion, color="0.5", ls=":", lw=1)
    axes[0].set_ylabel("Q")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(args.q_excursion, color="0.5", ls=":", lw=1)
    axes[1].set_ylabel("Q without\nAsp9/Arg16 contacts")
    axes[1].set_ylim(axes[0].get_ylim())
    axes[2].axhline(SALT_BRIDGE_CUTOFF, color="0.5", ls=":", lw=1)
    axes[2].set_ylabel("Asp9–Arg16 distance (nm)")
    axes[2].set_xlabel("Time (ns)")
    fig.savefig(os.path.join(args.out, "salt_bridge.png"))
    plt.close(fig)
    report["salt_bridge_D9_R16"] = {"cutoff_nm": SALT_BRIDGE_CUTOFF, **sb_rep}

    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    # ---- console report -----------------------------------------------------
    pd.set_option("display.width", 200)
    plain = {**METRICS, "rg_nm": "Rg (nm)", "salt_bridge_formed": "Salt bridge formed (fraction)",
             "p_Q_below_0.9": "P(Q < 0.9)"}
    tab = pd.DataFrame({f"{T:.0f} K": {plain.get(m, m): f"{v['mean']:.3f} ± {v['sem']:.3f}"
                                       for m, v in summ[T].items()} for T in temps})
    print("\nMean ± SEM (", "replicas" if min(len(g) for g in groups.values()) >= 2
          else f"{args.blocks} blocks", ")\n", tab.to_string(), sep="")
    print("\nFull report:", os.path.join(args.out, "report.json"))


if __name__ == "__main__":
    main()
