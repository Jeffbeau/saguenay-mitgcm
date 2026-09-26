#!/usr/bin/env python3
"""
Effet non hydrostatique au seuil : enfant A (hydrostatique) contre enfant B (non hydrostatique).

  1. Talweg : chemin le plus profond entre deux frontieres ouvertes de l'enfant (par defaut les
     deux segments d'OB les plus eloignes) ; seuil = point le moins profond du talweg.
  2. Coupes verticales le long du talweg a 8 phases du dernier cycle : w en couleur,
     isohalines en contours, une colonne par run.
  3. Chiffres pres du seuil (+- --rayon m le long du talweg) : w rms et |w| max par run,
     rapport B/A ; series sur le cycle.

Un seul run est accepte (coupes seules). Memoire : un champ 3D a la fois.

Usage
  python3 imbrication/seuil_nh.py RUN_A RUN_B [--noms A B] [--de X Y --a X Y] [--rayon 1000]
Ecrit RUN_B/diag/seuil_nh.txt, fig_seuil_coupes.png, fig_seuil_w.png (dans le dernier run donne).
"""
import argparse, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sagdiag"))
import sagdiag as sd

T_M2 = 44640.0


def segments_ob(wet):
    """Centres (j, i) des segments mouilles sur les 4 bords."""
    ny, nx = wet.shape
    out = []
    for side, line, pos in (("W", wet[:, 0], lambda k: (k, 0)), ("E", wet[:, -1], lambda k: (k, nx - 1)),
                            ("S", wet[0, :], lambda k: (0, k)), ("N", wet[-1, :], lambda k: (ny - 1, k))):
        lab, n = ndimage.label(line)
        for m in range(1, n + 1):
            idx = np.where(lab == m)[0]
            out.append((side, pos(int(idx[len(idx) // 2])), idx.size))
    return out


def talweg(H, a, b):
    """Chemin (liste de (j, i)) de a a b qui privilegie l'eau profonde."""
    from skimage.graph import route_through_array
    cost = np.where(H > 0, 1.0 / (H + 1.0) ** 2, np.inf)
    cost = np.where(np.isfinite(cost), cost / np.nanmin(cost[np.isfinite(cost)]), 1e12)
    path, _ = route_through_array(cost, a, b, fully_connected=True, geometric=True)
    return np.array(path)


def lire(run, prefix, it, noms):
    d, _ = sd.read_fields(os.path.join(run, f"{prefix}.{it:010d}"))
    return [np.asarray(d[n], np.float64) for n in noms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="dossiers de run (A puis B)")
    ap.add_argument("--noms", nargs="+", default=None)
    ap.add_argument("--de", nargs=2, type=float, help="debut du talweg (x y, m, repere du run)")
    ap.add_argument("--a", nargs=2, type=float, help="fin du talweg (x y, m)")
    ap.add_argument("--rayon", type=float, default=1000.0, help="demi-longueur de la zone du seuil (m)")
    ap.add_argument("--state3d", default=None)
    a = ap.parse_args()
    runs = a.runs
    noms = a.noms or (["A (hydrostatique)", "B (non hydrostatique)"][:len(runs)] if len(runs) == 2 else ["run"])

    g = sd.Grid(runs[0])
    H = g.Depth * g.maskC[0]
    for r in runs[1:]:
        assert sd.Grid(r).hFacC.shape == g.hFacC.shape, "les runs doivent avoir la meme grille"
    # --- talweg
    if a.de and a.a:
        ij = lambda xy: (int(np.argmin(np.abs(g.YC[:, 0] - xy[1]))), int(np.argmin(np.abs(g.XC[0] - xy[0]))))
        p0, p1 = ij(a.de), ij(a.a)
    else:
        segs = segments_ob(g.maskC[0])
        best = max(((s1, s2) for k, s1 in enumerate(segs) for s2 in segs[k + 1:]),
                   key=lambda p: math.dist(p[0][1], p[1][1]))
        p0, p1 = best[0][1], best[1][1]
    path = talweg(H, p0, p1)
    jj, ii = path[:, 0], path[:, 1]
    xs, ys = g.XC[jj, ii], g.YC[jj, ii]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    Hp = H[jj, ii]
    n = len(s); lo, hi = n // 10, n - n // 10
    ks = lo + int(np.argmin(Hp[lo:hi]))                         # seuil : col le long du talweg
    zone = np.abs(s - s[ks]) <= a.rayon
    zf = np.r_[0.0, np.cumsum(g.DRF)]
    se = np.r_[s[0] - (s[1] - s[0]) / 2, (s[1:] + s[:-1]) / 2, s[-1] + (s[-1] - s[-2]) / 2]   # bords

    # --- instantanes 3D du dernier cycle complet
    pre = a.state3d or sd.guess_prefixes(sd.scan_run(runs[0]), g.nz)[0]
    its = np.array(sd.list_iters(runs[0], pre)); t = its * g.dt
    dts = t[1] - t[0]                                  # la sortie finale peut manquer
    tfin = math.floor((t[-1] + dts) / T_M2 + 0.05) * T_M2   # cycle complet a 95 % accepte
    sel = (t >= tfin - T_M2 - 1e-6) & (t < tfin - 1e-6)
    its, t = its[sel], t[sel]
    phases = its[::max(1, len(its) // 8)][:8]

    res = {nm: dict(wrms=[], wmax=[]) for nm in noms}
    coupes = {nm: {} for nm in noms}
    msk = g.maskC[:, jj, ii]
    for run, nm in zip(runs, noms):
        for it in its:
            w, S = lire(run, pre, it, ("WVEL", "SALT"))
            wp, Sp = w[:, jj, ii], S[:, jj, ii]
            wz = np.where(msk[:, zone], wp[:, zone], np.nan)
            res[nm]["wrms"].append(float(np.sqrt(np.nanmean(wz ** 2))))
            res[nm]["wmax"].append(float(np.nanmax(np.abs(wz))))
            if it in phases:
                coupes[nm][it] = (np.where(msk, wp, np.nan), np.where(msk, Sp, np.nan))

    # --- figures
    od = os.path.join(runs[-1], "diag"); os.makedirs(od, exist_ok=True)
    wlim = max(np.nanpercentile(np.abs(np.concatenate([c[0].ravel() for nm in noms for c in coupes[nm].values()])), 99), 1e-4)
    fig, axs = plt.subplots(len(phases), len(runs), figsize=(7 * len(runs), 2.2 * len(phases)),
                            squeeze=False, sharex=True, sharey=True)
    for r, it in enumerate(phases):
        ph = ((it * g.dt) % T_M2) / T_M2 * 360
        for c, nm in enumerate(noms):
            wp, Sp = coupes[nm][it]
            ax = axs[r, c]
            pc = ax.pcolormesh(se / 1e3, -zf, wp, cmap="RdBu_r", vmin=-wlim, vmax=wlim, shading="flat")
            ax.contour(s / 1e3, -(zf[:-1] + zf[1:]) / 2, Sp, levels=np.arange(20, 34, 1.0), colors="k", linewidths=.5)
            ax.fill_between(s / 1e3, -Hp, -zf[-1], color="0.6")
            ax.axvline(s[ks] / 1e3, color="m", lw=.8, ls=":")
            ax.set_ylim(-min(zf[-1], 1.3 * Hp.max()), 0)
            ax.set_title(f"{nm} — phase M2 {ph:.0f}°", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("distance le long du talweg (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("z (m)")
    fig.colorbar(pc, ax=axs, label="w (m/s) ; contours : S", shrink=.6)
    fig.savefig(os.path.join(od, "fig_seuil_coupes.png"), dpi=90); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ph = ((t % T_M2) / T_M2) * 360
    for nm in noms:
        ax[0].plot(ph, np.array(res[nm]["wrms"]) * 1e3, label=nm)
        ax[1].plot(ph, np.array(res[nm]["wmax"]) * 1e3, label=nm)
    ax[0].set_ylabel("w rms au seuil (mm/s)"); ax[1].set_ylabel("|w| max au seuil (mm/s)")
    for x in ax:
        x.set_xlabel("phase M2 (°)"); x.legend()
    fig.tight_layout(); fig.savefig(os.path.join(od, "fig_seuil_w.png"), dpi=100); plt.close(fig)

    # --- resume
    lines = [f"Seuil : col {Hp[ks]:.1f} m a x={xs[ks]:.0f} y={ys[ks]:.0f} m (repere du run), "
             f"talweg {s[-1] / 1e3:.1f} km, zone +-{a.rayon:.0f} m ({zone.sum()} colonnes)",
             f"Dernier cycle : {t[0]:.0f}-{t[-1]:.0f} s, {len(its)} instantanes"]
    for nm in noms:
        lines.append(f"{nm:24s}: w rms moyen {1e3 * np.mean(res[nm]['wrms']):.2f} mm/s, "
                     f"|w| max {1e3 * np.max(res[nm]['wmax']):.1f} mm/s")
    if len(noms) == 2:
        r1 = np.mean(res[noms[1]]["wrms"]) / max(np.mean(res[noms[0]]["wrms"]), 1e-12)
        r2 = np.max(res[noms[1]]["wmax"]) / max(np.max(res[noms[0]]["wmax"]), 1e-12)
        lines.append(f"Rapport B/A : w rms {r1:.2f}, |w| max {r2:.2f}")
    lines.append(f"Figures : {od}/fig_seuil_coupes.png, fig_seuil_w.png")
    open(os.path.join(od, "seuil_nh.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
