#!/usr/bin/env python3
"""
Seuil et tourbillon de pointe : enfant A (hydrostatique) contre enfant B (non hydrostatique).

  1. Talweg : chemin le plus profond entre deux frontieres ouvertes de l'enfant (par defaut les
     deux segments d'OB les plus eloignes) ; seuil = point le moins profond du talweg.
  2. Coupes verticales le long du talweg a 8 phases du dernier cycle : w en couleur,
     isohalines en contours, une colonne par run.
  3. Chiffres pres du seuil (+- --rayon m le long du talweg) : w rms et |w| max par run,
     rapport B/A ; series sur le cycle.
  4. Nombre de Froude interne au col : |U moyen| / c1, c1 = (1/pi) int N dz (mode 1, WKB ;
     N depuis RHOAnoma) ; Fr > 1 : l'onde interne ne remonte pas le courant (supercritique).
  5. Tourbillon de pointe (niveau --niveau de la liste 2D, ~10 m par defaut) : pointe = maximum
     de |zeta| moyen sur le cycle hors de la zone du col (ou --pointe x y) ; dans un disque de
     --rayon-pointe : coeur Okubo-Weiss (W < -0,2 sigma_W, |Ro| > 0,2), circulation, aire,
     |Ro| max ; cartes zeta/f a 8 phases ; couplage (dephasage de correlation max) entre le
     Froude au col et la circulation du tourbillon.

Un seul run est accepte (coupes seules). Memoire : un champ 3D a la fois.

Usage
  python3 imbrication/seuil_nh.py RUN_A RUN_B [--noms A B] [--de X Y --a X Y] [--rayon 1000]
Ecrit dans RUN_B/diag (dernier run donne) : seuil_nh.txt, fig_seuil_coupes.png, fig_seuil_w.png,
fig_pointe_vorticite.png, fig_seuil_pointe.png.
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


def froude(u3, v3, rho, g, j, i, tx, ty):
    """Froude interne au col : |U moyen le long du talweg| / c1, c1 = (1/pi) int N dz
    (onde interne du mode 1, approximation WKB). Renvoie Fr, U moyen, c1, U surface."""
    k = np.where(g.maskC[:, j, i])[0]
    if k.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    U = 0.5 * (u3[k, j, i] + u3[k, j, i + 1]) * tx + 0.5 * (v3[k, j, i] + v3[k, j + 1, i]) * ty
    dz = g.DRF[k] * g.hFacC[k, j, i]
    zc = -g.RC[k]
    N2 = np.clip(sd.G / g.rho0 * np.diff(rho[k, j, i]) / np.diff(zc), 0, None)
    c1 = float((np.sqrt(N2) * np.diff(zc)).sum() / math.pi)
    Um = float((U * dz).sum() / dz.sum())
    return abs(Um) / max(c1, 1e-3), Um, c1, float(U[0])


def lonlat_vers_run(lon, lat, run, pipeline):
    """lon/lat -> x, y (m) dans le repere du run enfant : UTM 19N - origine du parent (grids.npz)
    - coin de l'enfant dans le parent (enfant.json, a cote du dossier run)."""
    import json
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True).transform(lon, lat)
    gr = np.load(os.path.join(pipeline, "grids.npz"))
    c = json.load(open(os.path.join(run, "..", "enfant.json")))
    return x - float(gr["parent_x0"]) - c["x0"], y - float(gr["parent_y0"]) - c["y0"]


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
    ap.add_argument("--pointe", nargs=2, type=float, help="centre du tourbillon de pointe (x y, m)")
    ap.add_argument("--pointe-lonlat", nargs=2, type=float, metavar=("LON", "LAT"),
                    help="centre du tourbillon en degres (converti avec --pipeline et enfant.json)")
    ap.add_argument("--pipeline", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                       "pipeline_grille", "output_mini"),
                    help="sortie du pipeline (grids.npz : origine UTM du parent)")
    ap.add_argument("--rayon-pointe", type=float, default=1500.0, help="rayon de la zone de la pointe (m)")
    ap.add_argument("--niveau", type=int, default=1, help="niveau de la liste 2D (0, 1, 2 : ~1, 10, 30 m)")
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

    res = {nm: dict(wrms=[], wmax=[], G=[], u1=[], u2=[], h1=[]) for nm in noms}
    coupes = {nm: {} for nm in noms}
    msk = g.maskC[:, jj, ii]
    k0, k1 = max(ks - 2, 0), min(ks + 2, n - 1)                 # direction du talweg au col
    tx, ty = xs[k1] - xs[k0], ys[k1] - ys[k0]
    tn = math.hypot(tx, ty); tx, ty = tx / tn, ty / tn
    avec_rho = "RHOAnoma" in sd.scan_run(runs[0])[pre]["fields"]
    for run, nm in zip(runs, noms):
        for it in its:
            if avec_rho:
                w, S, u3, v3, rho = lire(run, pre, it, ("WVEL", "SALT", "UVEL", "VVEL", "RHOAnoma"))
                Gc, u1, u2, h1 = froude(u3, v3, rho, g, jj[ks], ii[ks], tx, ty)
                for key, val in (("G", Gc), ("u1", u1), ("u2", u2), ("h1", h1)):
                    res[nm][key].append(val)
            else:
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
            for k in np.where(~g.interior[jj, ii])[0]:          # eponge OBCS : hachuree
                ax.axvspan(se[k] / 1e3, se[k + 1] / 1e3, color="0.85", alpha=.6, lw=0, zorder=0)
            ax.set_ylim(-min(zf[-1], 1.3 * Hp.max()), 0)
            ax.set_title(f"{nm} — phase M2 {ph:.0f}°", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("distance le long du talweg (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("z (m)")
    fig.colorbar(pc, ax=axs, label="w (m/s) ; contours : S ; gris clair : éponge OBCS", shrink=.6)
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

    # --- tourbillon de pointe (liste 2D)
    pre2 = sd.guess_prefixes(sd.scan_run(runs[0]), g.nz)[1]
    edd = None
    if pre2:
        levs = sd.diag_levels(runs[0], pre2)
        li = a.niveau
        kmod = (levs[li] - 1) if levs else li
        wet2 = g.maskC[kmod]
        it2 = np.array(sd.list_iters(runs[0], pre2)); t2 = it2 * g.dt
        s2 = (t2 >= tfin - T_M2 - 1e-6) & (t2 < tfin - 1e-6)
        it2, t2 = it2[s2], t2[s2]
        f0 = np.nanmean(g.fC)
        def zeta_ow(run, it):
            d, _ = sd.read_fields(os.path.join(run, f"{pre2}.{it:010d}"))
            u, v = d["UVEL"][li].astype(np.float32), d["VVEL"][li].astype(np.float32)
            kin = sd.kinematics(u, v, g, wet2)
            return kin["zeta_c"], kin["ow"]
        dcol = np.hypot(g.XC - xs[ks], g.YC - ys[ks])
        if a.pointe_lonlat:
            xp, yp = lonlat_vers_run(*a.pointe_lonlat, runs[-1], a.pipeline)
        elif a.pointe:
            xp, yp = a.pointe
        else:                                   # max de |zeta| moyen, hors de la zone du col
            acc = np.zeros(wet2.shape)
            for it in it2[::3]:
                acc += np.abs(np.nan_to_num(zeta_ow(runs[0], it)[0]))
            acc = ndimage.uniform_filter(acc, 5) * (wet2 & g.interior & (dcol > a.rayon))
            jp, ip = np.unravel_index(np.argmax(acc), acc.shape)
            xp, yp = float(g.XC[jp, ip]), float(g.YC[jp, ip])
        disque = (np.hypot(g.XC - xp, g.YC - yp) <= a.rayon_pointe) & wet2 & g.interior
        edd = {nm: dict(gam=[], aire=[], ro=[]) for nm in noms}
        cartes = {nm: {} for nm in noms}
        ph_it2 = [it2[int(np.argmin(np.abs(t2 - it * g.dt)))] for it in phases]
        for run, nm in zip(runs, noms):
            for it in it2:
                zc, ow = zeta_ow(run, it)
                sig = np.nanstd(ow[wet2 & g.interior])
                core = disque & np.isfinite(ow) & (ow < -0.2 * sig) & (np.abs(np.nan_to_num(zc)) > 0.2 * abs(f0))
                gp_ = float((np.nan_to_num(zc) * g.RAC * (core & (zc > 0))).sum())
                gm_ = float((np.nan_to_num(zc) * g.RAC * (core & (zc < 0))).sum())
                gam = gp_ if abs(gp_) >= abs(gm_) else gm_
                sgn = np.sign(gam) if gam else 1.0
                cc = core & (np.sign(np.nan_to_num(zc)) == sgn)
                edd[nm]["gam"].append(gam)
                edd[nm]["aire"].append(float((g.RAC * cc).sum()))
                edd[nm]["ro"].append(float(np.nanpercentile(np.abs(zc[cc]), 90) / abs(f0)) if cc.any() else 0.0)
                if it in ph_it2:
                    cartes[nm][it] = (zc / f0, cc)
        # cartes zeta/f (zoom talweg + pointe)
        mg = max(2 * a.rayon_pointe, 2000.0)                    # cadre : col + pointe
        box = (min(xp, xs[ks]) - mg, max(xp, xs[ks]) + mg, min(yp, ys[ks]) - mg, max(yp, ys[ks]) + mg)
        ib = (g.XC[0] >= box[0]) & (g.XC[0] <= box[1]); jb = (g.YC[:, 0] >= box[2]) & (g.YC[:, 0] <= box[3])
        asp = (box[3] - box[2]) / (box[1] - box[0])
        fig, axs = plt.subplots(len(ph_it2), len(runs), figsize=(4.5 * len(runs) + 1.5, 4.5 * asp * len(ph_it2) + 1),
                                squeeze=False, constrained_layout=True)
        rolim = max(np.nanpercentile(np.abs(np.concatenate([c[0][np.ix_(jb, ib)].ravel()
                                                             for nm in noms for c in cartes[nm].values()])), 99), 0.2)
        for r, it in enumerate(ph_it2):
            for c, nm in enumerate(noms):
                ro, cc = cartes[nm][it]
                ax = axs[r, c]
                pcm = ax.pcolormesh(g.XC[0][ib] / 1e3, g.YC[:, 0][jb] / 1e3, ro[np.ix_(jb, ib)], cmap="RdBu_r",
                                    vmin=-rolim, vmax=rolim, shading="auto")
                ax.contour(g.XC[0][ib] / 1e3, g.YC[:, 0][jb] / 1e3, cc[np.ix_(jb, ib)].astype(float), [0.5], colors="k", linewidths=.8)
                ax.contour(g.XC[0][ib] / 1e3, g.YC[:, 0][jb] / 1e3, wet2[np.ix_(jb, ib)].astype(float), [0.5], colors="0.3", linewidths=.6)
                ax.plot(xs / 1e3, ys / 1e3, "m-", lw=.6); ax.plot(xs[ks] / 1e3, ys[ks] / 1e3, "mv", ms=7)
                ax.add_patch(plt.Circle((xp / 1e3, yp / 1e3), a.rayon_pointe / 1e3, fill=False, color="g", lw=1))
                ax.set_aspect(1)
                ax.set_title(f"{nm} — phase {((it * g.dt) % T_M2) / T_M2 * 360:.0f}°", fontsize=9)
        for ax in axs.ravel():
            ax.set_xlim(box[0] / 1e3, box[1] / 1e3); ax.set_ylim(box[2] / 1e3, box[3] / 1e3)
        fig.colorbar(pcm, ax=axs, label=f"ζ/f à {-g.RC[kmod]:.0f} m ; noir : cœur O-W ; ▼ col ; cercle : pointe", shrink=.4)
        fig.savefig(os.path.join(od, "fig_pointe_vorticite.png"), dpi=80); plt.close(fig)

        # couplage seuil <-> pointe
        ph2 = ((t2 % T_M2) / T_M2) * 360
        fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        cpl = {}
        for nm in noms:
            if res[nm]["G"]:
                ax[0].plot(ph, res[nm]["G"], label=nm)
            ax[1].plot(ph, np.array(res[nm]["wrms"]) * 1e3, label=nm)
            ax[2].plot(ph2, np.array(edd[nm]["gam"]) / 1e3, label=nm)
            if res[nm]["G"] and np.ptp(edd[nm]["gam"]) > 0:     # pas de tourbillon : pas de correlation
                Gi = np.interp(t2, t, np.nan_to_num(res[nm]["G"]))
                gm = np.abs(edd[nm]["gam"])
                cor = [np.corrcoef(Gi, np.roll(gm, -L))[0, 1] for L in range(len(gm))]
                L = int(np.nanargmax(cor))
                cpl[nm] = (L * (t2[1] - t2[0]) / T_M2 * 360, float(cor[L]))
        ax[0].axhline(1, color="k", lw=.6, ls=":"); ax[0].set_ylabel("Froude interne au col")
        ax[1].set_ylabel("w rms au col (mm/s)"); ax[2].set_ylabel("circulation tourbillon (10³ m²/s)")
        ax[2].set_xlabel("phase M2 (°)")
        for x in ax:
            x.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(od, "fig_seuil_pointe.png"), dpi=100); plt.close(fig)

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
    for nm in noms:
        if res[nm]["G"]:
            Gs = np.array(res[nm]["G"])
            sup = ph[Gs > 1]
            lines.append(f"{nm:24s}: Froude au col max {np.nanmax(Gs):.2f} (phase {ph[np.nanargmax(Gs)]:.0f}°), median {np.nanmedian(Gs):.2f}, "
                         f"supercritique sur {100 * np.mean(Gs > 1):.0f} % du cycle"
                         + (f" (phases {sup.min():.0f}-{sup.max():.0f}°)" if sup.size else ""))
    if edd:
        lines.append(f"Pointe : x={xp:.0f} y={yp:.0f} m, rayon {a.rayon_pointe:.0f} m, niveau {-g.RC[kmod]:.0f} m")
        for nm in noms:
            gam = np.array(edd[nm]["gam"]); k = int(np.argmax(np.abs(gam)))
            if not np.any(gam):
                lines.append(f"{nm:24s}: aucun coeur de tourbillon dans le disque de la pointe")
                continue
            lines.append(f"{nm:24s}: tourbillon {'cyclonique' if gam[k] > 0 else 'anticyclonique'}, circulation max "
                         f"{gam[k] / 1e3:+.1f}e3 m2/s (phase {ph2[k]:.0f}°), aire max {max(edd[nm]['aire']) / 1e6:.2f} km2, "
                         f"|Ro| (90e centile du coeur) max {max(edd[nm]['ro']):.2f}"
                         + (f" ; correlation Froude -> |circulation| max {cpl[nm][1]:.2f} a +{cpl[nm][0]:.0f}°"
                            if nm in cpl else ""))
    lines.append(f"Figures : {od}/fig_seuil_coupes.png, fig_seuil_w.png, fig_pointe_vorticite.png, fig_seuil_pointe.png")
    open(os.path.join(od, "seuil_nh.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
