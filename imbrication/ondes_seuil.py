#!/usr/bin/env python3
"""
Hydraulique du 2e seuil, fronts internes et tourbillon de l'Anse-de-Roche (Q2, Q3).

Le long du talweg (du retrecissement cap / Anse-de-Roche, en amont, au bassin aval du col) et sur un
cycle M2, en heures depuis la basse mer (BM) au mouillage ADCP :
  1. profondeur vraie de l'isohaline S* (interface de la couche saumatre) : diagramme temps-distance ;
     un front ou un ressaut interne y apparait comme une marche qui se deplace ;
  2. vitesse moyenne de la couche de surface (au-dessus de S*) le long du talweg ;
  3. Froude composite a deux couches (Armi 1986) G^2 = u1^2/(g' h1) + u2^2/(g' h2), h1 = profondeur de
     S*, h2 = H - h1, g' = g (rho2 - rho1)/rho0 ; G = 1 : controle hydraulique (contour noir) ;
  4. w* moyen entre 5 et 30 m (montee ou descente de l'interface).
Le col (point le moins profond), le mouillage et, si --detection est donne (anse_detection.csv de
tourbillon_anse.py), le centre du cyclone projete sur le talweg sont reportes.
  5. Coupe en travers du fjord par la cible (cyclone observe), a 6 heures autour de la BM :
     isohalines et vitesse le long du fjord ; bombement de l'interface = profondeur de S* au centre de
     la coupe moins la mediane de la coupe (negatif : interface remontee, comme sous un cyclone).

Instantanes 3D (930 ou 1860 s) : les fronts (~0,5 m/s) sont suivis, pas les ondes solitaires de
5-6 min (il faut le run non hydrostatique a 25 m et la liste hf2D). Profondeur vraie en z* :
z = eta + r*(1 + eta/H). Memoire : un instantane 3D a la fois.

Usage
  python3 imbrication/ondes_seuil.py ~/runs/enfantA_v3/run ~/runs/v3_noslip/run --noms glissant noslip \\
      --origine-utm 430800 5330800 --detection ~/runs/v3_noslip/run/diag/anse_detection.csv
Ecrit dans DERNIER_RUN/diag : ondes_seuil.txt, fig_ondes_talweg.png, fig_ondes_coupe_<nom>.png,
fig_ondes_bombement.png.
"""
import argparse, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "sagdiag"))
import sagdiag as sd
from tourbillon_anse import Repere, lonlat_utm, heure_rel, CIBLE, MOUILLAGE, T_M2
from seuil_nh import talweg

AMONT_UTM = (434600.0, 5341000.0)     # chenal au retrecissement cap / Anse-de-Roche
AVAL_UTM = (438500.0, 5334000.0)      # bassin profond en aval du col


def accroche(g, x, y, rayon=600.0):
    """Cellule mouillee la plus profonde a moins de `rayon` du point, ramenee hors de l'eponge."""
    x = min(max(x, g.XC[0, g.nsponge + 1]), g.XC[0, -g.nsponge - 2])
    y = min(max(y, g.YC[g.nsponge + 1, 0]), g.YC[-g.nsponge - 2, 0])
    d = np.hypot(g.XC - x, g.YC - y)
    m = (d <= rayon) & g.maskC[0] & g.interior
    if not m.any():
        m = g.maskC[0] & g.interior
        return np.unravel_index(np.argmin(np.where(m, d, np.inf)), d.shape)
    return np.unravel_index(np.argmax(np.where(m, g.Depth, -1)), d.shape)


def eta_proche(run, pre, its_e, t, g):
    """ETAN de l'instantane eta2D le plus proche de t."""
    it = int(its_e[np.argmin(np.abs(its_e * g.dt - t))])
    return sd.read_fields(os.path.join(run, f"{pre}.{it:010d}"))[0]["ETAN"].reshape(g.ny, g.nx).astype(np.float64)


def profondeur_iso(S, z, s_star):
    """Profondeur (positive) ou S croise s_star en descendant (premiere traversee) ; nan si absente.
    S, z : (nz, n) ; z profondeur positive croissante ; nan sous le fond."""
    nz, n = S.shape
    out = np.full(n, np.nan)
    for j in range(n):
        col = S[:, j]; ok = np.isfinite(col)
        c, zz = col[ok], z[ok, j]
        if c.size < 2:
            continue
        if c[0] >= s_star:
            out[j] = zz[0]; continue
        k = np.where(c >= s_star)[0]
        if k.size:
            k = k[0]
            out[j] = zz[k - 1] + (s_star - c[k - 1]) / (c[k] - c[k - 1]) * (zz[k] - zz[k - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="dossiers de run (meme grille)")
    ap.add_argument("--noms", nargs="+", default=None)
    ap.add_argument("--origine-utm", nargs=2, type=float, metavar=("X0", "Y0"))
    ap.add_argument("--pipeline", default=os.path.join(ICI, "..", "pipeline_grille", "output_mini"))
    ap.add_argument("--amont-utm", nargs=2, type=float, default=AMONT_UTM)
    ap.add_argument("--aval-utm", nargs=2, type=float, default=AVAL_UTM)
    ap.add_argument("--cible", nargs=2, type=float, default=CIBLE, metavar=("LON", "LAT"))
    ap.add_argument("--mouillage", nargs=2, type=float, default=MOUILLAGE, metavar=("LON", "LAT"))
    ap.add_argument("--sstar", type=float, default=None, help="isohaline de l'interface (defaut : auto)")
    ap.add_argument("--cycle", type=int, default=None, help="cycle analyse (defaut : dernier commun complet)")
    ap.add_argument("--detection", default=None, help="anse_detection.csv (trajectoire du cyclone)")
    ap.add_argument("--coupe-km", type=float, default=3.0, help="longueur de la coupe transverse (km)")
    a = ap.parse_args()
    runs = a.runs
    noms = a.noms or [os.path.basename(os.path.normpath(r)) for r in runs]

    g = sd.Grid(runs[0])
    for r in runs[1:]:
        assert sd.Grid(r).hFacC.shape == g.hFacC.shape, "les runs doivent avoir la meme grille"
    rep = Repere(runs[-1], a.pipeline, a.origine_utm)
    info = sd.scan_run(runs[0])
    pre3, _, pre_e, _ = sd.guess_prefixes(info, g.nz)
    if not pre3 or "SALT" not in info[pre3]["fields"] or not pre_e:
        sys.exit("il faut les instantanes 3D (SALT, UVEL, VVEL, WVEL) et eta2D")
    avec_rho = "RHOAnoma" in info[pre3]["fields"]

    # --- talweg, col, mouillage, cible
    xa, ya = a.amont_utm[0] - rep.ox, a.amont_utm[1] - rep.oy
    xb, yb = a.aval_utm[0] - rep.ox, a.aval_utm[1] - rep.oy
    H = g.Depth * g.maskC[0]
    path = talweg(H, accroche(g, xa, ya), accroche(g, xb, yb))
    jj, ii = path[:, 0], path[:, 1]
    xs, ys = g.XC[jj, ii], g.YC[jj, ii]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    Hp = H[jj, ii]
    n = len(s); lo, hi = n // 10, n - n // 10
    kcol = lo + int(np.argmin(Hp[lo:hi]))
    xm, ym = map(float, rep.xy(*a.mouillage)); xt, yt = map(float, rep.xy(*a.cible))
    kmo = int(np.argmin(np.hypot(xs - xm, ys - ym))); kci = int(np.argmin(np.hypot(xs - xt, ys - yt)))
    jm, im = jj[kmo], ii[kmo]
    k0, k1 = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    tx, ty = xs[k1] - xs[k0], ys[k1] - ys[k0]
    tn = np.hypot(tx, ty); tx, ty = tx / tn, ty / tn            # tangente (vers l'aval)

    # coupe transverse par la cible
    kc = kci
    nx_, ny_ = -ty[kc], tx[kc]
    dl = g.dx_mean
    L = np.arange(-a.coupe_km * 500, a.coupe_km * 500 + 1, dl)
    cj = np.array([int(np.argmin(np.abs(g.YC[:, 0] - (yt + l * ny_)))) for l in L])
    ci = np.array([int(np.argmin(np.abs(g.XC[0] - (xt + l * nx_)))) for l in L])
    inside = g.maskC[0][cj, ci]
    cj, ci, L = cj[inside], ci[inside], L[inside]

    # --- temps : cycle commun, basse mer au mouillage
    it3 = np.array(sd.list_iters(runs[0], pre3))
    for r in runs[1:]:
        it3 = np.intersect1d(it3, sd.list_iters(r, pre3))
    t3 = it3 * g.dt
    dto = float(np.median(np.diff(t3)))
    cyc = np.floor(t3 / T_M2 + 1e-9).astype(int)
    complets = [c for c in np.unique(cyc) if (cyc == c).sum() >= 0.95 * T_M2 / dto]
    if not complets:
        sys.exit("aucun cycle complet commun")
    cy = a.cycle if a.cycle is not None else complets[-1]
    sel = cyc == cy
    it3, t3 = it3[sel], t3[sel]
    ite = np.array(sd.list_iters(runs[0], pre_e)); te = ite * g.dt
    se = (te >= cy * T_M2) & (te < (cy + 1) * T_M2)
    eta_m = np.array([sd.read_fields(os.path.join(runs[0], f"{pre_e}.{it:010d}"))[0]["ETAN"].reshape(g.ny, g.nx)[jm, im]
                      for it in ite[se]])
    tbm = float(te[se][np.argmin(eta_m)])
    h3 = heure_rel(t3, tbm); o = np.argsort(h3)
    hc = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5]                      # heures des coupes
    itc = [int(it3[np.argmin(np.abs(h3 - h))]) for h in hc]

    # --- S* : salinite a la profondeur du max de N2 moyen au col (premier instantane)
    zc = -g.RC
    if a.sstar is None:
        d, _ = sd.read_fields(os.path.join(runs[0], f"{pre3}.{it3[0]:010d}"))
        col = d["SALT"][:, jj[kcol], ii[kcol]].astype(float)
        kk = np.where(g.maskC[:, jj[kcol], ii[kcol]])[0]
        ks = kk[np.argmax(np.diff(col[kk]) / np.diff(zc[kk]))]
        sstar = float(round(0.5 * (col[ks] + col[ks + 1]), 1))
    else:
        sstar = a.sstar

    res, coupes = {}, {}
    msk = g.maskC[:, jj, ii]
    for run, nm in zip(runs, noms):
        R = dict(h1=[], u1=[], G=[], w=[], bomb=[], zmoor=[])
        coupes[nm] = {}
        for it, t in zip(it3, t3):
            d, _ = sd.read_fields(os.path.join(run, f"{pre3}.{it:010d}"))
            eta = eta_proche(run, pre_e, ite, t, g)
            Hd = np.maximum(g.Depth, 1e-3)
            def coupe(js, is_):
                e = eta[js, is_]; Hh = Hd[js, is_]
                z = -(e[None] + (-zc[:, None]) * (1 + e / Hh)[None])           # profondeur vraie (> 0)
                m = g.maskC[:, js, is_]
                S = np.where(m, d["SALT"][:, js, is_], np.nan)
                u = 0.5 * (d["UVEL"][:, js, is_] + d["UVEL"][:, js, np.minimum(is_ + 1, g.nx - 1)])
                v = 0.5 * (d["VVEL"][:, js, is_] + d["VVEL"][:, np.minimum(js + 1, g.ny - 1), is_])
                return z, m, S, np.where(m, u, np.nan), np.where(m, v, np.nan)
            z, m, S, u, v = coupe(jj, ii)
            ua = u * tx[None] + v * ty[None]                                    # vitesse vers l'aval
            h1 = profondeur_iso(S, z, sstar)
            dz = g.DRF[:, None] * g.hFacC[:, jj, ii]
            up = m & (z <= h1[None]); dn = m & (z > h1[None])
            u1 = np.nansum(ua * dz * up, 0) / np.maximum((dz * up).sum(0), 1e-6)
            u2 = np.nansum(ua * dz * dn, 0) / np.maximum((dz * dn).sum(0), 1e-6)
            if avec_rho:
                rho = np.where(m, d["RHOAnoma"][:, jj, ii], np.nan)
                r1 = np.nansum(rho * dz * up, 0) / np.maximum((dz * up).sum(0), 1e-6)
                r2 = np.nansum(rho * dz * dn, 0) / np.maximum((dz * dn).sum(0), 1e-6)
                gp = np.clip(sd.G * (r2 - r1) / g.rho0, 1e-5, None)
            else:                                                               # rho ~ 0,78 kg/m3 par psu
                S1 = np.nansum(S * dz * up, 0) / np.maximum((dz * up).sum(0), 1e-6)
                S2 = np.nansum(S * dz * dn, 0) / np.maximum((dz * dn).sum(0), 1e-6)
                gp = np.clip(sd.G * 0.78 * (S2 - S1) / g.rho0, 1e-5, None)
            h2 = np.maximum(Hp - h1, 0.5)
            G2 = u1 ** 2 / (gp * np.maximum(h1, 0.5)) + u2 ** 2 / (gp * h2)
            G = np.where(np.isfinite(h1), np.sqrt(G2), np.nan)
            wz = np.where(m & (z >= 5) & (z <= 30), d["WVEL"][:, jj, ii], np.nan)
            R["h1"].append(h1); R["u1"].append(np.where(np.isfinite(h1), u1, np.nan)); R["G"].append(G)
            R["w"].append(np.nanmean(wz, 0)); R["zmoor"].append(h1[kmo])
            zc_, mc, Sc, uc, vc = coupe(cj, ci)
            h1c = profondeur_iso(Sc, zc_, sstar)
            R["bomb"].append(float(h1c[len(h1c) // 2] - np.nanmedian(h1c)) if np.isfinite(h1c).sum() > 3 else np.nan)
            if it in itc:
                coupes[nm][int(it)] = (zc_, Sc, uc * tx[kc] + vc * ty[kc], h1c)
        res[nm] = {k: np.array(v) for k, v in R.items()}

    od = os.path.join(runs[-1], "diag"); os.makedirs(od, exist_ok=True)
    # trajectoire du cyclone (anse_detection.csv)
    traj = {}
    if a.detection and os.path.exists(a.detection):
        for r in csv.DictReader(open(a.detection)):
            if int(r["cycle"]) != cy or float(r["gam"]) <= 0 or r["x"] in ("nan", ""):
                continue
            x_, y_ = float(r["x"]), float(r["y"])
            k = int(np.argmin(np.hypot(xs - x_, ys - y_)))
            traj.setdefault(r["run"], []).append((float(r["h"]), s[k] / 1e3, float(r["gam"])))

    # --- figure temps-distance
    hh = h3[o]
    fig, axs = plt.subplots(4, len(runs), figsize=(6 * len(runs), 14), squeeze=False, sharex=True, sharey=True)
    h1lim = np.nanpercentile(np.concatenate([res[nm]["h1"].ravel() for nm in noms]), [2, 98])
    ulim = np.nanpercentile(np.abs(np.concatenate([res[nm]["u1"].ravel() for nm in noms])), 98)
    wlim = max(np.nanpercentile(np.abs(np.concatenate([res[nm]["w"].ravel() for nm in noms])), 98), 1e-4)
    for c, nm in enumerate(noms):
        R = res[nm]
        pan = ((R["h1"], "viridis_r", h1lim, f"profondeur de S = {sstar:g} (m)"),
               (R["u1"], "RdBu_r", (-ulim, ulim), "u couche de surface vers l'aval (m/s)"),
               (np.log10(np.clip(R["G"], 1e-2, None)), "PuOr_r", (-1, 1), "log10 G (Froude composite) ; noir : G = 1"),
               (R["w"], "RdBu_r", (-wlim, wlim), "w* moyen 5-30 m (m/s)"))
        for r_, (q, cm, lim, lab) in enumerate(pan):
            ax = axs[r_, c]
            pc = ax.pcolormesh(s / 1e3, hh, q[o], cmap=cm, vmin=lim[0], vmax=lim[1], shading="auto")
            if r_ == 2:
                ax.contour(s / 1e3, hh, np.nan_to_num(R["G"][o], nan=0), [1.0], colors="k", linewidths=.8)
            for k_, st, lb in ((kcol, "m--", "col"), (kmo, "k:", "mouillage")):
                ax.axvline(s[k_] / 1e3, color=st[0], ls=st[1:], lw=1)
            ax.axhline(0, color="0.4", lw=.6)
            if nm in traj:
                tr = np.array(traj[nm]); ax.plot(tr[:, 1], tr[:, 0], "o", ms=2.5, mfc="none", mec="m")
            fig.colorbar(pc, ax=ax, label=lab, shrink=.85)
            if c == 0:
                ax.set_ylabel("heure depuis la BM (h)")
        axs[0, c].set_title(f"{nm} — cycle {cy} ; tirets : col ({Hp[kcol]:.0f} m) ; pointillé : mouillage", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("distance le long du talweg (km) : amont (rétrécissement) → aval (bassin)")
    fig.tight_layout(); fig.savefig(os.path.join(od, "fig_ondes_talweg.png"), dpi=90); plt.close(fig)

    # --- coupes transverses par la cible
    for nm in noms:
        fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
        cl = np.nanpercentile(np.abs(np.concatenate([c[2].ravel() for c in coupes[nm].values()])), 98)
        for ax, h, it in zip(axs.ravel(), hc, itc):
            zz, Sc, ac, h1c = coupes[nm][it]
            X = np.broadcast_to(L[None] / 1e3, zz.shape)
            pc = ax.pcolormesh(X, -zz, ac, cmap="RdBu_r", vmin=-cl, vmax=cl, shading="nearest")
            ax.contour(X, -zz, np.nan_to_num(Sc, nan=40), levels=np.arange(10, 34, 2.0), colors="k", linewidths=.5)
            ax.plot(L / 1e3, -h1c, "m-", lw=1.2)
            ax.axvline(0, color="g", lw=.8, ls=":")
            ax.set_ylim(-min(60, np.nanmax(zz)), 1); ax.set_title(f"{nm} — BM {h:+.1f} h", fontsize=9)
        for ax in axs[-1]:
            ax.set_xlabel("distance à la cible en travers du fjord (km)")
        for ax in axs[:, 0]:
            ax.set_ylabel("z (m)")
        fig.colorbar(pc, ax=axs, label=f"vitesse vers l'aval (m/s) ; contours : S ; magenta : S = {sstar:g}", shrink=.6)
        fig.savefig(os.path.join(od, f"fig_ondes_coupe_{nm}.png"), dpi=90); plt.close(fig)

    # --- bombement et interface au mouillage
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for nm in noms:
        ax[0].plot(hh, res[nm]["bomb"][o], label=nm)
        ax[1].plot(hh, res[nm]["zmoor"][o], label=nm)
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_ylabel("bombement de S* à la cible (m)\n(< 0 : interface remontée)")
    ax[1].invert_yaxis(); ax[1].set_ylabel(f"profondeur de S = {sstar:g} au mouillage (m)")
    ax[1].set_xlabel("heure depuis la BM (h)")
    for x in ax:
        x.axvline(0, color="k", lw=.6, ls=":"); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(od, "fig_ondes_bombement.png"), dpi=100); plt.close(fig)

    # --- resume
    lines = [f"Talweg {s[-1] / 1e3:.1f} km (amont -> aval), col {Hp[kcol]:.1f} m a s = {s[kcol] / 1e3:.2f} km, "
             f"mouillage a s = {s[kmo] / 1e3:.2f} km (H talweg {Hp[kmo]:.0f} m), cible a s = {s[kci] / 1e3:.2f} km",
             f"Cycle {cy}, {len(it3)} instantanes 3D toutes les {dto:.0f} s ; BM au mouillage t = {tbm:.0f} s ; "
             f"interface S* = {sstar:g}" + (" (auto : max de N2 au col)" if a.sstar is None else "")]
    for nm in noms:
        R = res[nm]
        Gc = R["G"][:, kcol]; hcrit = h3[Gc > 1]
        amont = slice(0, kcol); aval = slice(kcol + 1, n)
        def frac(sl):
            q = R["G"][:, sl]
            return 100 * np.nanmean(np.nanmax(q, 1) > 1)
        zmo = R["zmoor"]
        kb = int(np.nanargmin(R["bomb"])) if np.isfinite(R["bomb"]).any() else 0
        lines.append(f"{nm:10s} col : G max {np.nanmax(Gc):.2f} (BM {h3[np.nanargmax(Gc)]:+.1f} h), critique (G > 1) "
                     f"{100 * np.nanmean(Gc > 1):.0f} % du cycle"
                     + (f" (BM {hcrit.min():+.1f} a {hcrit.max():+.1f} h)" if hcrit.size else "")
                     + f" ; un point critique existe en amont {frac(amont):.0f} %, en aval {frac(aval):.0f} % du cycle")
        lines.append(f"{'':10s} interface au mouillage : {np.nanmin(zmo):.1f} a {np.nanmax(zmo):.1f} m "
                     f"(la plus haute a BM {h3[np.nanargmin(zmo)]:+.1f} h) ; bombement a la cible min "
                     f"{np.nanmin(R['bomb']):+.1f} m a BM {h3[kb]:+.1f} h")
        if nm in traj:
            tr = np.array(traj[nm]); k = int(np.argmax(tr[:, 2]))
            lines.append(f"{'':10s} cyclone (detection) : au max de Gamma, s = {tr[k, 1]:.2f} km a BM {tr[k, 0]:+.1f} h")
    lines.append(f"Figures : {od}/fig_ondes_talweg.png, fig_ondes_coupe_*.png, fig_ondes_bombement.png")
    open(os.path.join(od, "ondes_seuil.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
