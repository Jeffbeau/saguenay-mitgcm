#!/usr/bin/env python3
"""
Animations d'un run sur le dernier cycle M2 complet, et arguments pour aller plus loin
(resolution, cout).

  anim_w_talweg.gif   w vraie le long du talweg (couleur) et isopycnes, surface libre (state3D)
  anim_w_carte.gif    w vraie en plan a 2 profondeurs (--profondeurs, defaut 10 et 40 m)
  anim_vorticite.gif  zeta/f et fleches a ~1 m (liste 2D)
  fig_echelles_w.png  longueur d'onde des structures de w le long du talweg (passages par zero),
                      comparee a la resolution effective ~8 dx de ce run et d'un run a dx/2,
                      et au tourbillon de pointe observe (Livernoche 2017 : r ~115 m)
  animations.txt      echelles ; cout du run (temps dans STDOUT.0000 ou output.txt) et
                      extrapolation aux configurations visees (docs/decisions.md)

En z*, WVEL est la vitesse r* : w vraie = w*(1 + eta/H) + (1 - d/H) deta/dt (d : profondeur r*).
Memoire : sections et cartes 2D seulement (quelques dizaines de Mo).

Usage
  python3 sagdiag/animations.py ~/runs/enfantA_v3/run --origine-utm 430800 5330800
"""
import argparse, glob, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sagdiag as sd
from cartes_coupes import (Run, T_M2, CAP_UTM, CM_DIV, fr, rond, origine, phases_cycle, etiquette,
                           fond_carte, fleches, faces_z, pcolor_colonnes, geometrie_talweg)

D_LIVERNOCHE = 230.0          # diametre du tourbillon de pointe observe (Livernoche 2017, r ~115 m)
# Configurations visees (docs/decisions.md) : nom, nx, ny, nr, dx (m), duree simulee (s)
CIBLES = [("Enfant 25 m au cap (mini25), 2 cycles", 160, 200, 32, 25.0, 2 * T_M2),
          ("Parent 100 m, tout le fjord (full), 6 jours", 1200, 532, 60, 100.0, 6 * 86400.0),
          ("Enfant 25 m, zone des seuils (full), 2 jours", 1504, 864, 60, 25.0, 2 * 86400.0)]
KO_PAR_POINT = 3.8e9 / 5.0e6 / 1e3   # borne basse : le profil lite (5 M points) a depasse 3,8 Go


def lire(R, prefix, it, noms):
    """Champs choisis d'un fichier de diagnostics : seuls leurs enregistrements sont lus."""
    base = R.base(prefix, it)
    if prefix not in R.__dict__.setdefault("_fl", {}):
        R._fl[prefix] = sd.parse_meta(sd._meta_files(base)[0])["fldList"]
    arr, _ = sd.read_mds(base, recs=[R._fl[prefix].index(n) for n in noms])
    return [np.asarray(a, np.float32) for a in arr]


def detadt(R, t):
    if R.pe is None:
        return np.zeros((R.g.ny, R.g.nx))
    h = 0.5 * float(R.te[1] - R.te[0])
    return (R.eta(t + h) - R.eta(t - h)) / (2 * h)


def w_vraie(R, wc, eta, de, H, d):
    """w vraie aux centres a partir de w* (z*) ; d : profondeur r* des centres, meme forme que wc."""
    if not R.zstar:
        return wc
    Hs = np.maximum(H, 1e-3)
    return wc * (1 + eta / Hs) + (1 - np.minimum(d, H) / Hs) * de


def inset_maree(fig, rect, serie, tdeb):
    if not serie:
        return None
    axm = fig.add_axes(rect)
    axm.plot((serie[0] - tdeb) / 3600, serie[1], color="#1f4e79", lw=1)
    pt, = axm.plot([], [], "o", color="#c51b7d", ms=5)
    axm.set_xlabel("h", fontsize=7); axm.set_ylabel("η (m)", fontsize=7); axm.tick_params(labelsize=6)
    return pt


def echelles(D, s, inter, frac_seuil=0.1):
    """Demi-longueurs d'onde de w le long du talweg : distance entre passages par zero successifs,
    par niveau et par instantane, sur les troncons mouilles hors eponge ; un troncon compte si
    |w| y depasse frac_seuil x le 99e centile. Renvoie longueurs d'onde (m) et poids w2 * longueur."""
    wl = np.nanpercentile(np.abs(np.concatenate([d["w"][:, inter].ravel() for d in D])), 99)
    ds = np.gradient(s)
    lam, ener = [], []
    for d in D:
        for row in d["w"]:
            ok = np.isfinite(row) & inter
            idx = np.where(ok)[0]
            for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                if run.size < 3:
                    continue
                r, sr = row[run], s[run]
                c = np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
                xc = sr[c] + (sr[c + 1] - sr[c]) * r[c] / (r[c] - r[c + 1])
                for m in range(c.size - 1):
                    seg = slice(c[m] + 1, c[m + 1] + 1)
                    if np.max(np.abs(r[seg])) < frac_seuil * wl:
                        continue
                    lam.append(2 * (xc[m + 1] - xc[m]))
                    ener.append(float(np.sum(r[seg] ** 2 * ds[run][seg])))
    return np.array(lam), np.array(ener), wl


def cout_run(run, g):
    """Temps de calcul (MITgcm : section ALL de THE_MODEL_MAIN), nombre de processus et de pas du
    meme segment de run (en-tete du meme fichier : nProcs, nTimeSteps relu de data)."""
    for f in [os.path.join(run, n) for n in ("STDOUT.0000", "output.txt", "nohup.out")] + \
             [os.path.join(run, "..", "output.txt")]:
        if not os.path.exists(f):
            continue
        with open(f, "rb") as fh:
            fh.seek(max(0, os.path.getsize(f) - 400_000))
            txt = fh.read().decode(errors="ignore")
        m = re.search(r'Seconds in section "ALL\s+\[THE_MODEL_MAIN\]".*?Wall clock time:\s*([0-9.Ee+-]+)', txt, re.S)
        if m:
            with open(f, "rb") as fh:
                tete = fh.read(400_000).decode(errors="ignore")
            mp = re.search(r"nProcs\s*=\s*(\d+)", tete)
            ms = re.search(r">\s*nTimeSteps\s*=\s*(\d+)", tete, re.I)
            nproc = int(mp.group(1)) if mp else (len(glob.glob(os.path.join(run, "STDOUT.[0-9]*"))) or 1)
            return float(m.group(1)), nproc, os.path.basename(f), (int(ms.group(1)) if ms else None)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", help="dossier (defaut RUN/diag/figures)")
    ap.add_argument("--origine-utm", nargs=2, type=float, metavar=("X0", "Y0"))
    ap.add_argument("--pipeline", default=os.path.join(HERE, "..", "pipeline_grille", "output_mini"))
    ap.add_argument("--cap", nargs=2, type=float, metavar=("X", "Y"), help="cap (m, repere du run)")
    ap.add_argument("--cycle", type=int, help="cycle [k T, (k+1) T) ; defaut : dernier complet")
    ap.add_argument("--profondeurs", nargs=2, type=float, default=[10.0, 40.0], help="w en plan (m)")
    ap.add_argument("--zmax", type=float, help="profondeur de l'animation du talweg (defaut : tout)")
    ap.add_argument("--pas", type=int, default=2, help="un instantane 2D sur N pour la vorticite")
    ap.add_argument("--niveau", type=int, default=0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--state3d"); ap.add_argument("--lev2d"); ap.add_argument("--eta")
    a = ap.parse_args()

    R = Run(a.run, a)
    g = R.g
    if "WVEL" not in R.champs3:
        sys.exit(f"pas de WVEL dans {R.p3}")
    od = a.out or os.path.join(a.run, "diag", "figures")
    os.makedirs(od, exist_ok=True)
    org = origine(a, a.run)
    ox, oy = (org[0], org[1]) if org else (0.0, 0.0)
    X, Y = (g.XC[0] + ox) / 1e3, (g.YC[:, 0] + oy) / 1e3
    xlab, ylab = ("x UTM 19N (km)", "y UTM 19N (km)") if org else ("x (km)", "y (km)")
    dx = float(g.DXC[0, 0])
    wet = g.maskC[0]
    H = np.where(wet, g.Depth, np.nan)
    Hz = g.Depth * wet
    inter2 = wet & g.interior
    f0 = float(np.nanmean(g.fC))
    cap = tuple(a.cap) if a.cap else ((CAP_UTM[0] - ox, CAP_UTM[1] - oy) if org else None)
    if cap and not (g.XG[0, 0] <= cap[0] <= g.XG[0, -1] and g.YG[0, 0] <= cap[1] <= g.YG[-1, 0]):
        cap = None

    tl = R.t3[-1] + (R.t3[1] - R.t3[0] if R.t3.size > 1 else 0)
    if R.pe is not None:
        tl = min(tl, R.te[-1] + (R.te[1] - R.te[0]))
    tdeb = a.cycle * T_M2 if a.cycle is not None else max(math.floor(tl / T_M2 + 0.05) * T_M2 - T_M2, R.t3[0])
    phases, serie, hpm, tpm = phases_cycle(R, tdeb)
    lines = [f"Run : {os.path.abspath(a.run)} ({g.nx}x{g.ny}x{g.nz}, dx {dx:.0f} m, "
             f"{'z*' if R.zstar else 'surface libre linéaire'}) ; cycle {tdeb:.0f}-{tdeb + T_M2:.0f} s"]

    # ------------------------------------------------------------------ passe 3D : talweg et cartes de w
    geo = geometrie_talweg(g)
    jj, ii, kc = geo["jj"], geo["ii"], geo["kc"]
    s = geo["s"]; skm = s / 1e3
    se = np.r_[skm[0] - (skm[1] - skm[0]) / 2, (skm[1:] + skm[:-1]) / 2, skm[-1] + (skm[-1] - skm[-2]) / 2]
    Hs = Hz[jj, ii]
    inter_s = g.interior[jj, ii]
    rho = "RHOAnoma" in R.champs3
    qn = "RHOAnoma" if rho else "SALT"
    d3 = np.broadcast_to(-g.RC[:, None, None], g.hFacC.shape)
    kp = [int(np.argmin(np.abs(-g.RC - p))) for p in a.profondeurs]
    its = R.it3[(R.t3 >= tdeb - 1e-6) & (R.t3 < tdeb + T_M2 - 1e-6)]
    D, cartes = [], []
    for it in its:
        t = it * g.dt
        w3, q3 = lire(R, R.p3, it, ("WVEL", qn))
        eta, de = R.eta(t), detadt(R, t)
        wc = w_vraie(R, sd.w2c(w3).astype(np.float64), eta[None], de[None], Hz[None], d3)
        wc[~g.maskC] = np.nan
        q = q3[:, jj, ii].astype(np.float64) + (g.rho0 - 1000.0 if rho else 0.0)
        q[~g.maskC[:, jj, ii]] = np.nan
        zf = faces_z(R, jj, ii, eta[jj, ii])
        D.append(dict(t=t, w=wc[:, jj, ii], q=q, zf=zf, zc=.5 * (zf[1:] + zf[:-1]), eta=eta[jj, ii]))
        cartes.append([wc[k].astype(np.float32) for k in kp])
        del w3, q3, wc
    if not D:
        sys.exit("aucun instantané 3D dans le cycle choisi")
    lines.append(f"{len(D)} instantanés 3D (toutes les {R.t3[1] - R.t3[0]:.0f} s)")

    # ------------------------------------------------------------------ animation : w le long du talweg
    wl = rond(float(np.nanpercentile(np.abs(np.concatenate([d["w"][:, inter_s].ravel() for d in D])), 99)))
    allq = np.concatenate([d["q"][np.isfinite(d["q"])] for d in D])
    qlev = np.unique(np.round(np.nanpercentile(allq, np.linspace(4, 96, 13)), 2))
    zmax = a.zmax or float(Hs.max())
    top = max(float(np.nanmax(d["eta"])) for d in D) + 0.5
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.065, 0.11, 0.74, 0.76])
    fig.colorbar(ScalarMappable(Normalize(-wl * 1e3, wl * 1e3), CM_DIV), cax=fig.add_axes([0.83, 0.11, 0.013, 0.52]),
                 label="w (mm/s) ; rouge : montant")
    pt = inset_maree(fig, [0.855, 0.72, 0.12, 0.16], serie, tdeb)
    X2 = np.broadcast_to(skm, D[0]["zc"].shape)
    def frame_talweg(n):
        d = D[n]
        ax.clear()
        pcolor_colonnes(ax, skm, d["zf"], d["w"], cmap=CM_DIV, vmin=-wl, vmax=wl)
        ax.contour(X2, d["zc"], np.ma.masked_invalid(d["q"]), levels=qlev, colors="k", linewidths=.5)
        ax.plot(skm, d["eta"], color="#1f4e79", lw=1)
        ax.fill_between(se, np.r_[-Hs, -Hs[-1]], -zmax - 10, step="post", color="0.6", zorder=3)
        for k in np.where(~inter_s)[0]:
            ax.axvspan(se[k], se[k + 1], ymin=.96, color="0.55", lw=0, zorder=5)
        ax.axvline(skm[kc], color="k", lw=.8, ls=":")
        ax.annotate("col", (skm[kc], 1.0), xycoords=("data", "axes fraction"), xytext=(2, -10),
                    textcoords="offset points", fontsize=8)
        ax.set_xlim(se[0], se[-1]); ax.set_ylim(-zmax, top)
        ax.set_xlabel(f"distance le long du talweg (km), amont (OB {geo['b0']}) → aval (OB {geo['b1']})")
        ax.set_ylabel("z (m)")
        ax.set_title(f"w le long du talweg — {etiquette(d['t'], serie, hpm)} ; contours : "
                     f"{'isopycnes' if rho else 'isohalines'} ; bande grise : éponge", fontsize=10)
        if pt is not None:
            pt.set_data([(d["t"] - tdeb) / 3600], [np.interp(d["t"], serie[0], serie[1], period=T_M2)])
    FuncAnimation(fig, frame_talweg, frames=len(D)).save(os.path.join(od, "anim_w_talweg.gif"),
                                                       writer=PillowWriter(fps=a.fps), dpi=80)
    plt.close(fig)

    # ------------------------------------------------------------------ animation : w en plan
    asp = (Y[-1] - Y[0]) / (X[-1] - X[0])
    fig = plt.figure(figsize=(11, 5.2 * asp + 1.6))
    axs = [fig.add_axes([0.06 + 0.42 * m, 0.08, 0.38, 0.8]) for m in range(2)]
    pcs, wls = [], []
    for m, ax in enumerate(axs):
        wm = rond(float(np.nanpercentile(np.abs(np.concatenate([c[m][inter2] for c in cartes])), 99)))
        wls.append(wm)
        pcs.append(ax.pcolormesh(X, Y, np.where(wet, cartes[0][m], np.nan), cmap=CM_DIV, vmin=-wm, vmax=wm,
                                 shading="nearest"))
        ax.set_facecolor("0.6")                     # fond moins profond que le niveau trace
        fond_carte(ax, R, X, Y, H)
        if cap:
            ax.plot((cap[0] + ox) / 1e3, (cap[1] + oy) / 1e3, "*", ms=10, mfc="gold", mec="k", zorder=5)
        ax.plot((geo["xs"][kc] + ox) / 1e3, (geo["ys"][kc] + oy) / 1e3, "v", ms=8, mfc="w", mec="k", zorder=5)
        ax.set_title(f"w à {fr(-g.RC[kp[m]])} m (échelle ± {fr(wm * 1e3, '{:g}')} mm/s)", fontsize=10)
        ax.set_xlabel(xlab)
    axs[0].set_ylabel(ylab); axs[1].set_yticklabels([])
    fig.colorbar(ScalarMappable(Normalize(-1, 1), CM_DIV), cax=fig.add_axes([0.9, 0.08, 0.012, 0.5]),
                 label="w / échelle du panneau ; rouge : montant ;\ngris foncé : fond moins profond")
    pt = inset_maree(fig, [0.875, 0.72, 0.11, 0.15], serie, tdeb)
    sup = fig.suptitle("", fontsize=11)
    def frame_carte(n):
        for m in range(2):
            pcs[m].set_array(np.where(wet, cartes[n][m], np.nan).ravel())
        sup.set_text(f"Vitesse verticale — {etiquette(D[n]['t'], serie, hpm)}"
                     + (" ; ▽ col, ★ cap" if cap else " ; ▽ col"))
        if pt is not None:
            pt.set_data([(D[n]["t"] - tdeb) / 3600], [np.interp(D[n]["t"], serie[0], serie[1], period=T_M2)])
    FuncAnimation(fig, frame_carte, frames=len(D)).save(os.path.join(od, "anim_w_carte.gif"),
                                                      writer=PillowWriter(fps=a.fps), dpi=80)
    plt.close(fig)
    for m, p in enumerate(a.profondeurs):
        wmx = max(float(np.nanmax(np.abs(c[m][inter2]))) for c in cartes)
        lines.append(f"w à {-g.RC[kp[m]]:.1f} m : échelle (99e centile arrondi) {wls[m] * 1e3:g} mm/s, max {wmx * 1e3:.1f} mm/s")
    wmx_t = max(float(np.nanmax(np.abs(d["w"][:, inter_s]))) for d in D)
    lines.append(f"w le long du talweg : échelle {wl * 1e3:g} mm/s, max {wmx_t * 1e3:.1f} mm/s")

    # ------------------------------------------------------------------ animation : vorticite de surface
    if R.p2 is not None:
        tt = R.it2 * g.dt
        it2 = R.it2[(tt >= tdeb - 1e-6) & (tt < tdeb + T_M2 - 1e-6)][::max(1, a.pas)]
        ech = [R.surface(it * g.dt) for it in it2[::4]]
        zl = max(float(np.nanpercentile(np.abs(np.concatenate([e[2][inter2] for e in ech])), 99)) / abs(f0), 0.5)
        vref = rond(float(np.nanpercentile(np.concatenate([np.hypot(e[0], e[1])[inter2] for e in ech]), 95)))
        st = max(1, int(round(g.nx / 28)))
        fig = plt.figure(figsize=(7.5, 7.5 * asp + 1.2))
        ax = fig.add_axes([0.1, 0.07, 0.72, 0.8])
        uc, vc, zc, ts = R.surface(it2[0] * g.dt)
        pc = ax.pcolormesh(X, Y, np.where(wet, zc / f0, np.nan), cmap=CM_DIV, vmin=-zl, vmax=zl, shading="nearest")
        fond_carte(ax, R, X, Y, H)
        q = fleches(ax, X, Y, uc, vc, st, vref, X[1] - X[0])
        ax.quiverkey(q, 0.8, 1.02, vref, f"{fr(vref, '{:g}')} m/s", labelpos="E", coordinates="axes",
                     fontproperties={"size": 8})
        if cap:
            ax.plot((cap[0] + ox) / 1e3, (cap[1] + oy) / 1e3, "*", ms=11, mfc="gold", mec="k", zorder=5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        zs = -g.RC[R.k2]
        fig.colorbar(pc, cax=fig.add_axes([0.85, 0.07, 0.025, 0.55]),
                     label=f"ζ/f à {fr(zs)} m ; rouge : cyclonique")
        ttl = ax.set_title("", fontsize=10, loc="left")
        pt = inset_maree(fig, [0.84, 0.72, 0.14, 0.15], serie, tdeb)
        def frame_vort(n):
            uc, vc, zc, ts = R.surface(it2[n] * g.dt)
            pc.set_array(np.where(wet, zc / f0, np.nan).ravel())
            q.set_UVC(np.ma.masked_invalid(uc[st // 2::st, st // 2::st]), np.ma.masked_invalid(vc[st // 2::st, st // 2::st]))
            ttl.set_text(f"Vorticité de surface — {etiquette(ts, serie, hpm)}")
            if pt is not None:
                pt.set_data([(ts - tdeb) / 3600], [np.interp(ts, serie[0], serie[1], period=T_M2)])
        FuncAnimation(fig, frame_vort, frames=len(it2)).save(os.path.join(od, "anim_vorticite.gif"),
                                                            writer=PillowWriter(fps=8), dpi=80)
        plt.close(fig)
        lines.append(f"Vorticité de surface : {len(it2)} images, |ζ/f| (99e centile) {zl:.2f}")

    # ------------------------------------------------------------------ echelles de w : ce que la grille resout
    lam, en, wl99 = echelles(D, s, inter_s)
    if lam.size > 10:
        p = en / en.sum()
        res, res2 = 8 * dx, 4 * dx
        lines.append(f"Structures de w le long du talweg : {lam.size} demi-ondes (|w| > {fr(0.1 * wl99 * 1e3)} mm/s), "
                     f"λ médiane {np.median(lam):.0f} m = {np.median(lam) / dx:.1f} Δx")
        for lab, L in ((f"8 Δx = {res:.0f} m (résolution effective de ce run)", res),
                       (f"8 Δx à {dx / 2:g} m = {res2:.0f} m", res2)):
            lines.append(f"  λ < {lab} : {100 * np.mean(lam < L):.0f} % des structures, "
                         f"{100 * p[lam < L].sum():.0f} % de l'énergie w²")
        bins = np.geomspace(min(1.5 * dx, lam.min() * 0.9, 0.8 * D_LIVERNOCHE), lam.max() * 1.1, 28)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        axs[0].hist(lam, bins=bins, weights=100 * p, color="#3d6ea8", edgecolor="w", linewidth=.6)
        axs[0].set_ylabel("part de l'énergie w² (%)")
        axs[0].set_ylim(0, 1.5 * axs[0].get_ylim()[1])           # place pour les etiquettes
        o = np.argsort(lam)
        axs[1].plot(lam[o], 100 * np.cumsum(p[o]), color="#3d6ea8", lw=2)
        axs[1].set_ylabel("énergie w² cumulée (%)"); axs[1].set_ylim(0, 100)
        reps = [(2 * dx, "2 Δx\n(maille)", "0.35"), (res, f"8 Δx = {res:.0f} m\nrésolution\neffective", "k"),
                (res2, f"8 Δx à {dx / 2:g} m", "#c51b7d"), (D_LIVERNOCHE, "tourbillon\nobservé\nD ≈ 230 m", "goldenrod")]
        for axx in axs:
            axx.set_xscale("log"); axx.set_xlabel("longueur d'onde λ de w le long du talweg (m)")
            axx.grid(alpha=.3, lw=.5, which="both")
            for L, lab, cl in reps:
                axx.axvline(L, color=cl, lw=1.2, ls="--")
            axx.set_xlim(bins[0], bins[-1])
        for m, (L, lab, cl) in enumerate(sorted(reps)):          # etiquettes etagees : lignes voisines
            axs[0].annotate(lab, (L, 1 - 0.08 * m), xycoords=("data", "axes fraction"), xytext=(3, -4),
                            textcoords="offset points", va="top", fontsize=7.5, color=cl)
        fig.suptitle(f"Taille des structures de w (passages par zéro le long du talweg, {len(D)} instantanés, "
                     f"tous niveaux, hors éponge) — Δx = {dx:.0f} m", fontsize=10)
        fig.savefig(os.path.join(od, "fig_echelles_w.png"), dpi=110); plt.close(fig)

    # ------------------------------------------------------------------ cout et extrapolation
    c = cout_run(a.run, g)
    dt = float(g.dt)
    npts = g.nx * g.ny * g.nz
    nst = (c[3] if c else None) or sd.nml_value(os.path.join(a.run, "data"), "nTimeSteps")
    if c and nst:
        wall, nproc, src = c[:3]
        cps = wall * nproc / (npts * nst)                  # coeur.s par point et par pas
        lines.append(f"Coût du run ({src}) : {wall / 3600:.1f} h sur {nproc} processus pour {int(nst)} pas de "
                     f"{dt:g} s, {npts / 1e6:.2f} M points -> {cps:.2e} cœur.s par point et par pas")
        lines.append("Extrapolation (hydrostatique, même code ; Δt ∝ Δx ; non hydrostatique : plus cher, à mesurer) :")
        for nom, nx, ny, nr, dxc, duree in [("Ce run, 1 cycle M2", g.nx, g.ny, g.nz, dx, T_M2)] + CIBLES:
            n = nx * ny * nr
            dtc = dt * dxc / dx
            ch = cps * n * (duree / dtc) / 3600
            g_ = lambda v: f"{v:.2g}" if v < 10 else f"{v:.0f}"
            lines.append(f"  {nom:46s} {n / 1e6:5.1f} M points, Δt {dtc:4.1f} s : {g_(ch):>6s} cœur.h "
                         f"(VM 3 cœurs : {g_(ch / 3 / 24):>5s} j ; 128 cœurs : {g_(ch / 128):>5s} h) ; "
                         f"mémoire ≳ {g_(n * KO_PAR_POINT / 1e6)} Go")
    else:
        lines.append("Coût du run : temps de calcul introuvable (STDOUT.0000 ou output.txt sans la section ALL)")

    lines.append(f"Animations et figure : {od}")
    open(os.path.join(od, "animations.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
