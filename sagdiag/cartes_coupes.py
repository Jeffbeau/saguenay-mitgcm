#!/usr/bin/env python3
"""
Figures de presentation d'un run : courants de surface et coupes verticales zoomees sur la
pycnocline, aux moments cles du dernier cycle M2 complet.

  Phases (eta moyen du domaine, eta2D) : flot (deta/dt max), pleine mer, jusant (deta/dt min),
  basse mer. Sans eta2D : 0, 90, 180 et 270 deg du cycle.
  Cartes (liste 2D, niveau --niveau, ~1 m ; sinon state3D au premier niveau) :
    fig_courants_surface.png   vitesse en couleur, fleches
    fig_vorticite_surface.png  zeta/f en couleur, fleches
    fig_cap_zoom.png           lignes de courant sur zeta/f autour du cap, 8 instants du cycle
  Coupes (state3D ; profondeur vraie z = eta + r*(1 + eta/H) en z*) :
    fig_carte_coupes.png       bathymetrie, talweg et coupes transversales (col, cap)
    fig_profils.png            profils moyens S, T, N2 : profondeur de la pycnocline
    fig_coupe_<nom>.png        sigma (couleur, isopycnes) | vitesse vers l'aval, 4 phases,
                               de la surface a --zmax m (auto : 2 x la base de la pycnocline)
  courants_surface.gif (--gif) : animation des courants de surface sur le cycle
  cartes_coupes.txt          resume (phases, pycnocline, vitesses, vorticite au cap)

Reperes : axes en km UTM 19N si l'origine est connue (--origine-utm, ou grids.npz de --pipeline
et enfant.json a cote du run) ; le cap de la Pointe-aux-Crepes est alors place d'office.
Memoire : un champ 3D a la fois (VM de 3,8 Go ok).

Usage
  python3 sagdiag/cartes_coupes.py ~/runs/enfantA_v3/run --origine-utm 430800 5330800 [--gif]
"""
import argparse, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "imbrication"))
import sagdiag as sd
from seuil_nh import segments_ob, talweg

T_M2 = 44640.0
CAP_UTM = (433540.0, 5340762.0)        # cap de la Pointe-aux-Crepes, 48,2166 N 69,8947 O
PYC = "#c51b7d"                        # trait de la pycnocline


def tronque(nom, a=0.05, b=0.85):
    cm = plt.get_cmap(nom)
    return LinearSegmentedColormap.from_list(nom + "_t", cm(np.linspace(a, b, 256)))


CM_VIT, CM_SIG, CM_DIV = tronque("Greens"), tronque("Blues"), plt.get_cmap("RdBu_r")


def fr(x, fmt="{:.1f}"):
    return fmt.format(x).replace(".", ",").replace("-", "−")


def rond(v):
    """Valeur ronde juste au-dessus de v (echelles de fleches, pas d'isopycnes)."""
    for c in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        if v <= c:
            return c
    return float(math.ceil(v))


def boussole(dx, dy):
    a = math.degrees(math.atan2(dy, dx)) % 360
    return ["E", "NE", "N", "NO", "O", "SO", "S", "SE"][int((a + 22.5) // 45) % 8]


def origine(a, run):
    if a.origine_utm:
        return a.origine_utm[0], a.origine_utm[1], "--origine-utm"
    pj, ej = os.path.join(a.pipeline, "grids.npz"), os.path.join(run, "..", "enfant.json")
    if os.path.exists(pj) and os.path.exists(ej):
        gr, c = np.load(pj), json.load(open(ej))
        return float(gr["parent_x0"]) + c["x0"], float(gr["parent_y0"]) + c["y0"], "grids.npz + enfant.json"
    return None


class Run:
    """Acces aux sorties : instantanes 3D, liste 2D, eta (interpole en temps)."""

    def __init__(self, run, a):
        self.run, self.g = run, sd.Grid(run)
        g = self.g
        info = sd.scan_run(run)
        p3, p2, pe, _ = sd.guess_prefixes(info, g.nz)
        self.p3, self.p2, self.pe = a.state3d or p3, a.lev2d or p2, a.eta or pe
        if self.p3 is None:
            sys.exit("aucun instantane 3D (UVEL, VVEL sur Nr niveaux) dans " + run)
        self.champs3 = info[self.p3]["fields"]
        self.it3 = np.array(sd.list_iters(run, self.p3)); self.t3 = self.it3 * g.dt
        self.ite = np.array(sd.list_iters(run, self.pe)) if self.pe else None
        self.te = self.ite * g.dt if self.pe else None
        self.it2 = np.array(sd.list_iters(run, self.p2)) if self.p2 else None
        self.li, self.k2 = 0, 0
        if self.p2:
            self.li = a.niveau
            levs = sd.diag_levels(run, self.p2)
            self.k2 = (levs[self.li] - 1) if levs else self.li
        d = os.path.join(run, "data")
        self.zstar = (int(sd.nml_value(d, "nonlinFreeSurf", 0) or 0) > 0
                      and int(sd.nml_value(d, "select_rStar", 0) or 0) > 0)
        self._eta = {}

    def base(self, prefix, it):
        return os.path.join(self.run, f"{prefix}.{int(it):010d}")

    def eta_it(self, it):
        if it not in self._eta:
            if len(self._eta) > 8:
                self._eta.pop(next(iter(self._eta)))
            d, _ = sd.read_fields(self.base(self.pe, it))
            self._eta[it] = np.asarray(d["ETAN"], np.float64).reshape(self.g.ny, self.g.nx)
        return self._eta[it]

    def eta(self, t):
        """eta (carte) a l'instant t, lineaire entre les instantanes ; zeros sans eta2D."""
        if self.pe is None:
            return np.zeros((self.g.ny, self.g.nx))
        k = int(np.searchsorted(self.te, t))
        if k <= 0 or k >= self.te.size:
            return self.eta_it(self.ite[min(max(k, 0), self.te.size - 1)])
        w = (t - self.te[k - 1]) / (self.te[k] - self.te[k - 1])
        return (1 - w) * self.eta_it(self.ite[k - 1]) + w * self.eta_it(self.ite[k])

    def surface(self, t):
        """u, v (centres) et zeta (centres) au niveau de surface, instantane le plus proche de t."""
        g = self.g
        if self.p2:
            it = self.it2[np.argmin(np.abs(self.it2 * g.dt - t))]
            d, _ = sd.read_fields(self.base(self.p2, it))
            u, v = d["UVEL"][self.li], d["VVEL"][self.li]
            wet = g.maskC[self.k2]
        else:
            it = self.it3[np.argmin(np.abs(self.t3 - t))]
            d, _ = sd.read_fields(self.base(self.p3, it), levels=[0])
            u, v = d["UVEL"][0], d["VVEL"][0]
            wet = g.maskC[0]
        u, v = np.asarray(u, np.float32), np.asarray(v, np.float32)
        uc, vc = sd.u2c(u), sd.v2c(v)
        uc[~wet] = np.nan; vc[~wet] = np.nan
        zc = sd.kinematics(u, v, g, wet)["zeta_c"]
        return uc, vc, zc, it * g.dt

    def etat(self, t):
        """Instantane 3D le plus proche de t : uc, vc (centres), sigma (ou S), nom du champ."""
        g = self.g
        it = self.it3[np.argmin(np.abs(self.t3 - t))]
        rho = "RHOAnoma" in self.champs3
        noms = ("UVEL", "VVEL", "RHOAnoma" if rho else "SALT", "THETA") if "THETA" in self.champs3 else \
               ("UVEL", "VVEL", "RHOAnoma" if rho else "SALT")
        d, _ = sd.read_fields(self.base(self.p3, it))
        uc, vc = sd.u2c(np.asarray(d["UVEL"], np.float32)), sd.v2c(np.asarray(d["VVEL"], np.float32))
        q = np.asarray(d[noms[2]], np.float64) + (g.rho0 - 1000.0 if rho else 0.0)
        S = np.asarray(d["SALT"], np.float64) if "SALT" in d else None
        T = np.asarray(d["THETA"], np.float64) if "THETA" in d else None
        for f in (uc, vc, q):
            f[~g.maskC] = np.nan
        return uc, vc, q, S, T, it * g.dt, ("σ (kg/m³)" if rho else "S")


def phases_cycle(R, tdeb):
    """Flot, pleine mer, jusant, basse mer sur eta moyen ; (cle, titre, t), serie eta, t PM."""
    g = R.g
    inter = g.maskC[0] & g.interior
    if R.pe is None:
        ph = [("flot", "0°", tdeb), ("pm", "90°", tdeb + T_M2 / 4),
              ("jusant", "180°", tdeb + T_M2 / 2), ("bm", "270°", tdeb + 3 * T_M2 / 4)]
        return ph, None, None, tdeb
    sel = (R.te >= tdeb - 1e-6) & (R.te < tdeb + T_M2 - 1e-6)
    tw = R.te[sel]
    em = np.array([R.eta_it(it)[inter].mean() for it in R.ite[sel]])
    dte = float(np.median(np.diff(tw)))
    if abs(tw.size * dte - T_M2) < 0.5 * dte:            # cycle complet : derivee periodique
        de = (np.roll(em, -1) - np.roll(em, 1)) / (2 * dte)
    else:
        de = np.gradient(em, tw)
    tpm = tw[np.argmax(em)]
    def hpm(t):
        h = ((t - tpm + T_M2 / 2) % T_M2 - T_M2 / 2) / 3600
        return "PM" if abs(h) < 0.05 else f"PM{'+' if h > 0 else '−'}{fr(abs(h))} h"
    ph = [("flot", f"Flot ({hpm(tw[np.argmax(de)])})", tw[np.argmax(de)]),
          ("pm", f"Pleine mer (η = {fr(em.max(), '{:+.2f}')} m)", tpm),
          ("jusant", f"Jusant ({hpm(tw[np.argmin(de)])})", tw[np.argmin(de)]),
          ("bm", f"Basse mer (η = {fr(em.min(), '{:+.2f}')} m, {hpm(tw[np.argmin(em)])})", tw[np.argmin(em)])]
    return ph, (tw, em, de), hpm, tpm


def etiquette(ts, serie, hpm):
    """'PM+1,6 h — jusant' : heure par rapport a la pleine mer, flot/jusant/etale selon deta/dt."""
    if not serie:
        return f"{(ts % T_M2) / T_M2 * 360:.0f}°"
    de = np.interp(ts, serie[0], serie[2], period=T_M2)
    q = de / max(np.abs(serie[2]).max(), 1e-12)
    return hpm(ts) + (" — flot" if q > 0.2 else " — jusant" if q < -0.2 else " — étale")


def fond_carte(ax, R, X, Y, H, lim=None, contours=(50, 100, 200)):
    g = R.g
    ax.pcolormesh(X, Y, np.where(g.maskC[0], np.nan, 1.0), cmap="Greys", vmin=0, vmax=4.5,
                  shading="nearest", zorder=0)
    cs = ax.contour(X, Y, H, levels=list(contours), colors="0.45", linewidths=.4, zorder=1)
    ax.clabel(cs, fmt="%d m", fontsize=6)
    n = g.nsponge
    if n:
        dx = X[1] - X[0]
        ax.add_patch(Rectangle((X[n] - dx / 2, Y[n] - dx / 2), X[-n - 1] - X[n] + dx, Y[-n - 1] - Y[n] + dx,
                               fill=False, ls="--", lw=.7, ec="0.35", zorder=3))
    ax.set_aspect(1)
    if lim:
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])


def fleches(ax, X, Y, uc, vc, st, vref, dxkm, color="k"):
    sl = (slice(st // 2, None, st), slice(st // 2, None, st))
    return ax.quiver(X[sl[1]], Y[sl[0]], np.ma.masked_invalid(uc[sl]), np.ma.masked_invalid(vc[sl]),
                     scale=vref / (st * dxkm), scale_units="xy", angles="xy", width=.0022,
                     headwidth=3.5, headlength=4, color=color, zorder=4)


def section_droite(g, x0, y0, nx_, ny_, lmax):
    """Cellules mouillees le long de la droite (x0, y0) + s (nx_, ny_), jusqu'au rivage des deux cotes."""
    ds = float(g.DXC[0, 0])
    xg, yg = g.XG[0], g.YG[:, 0]
    def cell(x, y):
        i, j = int(np.searchsorted(xg, x, "right") - 1), int(np.searchsorted(yg, y, "right") - 1)
        ok = 0 <= i < g.nx and 0 <= j < g.ny and x < xg[-1] + ds and y < yg[-1] + ds
        return (j, i) if ok and g.maskC[0, j, i] else None
    pts = []
    for sg in (-1, 1):
        k = 1 if sg > 0 else 0
        while k * ds <= lmax:
            c = cell(x0 + sg * k * ds * nx_, y0 + sg * k * ds * ny_)
            if c is None:
                break
            pts.append((sg * k * ds, *c)); k += 1
    pts.sort()
    s = np.array([p[0] for p in pts]); jj = np.array([p[1] for p in pts]); ii = np.array([p[2] for p in pts])
    return s - s[0], jj, ii


def faces_z(R, jj, ii, eta):
    """Profondeur vraie des faces (nz+1, n) des colonnes (jj, ii) : z* ou surface libre lineaire."""
    g = R.g
    Hc = g.Depth[jj, ii]
    rf = -np.r_[0.0, np.cumsum(g.DRF)][:, None]
    rf = np.maximum(rf, -Hc[None, :])
    if R.zstar:
        return eta[None, :] + rf * (1 + eta / np.maximum(Hc, 1e-3))[None, :]
    zf = rf.copy(); zf[0] = eta
    return zf


def pcolor_colonnes(ax, s, zf, q, **kw):
    """Chaque colonne avec ses propres faces (z vraie) : coins dedoubles, colonnes intercalees vides."""
    n = s.size
    se = np.r_[s[0] - (s[1] - s[0]) / 2, (s[1:] + s[:-1]) / 2, s[-1] + (s[-1] - s[-2]) / 2]
    Xc = np.repeat(se, 2)[1:-1]
    Xc = np.broadcast_to(Xc, (zf.shape[0], 2 * n))
    Zc = np.repeat(zf, 2, axis=1)
    C = np.full((zf.shape[0] - 1, 2 * n - 1), np.nan)
    C[:, ::2] = q
    return ax.pcolormesh(Xc, Zc, np.ma.masked_invalid(C), shading="flat", **kw)


def pycnocline_colonne(sig, zc, zmin=-np.inf, seuil=0.0):
    """z de N2 max par colonne (interface entre les deux cellules au plus fort gradient), au-dessus
    de zmin ; NaN si le gradient max est sous `seuil` (stratification faible)."""
    dsz = (sig[1:] - sig[:-1]) / np.maximum(zc[:-1] - zc[1:], 1e-6)
    zi = 0.5 * (zc[1:] + zc[:-1])
    dsz = np.where(np.isfinite(dsz) & (zi >= zmin), dsz, -np.inf)
    k = np.argmax(dsz, axis=0)
    ok = np.isfinite(dsz.max(axis=0)) & (dsz.max(axis=0) > max(seuil, 0.0))
    n = np.arange(sig.shape[1])
    return np.where(ok, 0.5 * (zc[k, n] + zc[k + 1, n]), np.nan)


def geometrie_talweg(g):
    """Talweg entre les deux frontieres ouvertes les plus eloignees, oriente amont (ouest) -> aval :
    cellules (jj, ii), positions, distance s, tangente lissee sur ~500 m (tx, ty), indice du col."""
    wet = g.maskC[0]
    Hz = g.Depth * wet
    segs = segments_ob(wet)
    if len(segs) >= 2:
        best = max(((s1, s2) for k, s1 in enumerate(segs) for s2 in segs[k + 1:]),
                   key=lambda p: math.dist(p[0][1], p[1][1]))
        p0, p1 = best[0][1], best[1][1]
        b0, b1 = best[0][0], best[1][0]
    else:                                   # pas deux frontieres ouvertes : extremes du domaine mouille
        jw, iw = np.where(wet)
        p0, p1 = (jw[np.argmin(iw)], iw.min()), (jw[np.argmax(iw)], iw.max())
        b0, b1 = "O", "E"
    if g.XC[p0] > g.XC[p1]:                 # amont (ouest) -> aval (est, embouchure)
        p0, p1, b0, b1 = p1, p0, b1, b0
    path = talweg(Hz, p0, p1)
    jj, ii = path[:, 0], path[:, 1]
    xs, ys = g.XC[jj, ii], g.YC[jj, ii]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    nw = max(2, int(round(250.0 / float(g.DXC[0, 0]))))
    tx = np.array([xs[min(n + nw, xs.size - 1)] - xs[max(n - nw, 0)] for n in range(xs.size)])
    ty = np.array([ys[min(n + nw, ys.size - 1)] - ys[max(n - nw, 0)] for n in range(ys.size)])
    nn = np.hypot(tx, ty)
    Hp = Hz[jj, ii]; n = s.size
    kc = n // 10 + int(np.argmin(Hp[n // 10: n - n // 10]))
    return dict(jj=jj, ii=ii, xs=xs, ys=ys, s=s, tx=tx / nn, ty=ty / nn, kc=kc, b0=b0, b1=b1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", help="dossier des figures (defaut RUN/diag/figures)")
    ap.add_argument("--origine-utm", nargs=2, type=float, metavar=("X0", "Y0"),
                    help="coin sud-ouest du run en UTM 19N (m) ; enfant 50 m v3 : 430800 5330800")
    ap.add_argument("--pipeline", default=os.path.join(HERE, "..", "pipeline_grille", "output_mini"),
                    help="sortie du pipeline (grids.npz) pour l'origine, avec enfant.json a cote du run")
    ap.add_argument("--cap", nargs=2, type=float, metavar=("X", "Y"), help="cap (m, repere du run)")
    ap.add_argument("--rayon-cap", type=float, default=1500.0, help="demi-largeur du zoom sur le cap (m)")
    ap.add_argument("--cycle", type=int, help="cycle a montrer, [k T, (k+1) T) ; defaut : dernier complet")
    ap.add_argument("--zmax", type=float, help="profondeur du zoom des coupes (m) ; 0 : toute la colonne")
    ap.add_argument("--niveau", type=int, default=0, help="niveau de la liste 2D pour les cartes (0 : ~1 m)")
    ap.add_argument("--gif", action="store_true", help="animation des courants de surface sur le cycle")
    ap.add_argument("--gif-pas", type=int, default=2, help="un instantane 2D sur N dans l'animation")
    ap.add_argument("--state3d"); ap.add_argument("--lev2d"); ap.add_argument("--eta")
    a = ap.parse_args()

    R = Run(a.run, a)
    g = R.g
    od = a.out or os.path.join(a.run, "diag", "figures")
    os.makedirs(od, exist_ok=True)
    org = origine(a, a.run)
    ox, oy = (org[0], org[1]) if org else (0.0, 0.0)
    X, Y = (g.XC[0] + ox) / 1e3, (g.YC[:, 0] + oy) / 1e3
    xlab, ylab = ("x UTM 19N (km)", "y UTM 19N (km)") if org else ("x (km)", "y (km)")
    dxkm = float(X[1] - X[0])
    wet = g.maskC[0]
    H = np.where(wet, g.Depth, np.nan)
    inter = wet & g.interior
    f0 = float(np.nanmean(g.fC))
    lines = [f"Run : {os.path.abspath(a.run)} ({g.nx}x{g.ny}x{g.nz}, dx {dxkm * 1e3:.0f} m, "
             f"{'z*' if R.zstar else 'surface libre lineaire'})",
             f"Repere : {'UTM 19N, origine ' + f'{ox:.0f} {oy:.0f} m ({org[2]})' if org else 'coin SO du run (origine UTM inconnue)'}"]

    # cap (repere du run, m)
    cap = None
    if a.cap:
        cap = tuple(a.cap)
    elif org:
        cap = (CAP_UTM[0] - ox, CAP_UTM[1] - oy)
    if cap and not (g.XG[0, 0] <= cap[0] <= g.XG[0, -1] and g.YG[0, 0] <= cap[1] <= g.YG[-1, 0]):
        lines.append("Cap de la Pointe-aux-Crêpes hors du domaine : pas de zoom ni de coupe au cap")
        cap = None

    # ------------------------------------------------------------------ cycle et phases
    tl = R.t3[-1] + (R.t3[1] - R.t3[0] if R.t3.size > 1 else 0)
    if R.pe is not None:
        tl = min(tl, R.te[-1] + (R.te[1] - R.te[0]))
    if a.cycle is not None:
        tdeb = a.cycle * T_M2
    else:
        tdeb = math.floor(tl / T_M2 + 0.05) * T_M2 - T_M2
        if tdeb < R.t3[0] - 1e-6:
            tdeb = R.t3[0]
            lines.append("ATTENTION : moins d'un cycle complet dans le run")
    phases, serie, hpm, tpm = phases_cycle(R, tdeb)
    lines.append(f"Cycle montré : {tdeb:.0f}-{tdeb + T_M2:.0f} s (cycle {tdeb / T_M2:.2f}) ; "
                 f"instantanés 3D toutes les {R.t3[1] - R.t3[0]:.0f} s"
                 + (f", liste 2D niveau {-g.RC[R.k2]:.1f} m" if R.p2 else ", pas de liste 2D (cartes au 1er niveau 3D)"))
    for k, ttl, t in phases:
        lines.append(f"  {ttl:45s} t = {t:.0f} s, phase M2 {(t % T_M2) / T_M2 * 360:.0f}°")

    # ------------------------------------------------------------------ cartes de surface
    surf = {k: R.surface(t) for k, _, t in phases}
    sp = {k: np.hypot(v[0], v[1]) for k, v in surf.items()}
    vmax = float(np.nanpercentile(np.concatenate([s[inter] for s in sp.values()]), 99.5))
    vref = rond(float(np.nanpercentile(np.concatenate([s[inter] for s in sp.values()]), 95)))
    st = max(1, int(round(g.nx / 28)))
    zlim = max(float(np.nanpercentile(np.abs(np.concatenate([v[2][inter] for v in surf.values()])), 99)) / abs(f0), 0.5)
    zsurf = -g.RC[R.k2] if R.p2 else -g.RC[0]
    asp = (Y[-1] - Y[0]) / (X[-1] - X[0])
    for nom, champ in (("courants", "vitesse"), ("vorticite", "zeta")):
        fig, axs = plt.subplots(2, 2, figsize=(11, 11 * asp * 0.92 + 1.2), sharex=True, sharey=True,
                                constrained_layout=True)
        for ax, (k, ttl, t) in zip(axs.ravel(), phases):
            uc, vc, zc, ts = surf[k]
            if champ == "vitesse":
                pc = ax.pcolormesh(X, Y, np.where(wet, sp[k], np.nan), cmap=CM_VIT, vmin=0, vmax=vmax,
                                   shading="nearest")
            else:
                pc = ax.pcolormesh(X, Y, np.where(wet, zc / f0, np.nan), cmap=CM_DIV, vmin=-zlim, vmax=zlim,
                                   shading="nearest")
            fond_carte(ax, R, X, Y, H)
            q = fleches(ax, X, Y, uc, vc, st, vref, dxkm)
            if cap:
                ax.plot((cap[0] + ox) / 1e3, (cap[1] + oy) / 1e3, "*", ms=11, mfc="gold", mec="k", zorder=5)
            ax.set_title(ttl, fontsize=10)
        ax.quiverkey(q, 0.86, 1.03, vref, f"{fr(vref, '{:g}')} m/s", labelpos="E", coordinates="axes",
                     fontproperties={"size": 8})
        for ax in axs[-1]:
            ax.set_xlabel(xlab)
        for ax in axs[:, 0]:
            ax.set_ylabel(ylab)
        lab = (f"vitesse à {fr(zsurf)} m (m/s)" if champ == "vitesse" else f"ζ/f à {fr(zsurf)} m")
        fig.colorbar(pc, ax=axs, shrink=.6, label=lab + " ; tirets : limite de l'éponge OBCS"
                     + (" ; ★ cap de la Pointe-aux-Crêpes" if cap else ""))
        fig.savefig(os.path.join(od, f"fig_{nom}_surface.png"), dpi=110); plt.close(fig)
    lines.append(f"Surface ({fr(zsurf)} m) : vitesse max {max(np.nanmax(s[inter]) for s in sp.values()):.2f} m/s, "
                 f"|ζ/f| (99e centile) {zlim:.2f}")

    # zoom sur le cap : lignes de courant, 8 instants du cycle a partir de la pleine mer
    if cap:
        rc = a.rayon_cap
        ib = np.where(np.abs(g.XC[0] - cap[0]) <= rc)[0]; jb = np.where(np.abs(g.YC[:, 0] - cap[1]) <= rc)[0]
        ib = ib if ib.size > 3 else np.arange(g.nx); jb = jb if jb.size > 3 else np.arange(g.ny)
        sl = np.ix_(jb, ib)
        t8 = [tdeb + ((tpm - tdeb) + n * T_M2 / 8) % T_M2 for n in range(8)]
        z8 = [R.surface(t) for t in t8]
        disque = (np.hypot(g.XC - cap[0], g.YC - cap[1]) <= min(rc, 600.0)) & inter
        zl = max(float(np.nanpercentile(np.abs(np.concatenate([z[2][sl].ravel() for z in z8])), 99)) / abs(f0), 0.5)
        smax = max(float(np.nanmax(np.hypot(z[0][sl], z[1][sl]))) for z in z8)
        fig, axs = plt.subplots(2, 4, figsize=(16, 8.6), sharex=True, sharey=True, constrained_layout=True)
        lines.append(f"Cap (x={cap[0]:.0f} y={cap[1]:.0f} m, repère du run) : |ζ/f| max à moins de "
                     f"{min(rc, 600.0):.0f} m, par instant :")
        for ax, t, (uc, vc, zc, ts) in zip(axs.ravel(), t8, z8):
            ttl = etiquette(ts, serie, hpm)
            pc = ax.pcolormesh(X[ib], Y[jb], np.where(wet[sl], zc[sl] / f0, np.nan), cmap=CM_DIV,
                               vmin=-zl, vmax=zl, shading="nearest")
            fond_carte(ax, R, X, Y, H, lim=(X[ib[0]], X[ib[-1]], Y[jb[0]], Y[jb[-1]]), contours=(20, 50, 100, 200))
            spd = np.hypot(uc[sl], vc[sl])
            if np.isfinite(spd).sum() > 10:
                ax.streamplot(X[ib], Y[jb], np.ma.masked_invalid(uc[sl]), np.ma.masked_invalid(vc[sl]),
                              density=1.3, color="k", linewidth=np.nan_to_num(0.3 + 1.5 * spd / smax),
                              arrowsize=.7, zorder=4)
            ax.plot((cap[0] + ox) / 1e3, (cap[1] + oy) / 1e3, "*", ms=12, mfc="gold", mec="k", zorder=5)
            ax.set_title(ttl, fontsize=10)
            zd = np.where(disque, zc / f0, np.nan)
            if np.isfinite(zd).any():
                km_ = np.nanargmax(np.abs(zd))
                lines.append(f"  {ttl:22s} ζ/f = {zd.flat[km_]:+.2f}, vitesse max zoom {np.nanmax(spd):.2f} m/s")
        for ax in axs[-1]:
            ax.set_xlabel(xlab)
        for ax in axs[:, 0]:
            ax.set_ylabel(ylab)
        fig.colorbar(pc, ax=axs, shrink=.6, label=f"ζ/f à {fr(zsurf)} m (rouge : cyclonique) ; "
                     f"lignes de courant, épaisseur ∝ vitesse (max {fr(smax, '{:.2f}')} m/s)")
        fig.savefig(os.path.join(od, "fig_cap_zoom.png"), dpi=100); plt.close(fig)

    # ------------------------------------------------------------------ coupes : geometrie
    Hz = g.Depth * wet
    T_ = geometrie_talweg(g)
    tj, ti, xs, ys, s_t, tx, ty, kc, b0, b1 = (T_[k] for k in ("jj", "ii", "xs", "ys", "s", "tx", "ty", "kc", "b0", "b1"))
    Hp = Hz[tj, ti]
    coupes = [dict(nom="talweg", titre=f"talweg, amont (OB {b0}) → aval (OB {b1})", s=s_t, jj=tj, ii=ti,
                   tx=tx, ty=ty, xlab=f"distance le long du talweg (km), amont (OB {b0}) → aval (OB {b1})")]
    def transverse(nom, titre, kp):
        nx_, ny_ = -ty[kp], tx[kp]                       # normale au talweg
        if (abs(ny_) >= abs(nx_) and ny_ < 0) or (abs(ny_) < abs(nx_) and nx_ < 0):
            nx_, ny_ = -nx_, -ny_
        s, jj, ii = section_droite(g, xs[kp], ys[kp], nx_, ny_, 6000.0)
        if s.size < 4:
            lines.append(f"Coupe {nom} ignorée : {s.size} cellules mouillées seulement")
            return
        coupes.append(dict(nom=nom, titre=titre, s=s, jj=jj, ii=ii, tx=np.full(s.size, tx[kp]),
                           ty=np.full(s.size, ty[kp]),
                           xlab=f"distance (km), {boussole(-nx_, -ny_)} → {boussole(nx_, ny_)}"))
    transverse("col", f"coupe transversale au col ({Hp[kc]:.0f} m)", kc)
    kcap = int(np.argmin(np.hypot(xs - cap[0], ys - cap[1]))) if cap else None
    if cap:
        transverse("cap", "coupe transversale au cap", kcap)

    # carte des coupes
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * asp + .6), constrained_layout=True)
    pc = ax.pcolormesh(X, Y, H, cmap=tronque("Blues", .1, .95), shading="nearest")
    fond_carte(ax, R, X, Y, H)
    for c in coupes:
        xx, yy = (g.XC[c["jj"], c["ii"]] + ox) / 1e3, (g.YC[c["jj"], c["ii"]] + oy) / 1e3
        ax.plot(xx, yy, "-", color="k" if c["nom"] == "talweg" else PYC, lw=1.4)
        ax.annotate(c["nom"], (xx[-1], yy[-1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.plot((xs[kc] + ox) / 1e3, (ys[kc] + oy) / 1e3, "v", ms=9, mfc="w", mec="k", zorder=5)
    if cap:
        ax.plot((cap[0] + ox) / 1e3, (cap[1] + oy) / 1e3, "*", ms=12, mfc="gold", mec="k", zorder=5)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    fig.colorbar(pc, ax=ax, shrink=.7, label="profondeur (m) ; ▽ col" + (" ; ★ cap" if cap else ""))
    fig.savefig(os.path.join(od, "fig_carte_coupes.png"), dpi=110); plt.close(fig)

    # ------------------------------------------------------------------ coupes : donnees aux 4 phases
    dat = {c["nom"]: [] for c in coupes}
    prof = []
    for k, ttl, t in phases:
        uc, vc, q, S, T, ts, qnom = R.etat(t)
        eta = R.eta(ts)
        for c in coupes:
            jj, ii = c["jj"], c["ii"]
            zf = faces_z(R, jj, ii, eta[jj, ii])
            ua = uc[:, jj, ii] * c["tx"] + vc[:, jj, ii] * c["ty"]
            dat[c["nom"]].append(dict(ttl=ttl, ts=ts, zf=zf, zc=.5 * (zf[1:] + zf[:-1]), q=q[:, jj, ii],
                                      u=ua, eta=eta[jj, ii]))
        m3 = g.maskC & g.interior[None]
        moy = lambda f: np.array([np.nanmean(f[kk][m3[kk]]) if m3[kk].any() else np.nan for kk in range(g.nz)])
        prof.append((moy(q), moy(S) if S is not None else None, moy(T) if T is not None else None))
        del uc, vc, q, S, T

    # profils moyens et pycnocline
    zr = -g.RC
    qm = np.nanmean([p[0] for p in prof], axis=0)
    dq = np.diff(qm) / np.diff(zr)
    zi = 0.5 * (zr[1:] + zr[:-1])
    sig = qnom.startswith("σ")
    N2 = sd.G / g.rho0 * dq if sig else dq
    ok = np.isfinite(N2)
    kp = int(np.nanargmax(np.where(ok, N2, -np.inf)))
    kb = kp
    while kb + 1 < N2.size and ok[kb + 1] and N2[kb + 1] >= 0.1 * N2[kp]:
        kb += 1
    kt = kp
    while kt - 1 >= 0 and ok[kt - 1] and N2[kt - 1] >= 0.1 * N2[kp]:
        kt -= 1
    zpyc, zbase, ztop = zi[kp], zi[kb], zi[kt]
    if a.zmax is not None:
        zmax = a.zmax if a.zmax > 0 else float(np.nanmax(Hz))
    else:
        zmax = min(max(5 * math.ceil(2 * zbase / 5), 15.0), float(np.nanmax(Hz)))
    lines.append(f"Pycnocline (moyenne horizontale hors éponge, 4 phases) : N² max à {zpyc:.1f} m "
                 f"({'N² = ' + fr(N2[kp], '{:.2e}') + ' s⁻²' if sig else 'dS/dz max'}), "
                 f"de {ztop:.1f} à {zbase:.1f} m (N² > 10 % du max) ; zoom des coupes 0-{zmax:.0f} m")
    for (k, ttl, t), p in zip(phases, prof):
        dqp = np.diff(p[0]) / np.diff(zr)
        lines.append(f"  {ttl:45s} N² max à {zi[int(np.nanargmax(np.where(np.isfinite(dqp), dqp, -np.inf)))]:.1f} m")

    fig, axs = plt.subplots(1, 3, figsize=(11, 5), sharey=True, constrained_layout=True)
    zmp = min(max(2.5 * zbase, 30.0), float(np.nanmax(Hz)))
    km_ = zr <= zmp
    vals = [(qm, qnom)]
    if prof[0][1] is not None and sig:
        vals = [(np.nanmean([p[1] for p in prof], axis=0), "S")]
    vals.append((np.nanmean([p[2] for p in prof], axis=0), "T (°C)") if prof[0][2] is not None else (qm, qnom))
    for ax, (v, lab) in zip(axs[:2], vals):
        ax.plot(v[km_], -zr[km_], "o-", color="#1f4e79", ms=3, lw=1.6)
        ax.set_xlabel(lab)
    axs[2].plot(N2[zi <= zmp], -zi[zi <= zmp], "o-", color=PYC, ms=3, lw=1.6)
    axs[2].set_xlabel("N² (s⁻²)" if sig else "dS/dz (1/m)")
    for ax in axs:
        ax.axhspan(-zbase, -ztop, color=PYC, alpha=.08, lw=0)
        ax.axhline(-zpyc, color=PYC, ls="--", lw=.9)
        ax.grid(alpha=.3, lw=.5)
    axs[0].set_ylabel("profondeur sous la surface (m)")
    axs[2].annotate(f"pycnocline {fr(zpyc)} m", (N2[kp], -zpyc), xytext=(-8, 6), textcoords="offset points",
                    ha="right", fontsize=8, color=PYC)
    fig.suptitle("Profils moyens (hors éponge, moyenne des 4 phases) ; bande : N² > 10 % du max", fontsize=10)
    fig.savefig(os.path.join(od, "fig_profils.png"), dpi=110); plt.close(fig)

    # figures des coupes : sigma | vitesse vers l'aval, zoom 0-zmax
    for c in coupes:
        D = dat[c["nom"]]
        s = c["s"] / 1e3
        se = np.r_[s[0] - (s[1] - s[0]) / 2, (s[1:] + s[:-1]) / 2, s[-1] + (s[-1] - s[-2]) / 2]
        top = max(float(np.nanmax(d["eta"])) for d in D) + 0.5
        wz = [d["zc"] >= -zmax for d in D]
        allq = np.concatenate([d["q"][w & np.isfinite(d["q"])] for d, w in zip(D, wz)])
        qlo, qhi = np.nanpercentile(allq, [1, 99]) if allq.size else (0, 1)
        dlev = rond((qhi - qlo) / 12) if qhi > qlo else 1.0
        lev = np.arange(math.floor(qlo / dlev) * dlev, qhi + dlev, dlev)
        ul = max(float(np.nanpercentile(np.abs(np.concatenate([d["u"][w & np.isfinite(d["u"])] for d, w in zip(D, wz)])), 99)), 0.05)
        fig, axs = plt.subplots(len(D), 2, figsize=(14, 2.5 * len(D) + 1.2), sharex=True, sharey=True,
                                constrained_layout=True, squeeze=False)
        pcs = [None, None]
        for r, d in enumerate(D):
            pyc = pycnocline_colonne(d["q"], d["zc"], -zmax, 0.2 * dq[kp])
            for col, (fld, cm, lo, hi) in enumerate(((d["q"], CM_SIG, qlo, qhi), (d["u"], CM_DIV, -ul, ul))):
                ax = axs[r, col]
                pcs[col] = pcolor_colonnes(ax, s, d["zf"], fld, cmap=cm, vmin=lo, vmax=hi)
                X2 = np.broadcast_to(s, d["zc"].shape)
                cs = ax.contour(X2, d["zc"], np.ma.masked_invalid(d["q"]), levels=lev, colors="k", linewidths=.45)
                if col == 0:
                    ax.clabel(cs, cs.levels[::2], fmt=lambda v: fr(v, "{:g}"), fontsize=6)
                ax.plot(s, pyc, ls="--", color=PYC, lw=1.3)
                ax.plot(s, d["eta"], color="#1f4e79", lw=1)
                hb = -Hz[c["jj"], c["ii"]]
                ax.fill_between(se, np.r_[hb, hb[-1]], -zmax - 5, step="post", color="0.6", zorder=3)
                for kk in np.where(~g.interior[c["jj"], c["ii"]])[0]:      # eponge : bande en haut
                    ax.axvspan(se[kk], se[kk + 1], ymin=.94, color="0.55", lw=0, zorder=5)
                if c["nom"] == "talweg":
                    for nm_, kk, cl in [("col", kc, "k")] + ([("cap", kcap, "goldenrod")] if cap else []):
                        ax.axvline(s[kk], color=cl, lw=.8, ls=":")
                        if r == 0:
                            ax.annotate(nm_, (s[kk], 1.0), xycoords=("data", "axes fraction"), xytext=(2, -9),
                                        textcoords="offset points", fontsize=8, color=cl)
                ax.set_ylim(-zmax, top); ax.set_xlim(se[0], se[-1])
                ax.set_title(f"{d['ttl']}", fontsize=9)
        for ax in axs[-1]:
            ax.set_xlabel(c["xlab"])
        for ax in axs[:, 0]:
            ax.set_ylabel("z (m)")
        lab0 = (qnom if sig else "S") + " ; contours : " + ("isopycnes" if sig else "isohalines")
        fig.colorbar(pcs[0], ax=axs[:, 0], shrink=.5, label=lab0)
        fig.colorbar(pcs[1], ax=axs[:, 1], shrink=.5,
                     label="vitesse vers l'aval (m/s) ; rouge : jusant, bleu : flot")
        extra = " ; pointillés : col" + (", cap" if cap else "") if c["nom"] == "talweg" else ""
        fig.suptitle(f"{c['titre'][0].upper() + c['titre'][1:]} — tirets : pycnocline (N² max) ; "
                     f"trait bleu : surface libre ; bande grise en haut : éponge OBCS{extra}", fontsize=10)
        fig.savefig(os.path.join(od, f"fig_coupe_{c['nom']}.png"), dpi=100); plt.close(fig)
        pz = [np.nanmedian(pycnocline_colonne(d["q"], d["zc"], -zmax, 0.2 * dq[kp])) for d in D]
        lines.append(f"Coupe {c['nom']} ({s[-1]:.1f} km) : pycnocline médiane " +
                     ", ".join(f"{fr(-p)} m" for p in pz) + " (4 phases) ; |u aval| max "
                     f"{max(np.nanmax(np.abs(d['u'][w])) for d, w in zip(D, wz)):.2f} m/s dans le zoom")

    # ------------------------------------------------------------------ animation
    if a.gif and R.p2 is not None:
        from matplotlib.animation import FuncAnimation, PillowWriter
        tt = R.it2 * g.dt
        itg = R.it2[(tt >= tdeb - 1e-6) & (tt < tdeb + T_M2 - 1e-6)][::max(1, a.gif_pas)]
        fig = plt.figure(figsize=(7.5, 7.5 * asp + 1.2))
        ax = fig.add_axes([0.1, 0.07, 0.72, 0.8])
        uc, vc, zc, ts = R.surface(itg[0] * g.dt)
        pc = ax.pcolormesh(X, Y, np.where(wet, np.hypot(uc, vc), np.nan), cmap=CM_VIT, vmin=0, vmax=vmax,
                           shading="nearest")
        fond_carte(ax, R, X, Y, H)
        q = fleches(ax, X, Y, uc, vc, st, vref, dxkm)
        ax.quiverkey(q, 0.8, 1.02, vref, f"{fr(vref, '{:g}')} m/s", labelpos="E", coordinates="axes",
                     fontproperties={"size": 8})
        if cap:
            ax.plot((cap[0] + ox) / 1e3, (cap[1] + oy) / 1e3, "*", ms=11, mfc="gold", mec="k", zorder=5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        fig.colorbar(pc, cax=fig.add_axes([0.85, 0.07, 0.025, 0.55]), label=f"vitesse à {fr(zsurf)} m (m/s)")
        ttl = ax.set_title("", fontsize=10, loc="left")
        if serie:
            axm = fig.add_axes([0.84, 0.72, 0.14, 0.15])
            axm.plot((serie[0] - tdeb) / 3600, serie[1], color="#1f4e79", lw=1)
            pt, = axm.plot([], [], "o", color=PYC, ms=5)
            axm.set_xlabel("h", fontsize=7); axm.set_ylabel("η (m)", fontsize=7); axm.tick_params(labelsize=6)
        def frame(n):
            uc, vc, zc, ts = R.surface(itg[n] * g.dt)
            pc.set_array(np.where(wet, np.hypot(uc, vc), np.nan).ravel())
            q.set_UVC(np.ma.masked_invalid(uc[st // 2::st, st // 2::st]), np.ma.masked_invalid(vc[st // 2::st, st // 2::st]))
            ttl.set_text(etiquette(ts, serie, hpm))
            if serie:
                pt.set_data([(ts - tdeb) / 3600], [np.interp(ts, serie[0], serie[1])])
            return pc, q
        FuncAnimation(fig, frame, frames=len(itg), blit=False).save(
            os.path.join(od, "courants_surface.gif"), writer=PillowWriter(fps=8), dpi=80)
        plt.close(fig)
        lines.append(f"Animation : {len(itg)} images, une toutes les {(itg[1] - itg[0]) * g.dt if len(itg) > 1 else 0:.0f} s")

    lines.append(f"Figures : {od}")
    open(os.path.join(od, "cartes_coupes.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
