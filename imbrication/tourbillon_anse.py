#!/usr/bin/env python3
"""
Tourbillon de l'Anse-de-Roche et mouillage ADCP : comparaison aux observations de l'utilisateur.

Observations (derive de glace, mars 2019 et 2021) : cyclone qui se detache de la cote au sud de
l'Anse-de-Roche en fin de jusant, centre ~48,200 N 69,882 O (~2,5 km en amont du col du 2e seuil),
diametre ~1 km (presque toute la largeur du fjord), zeta/f ~20-30, vitesses 0,4-0,8 m/s, visible
autour de la basse mer. Ondes internes (ADCP, 4 juillet 2018, 48,1980 N 69,87735 O) : trains
d'ondes de depression, ~10 m d'amplitude, periode ~5-6 min, w +-0,1 m/s dans les 15 m du haut.

  1. Cartes de surface (niveau --niveau de la liste 2D) dans la fenetre des figures d'observation :
     zeta/f (meme echelle +-20) avec fleches, et divergence/f, a des heures relatives a la basse
     mer (BM) du mouillage, sur le dernier cycle analyse.
  2. Detection du cyclone a chaque instantane 2D : coeur Okubo-Weiss cyclonique (W < -0,2 sigma_W,
     zeta > 0) dont le centre tombe a moins de --rayon-cible de la cible ; circulation, aire,
     diametre equivalent, zeta/f max, vitesse ~ Gamma / (pi D), centre en lon/lat, rapport
     diametre / largeur du fjord. Aussi : le plus fort coeur anticyclonique de la fenetre (dipole).
  3. Verrouillage de phase : heure (par rapport a la BM) du maximum de circulation, cycle par cycle.
  4. Mouillage virtuel (instantanes 3D) : diagramme temps-profondeur de w et des isohalines,
     profondeur de la pycnocline, c1 (mode 1, WKB). A 930 s, les ondes de 5-6 min ne sont pas vues :
     cela situe la stratification et la phase de maree, pas les ondes.

Memoire : un instantane 2D ou 3D a la fois.

Usage
  python3 imbrication/tourbillon_anse.py ~/runs/enfantA_v3/run --origine-utm 430800 5330800 [--noms A]
  python3 imbrication/tourbillon_anse.py RUN_A RUN_B --noms A B [--skip 1] [--pipeline DIR]
Origine du run (lon/lat -> repere du run) : --origine-utm, ou grids.npz de --pipeline + enfant.json.
Sinon : --fenetre-xy, --cible-xy, --mouillage-xy (m, repere du run).
Ecrit dans DERNIER_RUN/diag : anse.txt, anse_detection.csv, fig_anse_vorticite_<nom>.png,
fig_anse_divergence_<nom>.png, fig_anse_series.png, fig_anse_mouillage.png.
"""
import argparse, csv, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sagdiag"))
import sagdiag as sd

T_M2 = 44640.0
ICI = os.path.dirname(os.path.abspath(__file__))
# Valeurs par defaut : figures d'observation de l'utilisateur (vort_3pan, isw_2018074_2)
FENETRE = (-69.917, -69.850, 48.167, 48.222)     # lon0 lon1 lat0 lat1
CIBLE = (-69.882, 48.200)                         # centre observe du cyclone
MOUILLAGE = (-69.87735, 48.1980)                  # ADCP du 4 juillet 2018


_A, _F, _K0, _LON0 = 6378137.0, 1 / 298.257223563, 0.9996, -69.0     # WGS84, UTM 19N
_E2 = _F * (2 - _F); _EP2 = _E2 / (1 - _E2)


def _arc(phi):
    e2, e4, e6 = _E2, _E2 ** 2, _E2 ** 3
    return _A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
                 + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi) - 35 * e6 / 3072 * np.sin(6 * phi))


def lonlat_utm(lon, lat):
    """lon/lat (WGS84) -> UTM 19N (m), formules de Snyder (USGS 1987), ~0,1 m pres dans la zone."""
    phi = np.radians(np.asarray(lat, float)); s, c = np.sin(phi), np.cos(phi)
    N = _A / np.sqrt(1 - _E2 * s ** 2); T = np.tan(phi) ** 2; C = _EP2 * c ** 2
    A = c * np.radians(np.asarray(lon, float) - _LON0)
    x = _K0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * _EP2) * A ** 5 / 120) + 500000.0
    y = _K0 * (_arc(phi) + N * np.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                             + (61 - 58 * T + T ** 2 + 600 * C - 330 * _EP2) * A ** 6 / 720))
    return x, y


def utm_lonlat(x, y):
    """UTM 19N (m) -> lon/lat : inverse de lonlat_utm (Snyder)."""
    mu = (np.asarray(y, float) / _K0) / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + 151 * e1 ** 3 / 96 * np.sin(6 * mu) + 1097 * e1 ** 4 / 512 * np.sin(8 * mu))
    s, c = np.sin(p1), np.cos(p1)
    C1 = _EP2 * c ** 2; T1 = np.tan(p1) ** 2; N1 = _A / np.sqrt(1 - _E2 * s ** 2)
    R1 = _A * (1 - _E2) / (1 - _E2 * s ** 2) ** 1.5; D = (np.asarray(x, float) - 500000.0) / (N1 * _K0)
    lat = p1 - N1 * np.tan(p1) / R1 * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * _EP2) * D ** 4 / 24
                                        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * _EP2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * _EP2 + 24 * T1 ** 2) * D ** 5 / 120) / c
    return _LON0 + np.degrees(lon), np.degrees(lat)


class Repere:
    """lon/lat <-> x, y (m) du run = UTM 19N - origine (coin sud-ouest de l'enfant). Origine :
    --origine-utm, sinon grids.npz (--pipeline) + enfant.json a cote du dossier run (comme
    seuil_nh.lonlat_vers_run et cartes_coupes.py)."""
    def __init__(self, run, pipeline, origine=None):
        if origine:
            self.ox, self.oy = origine
        else:
            gr = np.load(os.path.join(pipeline, "grids.npz"))
            c = json.load(open(os.path.join(run, "..", "enfant.json")))
            self.ox = float(gr["parent_x0"]) + c["x0"]
            self.oy = float(gr["parent_y0"]) + c["y0"]

    def xy(self, lon, lat):
        x, y = lonlat_utm(lon, lat)
        return x - self.ox, y - self.oy

    def lonlat(self, x, y):
        return utm_lonlat(np.asarray(x) + self.ox, np.asarray(y) + self.oy)


def divergence(u, v, g, k, wet):
    """Divergence horizontale aux centres (s-1) au niveau k du modele."""
    fu = u * g.DYG * g.hFacW[k]
    fv = v * g.DXG * g.hFacS[k]
    d = np.full(u.shape, np.nan, np.float32)
    d[:-1, :-1] = ((fu[:-1, 1:] - fu[:-1, :-1]) + (fv[1:, :-1] - fv[:-1, :-1])) / (g.RAC[:-1, :-1] * np.maximum(g.hFacC[k, :-1, :-1], 1e-3))
    d[~wet] = np.nan
    return d


def largeur(wet, j, i, dx):
    """Largeur mouillee (m) a travers (j, i) : min des longueurs contigues en x et en y."""
    def run(line, p):
        if not line[p]:
            return 0
        a = p
        while a > 0 and line[a - 1]:
            a -= 1
        b = p
        while b < len(line) - 1 and line[b + 1]:
            b += 1
        return b - a + 1
    return min(run(wet[j, :], i), run(wet[:, i], j)) * dx


def coeurs(zc, ow, wet, f0):
    """Coeurs Okubo-Weiss etiquetes (cycloniques, anticycloniques)."""
    sig = np.nanstd(ow[wet])
    base = wet & np.isfinite(ow) & np.isfinite(zc) & (ow < -0.2 * sig) & (np.abs(zc) > 0.2 * abs(f0))
    lp, npl = ndimage.label(base & (zc > 0))
    lm, nm = ndimage.label(base & (zc < 0))
    return (lp, npl), (lm, nm)


def mesure(lab, n, zc, g, f0, filtre=None):
    """Plus forte composante (|Gamma|) : dict ou None. filtre(xc, yc) -> bool."""
    best = None
    if n == 0:
        return None
    z = np.nan_to_num(zc)
    gam = ndimage.sum(z * g.RAC, lab, index=np.arange(1, n + 1))
    aire = ndimage.sum(g.RAC, lab, index=np.arange(1, n + 1))
    wx = ndimage.sum(z * g.RAC * g.XC, lab, index=np.arange(1, n + 1))
    wy = ndimage.sum(z * g.RAC * g.YC, lab, index=np.arange(1, n + 1))
    for m in np.argsort(-np.abs(gam)):
        xc, yc = wx[m] / gam[m], wy[m] / gam[m]
        if filtre is None or filtre(xc, yc):
            cc = lab == (m + 1)
            D = 2 * math.sqrt(aire[m] / math.pi)
            best = dict(gam=float(gam[m]), aire=float(aire[m]), D=D, x=float(xc), y=float(yc),
                        romax=float(np.nanmax(np.abs(zc[cc])) / abs(f0)),
                        ro90=float(np.nanpercentile(np.abs(zc[cc]), 90) / abs(f0)),
                        V=float(abs(gam[m]) / (math.pi * D)), masque=cc)
            break
    return best


def heure_rel(t, tbm):
    """Heure relative a la basse mer, ramenee dans [-T/2, T/2)."""
    return ((t - tbm + T_M2 / 2) % T_M2 - T_M2 / 2) / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="dossiers de run (A puis B, meme grille)")
    ap.add_argument("--noms", nargs="+", default=None, help="noms courts (fichiers et legendes)")
    ap.add_argument("--pipeline", default=os.path.join(ICI, "..", "pipeline_grille", "output_mini"),
                    help="sortie du pipeline (grids.npz : origine UTM du parent)")
    ap.add_argument("--origine-utm", nargs=2, type=float, metavar=("X0", "Y0"),
                    help="coin sud-ouest du run en UTM 19N (enfant 50 m v3 : 430800 5330800)")
    ap.add_argument("--fenetre", nargs=4, type=float, default=FENETRE, metavar=("LON0", "LON1", "LAT0", "LAT1"))
    ap.add_argument("--cible", nargs=2, type=float, default=CIBLE, metavar=("LON", "LAT"))
    ap.add_argument("--mouillage", nargs=2, type=float, default=MOUILLAGE, metavar=("LON", "LAT"))
    ap.add_argument("--fenetre-xy", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--cible-xy", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--mouillage-xy", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--rayon-cible", type=float, default=1200.0, help="rayon de recherche du cyclone (m)")
    ap.add_argument("--niveau", type=int, default=0, help="niveau de la liste 2D (0, 1, 2 : ~1, 10, 30 m)")
    ap.add_argument("--skip", type=int, default=1, help="cycles complets ignores au debut (demarrage)")
    ap.add_argument("--heures", nargs=2, type=float, default=(-4.0, 1.5), help="cartes : de H0 a H1 (h, BM = 0)")
    ap.add_argument("--pas", type=float, default=0.5, help="cartes : pas (h)")
    ap.add_argument("--seuil-patch", type=float, default=5.0, help="taille de la zone zeta > S f autour du coeur")
    ap.add_argument("--zlim", type=float, default=20.0, help="echelle de zeta/f (observations : 20)")
    a = ap.parse_args()
    runs = a.runs
    noms = a.noms or [os.path.basename(os.path.normpath(r)) for r in runs]

    g = sd.Grid(runs[0])
    for r in runs[1:]:
        assert sd.Grid(r).hFacC.shape == g.hFacC.shape, "les runs doivent avoir la meme grille"
    info = sd.scan_run(runs[0])
    pre3, pre2, pre_eta, _ = sd.guess_prefixes(info, g.nz)
    if not pre2 or not pre_eta:
        sys.exit("il faut la liste 2D (lev2D : UVEL, VVEL) et eta2D (ETAN)")
    levs = sd.diag_levels(runs[0], pre2)
    li = a.niveau
    kmod = (levs[li] - 1) if levs else li
    wet = g.maskC[kmod]
    f0 = float(np.nanmean(g.fC))

    # --- reperes
    rep = None
    if not (a.fenetre_xy and a.cible_xy and a.mouillage_xy):
        try:
            rep = Repere(runs[-1], a.pipeline, a.origine_utm)
        except (OSError, KeyError) as e:                # grids.npz ou enfant.json introuvable
            sys.exit(f"origine du run inconnue ({e}) : donner --origine-utm, ou --fenetre-xy, --cible-xy, --mouillage-xy")
    if a.fenetre_xy:
        box = tuple(a.fenetre_xy)
    else:
        lo0, lo1, la0, la1 = a.fenetre
        cx, cy = rep.xy([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
        box = (float(cx.min()), float(cx.max()), float(cy.min()), float(cy.max()))
    xt, yt = a.cible_xy if a.cible_xy else map(float, rep.xy(*a.cible))
    xm, ym = a.mouillage_xy if a.mouillage_xy else map(float, rep.xy(*a.mouillage))
    ib = (g.XC[0] >= box[0]) & (g.XC[0] <= box[1])
    jb = (g.YC[:, 0] >= box[2]) & (g.YC[:, 0] <= box[3])
    if ib.sum() < 3 or jb.sum() < 3:
        sys.exit(f"fenetre hors du domaine : x {box[0]:.0f}-{box[1]:.0f}, y {box[2]:.0f}-{box[3]:.0f} m ; "
                 f"domaine x {g.XC[0, 0]:.0f}-{g.XC[0, -1]:.0f}, y {g.YC[0, 0]:.0f}-{g.YC[-1, 0]:.0f} m")
    fen = np.zeros(wet.shape, bool); fen[np.ix_(jb, ib)] = True
    fen &= wet & g.interior
    dm = np.where(g.maskC[0], np.hypot(g.XC - xm, g.YC - ym), np.inf)
    jm, im = np.unravel_index(np.argmin(dm), dm.shape)          # colonne mouillee la plus proche
    dx = g.dx_mean
    if not g.interior[jm, im] or np.hypot(g.XC - xt, g.YC - yt)[~g.interior].min() < a.rayon_cible:
        print("Attention : la cible ou le mouillage touche l'eponge OBCS")

    # --- temps : cycles complets, basse mer au mouillage
    it2 = np.array(sd.list_iters(runs[0], pre2))
    for r in runs[1:]:
        it2 = np.intersect1d(it2, sd.list_iters(r, pre2))
    t2 = it2 * g.dt
    dto = float(np.median(np.diff(t2)))
    cyc = np.floor(t2 / T_M2 + 1e-9).astype(int)
    cycles = [c for c in np.unique(cyc) if (cyc == c).sum() >= 0.95 * T_M2 / dto]
    cycles = cycles[a.skip:]
    if not cycles:
        sys.exit("aucun cycle complet apres --skip")
    ite = np.array(sd.list_iters(runs[0], pre_eta))
    te = ite * g.dt
    eta_m = np.array([sd.read_fields(os.path.join(runs[0], f"{pre_eta}.{it:010d}"))[0]["ETAN"].reshape(-1, g.ny, g.nx)[0, jm, im]
                      if c in cycles else np.nan for it, c in zip(ite, np.floor(te / T_M2 + 1e-9).astype(int))])
    tbm = {}
    for c in cycles:
        s = (np.floor(te / T_M2 + 1e-9).astype(int) == c) & np.isfinite(eta_m)
        tbm[c] = float(te[s][np.argmin(eta_m[s])])

    # --- detection a chaque instantane 2D
    filtre = lambda x, y: math.hypot(x - xt, y - yt) <= a.rayon_cible
    det = {nm: [] for nm in noms}
    dernier = cycles[-1]
    heures = np.arange(a.heures[0], a.heures[1] + 1e-6, a.pas)
    tcart = [tbm[dernier] + h * 3600 for h in heures]
    itc = [int(it2[np.argmin(np.abs(t2 - tc))]) if np.min(np.abs(t2 - tc)) <= dto else None for tc in tcart]
    cartes = {nm: {} for nm in noms}
    for run, nm in zip(runs, noms):
        for it, t, c in zip(it2, t2, cyc):
            if c not in cycles:
                continue
            d, _ = sd.read_fields(os.path.join(run, f"{pre2}.{it:010d}"))
            u, v = d["UVEL"][li].astype(np.float32), d["VVEL"][li].astype(np.float32)
            kin = sd.kinematics(u, v, g, wet)
            zc, ow = kin["zeta_c"], kin["ow"]
            dv = divergence(u, v, g, kmod, wet)
            (lp, npl), (lm, nmm) = coeurs(np.where(fen, zc, np.nan), np.where(fen, ow, np.nan), fen, f0)
            cy_ = mesure(lp, npl, zc, g, f0, filtre)
            an_ = mesure(lm, nmm, zc, g, f0)
            row = dict(cycle=int(c), t=float(t), h=heure_rel(t, tbm[c]),
                       div_rms=float(np.sqrt(np.nanmean(dv[fen] ** 2))) / abs(f0),
                       umax=float(np.nanmax(np.hypot(sd.u2c(u), sd.v2c(v))[fen])))
            if cy_:
                jc = int(np.argmin(np.abs(g.YC[:, 0] - cy_["y"]))); ic = int(np.argmin(np.abs(g.XC[0] - cy_["x"])))
                L = largeur(wet, jc, ic, dx)
                # « patch » : zone zeta > --seuil-patch f qui contient le coeur (comme la tache rouge observee)
                lab5, _ = ndimage.label(fen & (np.nan_to_num(zc) > a.seuil_patch * abs(f0)))
                ids = np.unique(lab5[cy_["masque"]]); ids = ids[ids > 0]
                pm = np.isin(lab5, ids) if ids.size else cy_["masque"]
                A5 = float((g.RAC * pm).sum())
                row.update(gam=cy_["gam"], D=cy_["D"], romax=cy_["romax"], ro90=cy_["ro90"], V=cy_["V"],
                           x=cy_["x"], y=cy_["y"], largeur=L, frac=cy_["D"] / L if L else np.nan,
                           gam5=float((np.nan_to_num(zc) * g.RAC * pm).sum()), D5=2 * math.sqrt(A5 / math.pi))
            else:
                pm = None
                row.update(gam=0.0, D=0.0, romax=0.0, ro90=0.0, V=0.0, x=np.nan, y=np.nan, largeur=np.nan, frac=np.nan,
                           gam5=0.0, D5=0.0)
            row.update(gam_anti=an_["gam"] if an_ else 0.0, x_anti=an_["x"] if an_ else np.nan,
                       y_anti=an_["y"] if an_ else np.nan)
            det[nm].append(row)
            if c == dernier and it in itc:
                cartes[nm][int(it)] = (zc / f0, dv / abs(f0), u, v, cy_["masque"] if cy_ else None, pm)

    od = os.path.join(runs[-1], "diag"); os.makedirs(od, exist_ok=True)

    # --- cartes
    X, Y = g.XC[0][ib] / 1e3, g.YC[:, 0][jb] / 1e3
    st = max(1, int(round(150.0 / dx)))                        # une fleche tous les ~150 m
    asp = (box[3] - box[2]) / (box[1] - box[0])
    phs = [(h, it) for h, it in zip(heures, itc) if it is not None]
    ncol = min(4, len(phs)); nrow = math.ceil(len(phs) / ncol)
    for nm in noms:
        for quoi, fn, lim, lab in (("z", "vorticite", a.zlim, f"ζ/f à {-g.RC[kmod]:.0f} m ; noir : cœur cyclonique détecté (tirets : ζ > {a.seuil_patch:g} f)"),
                                   ("d", "divergence", None, f"divergence / f à {-g.RC[kmod]:.0f} m")):
            if quoi == "d":
                vals = np.concatenate([cartes[nm][it][1][np.ix_(jb, ib)].ravel() for _, it in phs])
                lim = max(float(np.nanpercentile(np.abs(vals), 99)), 0.1)
            fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 1.2, 3.6 * asp * nrow + 0.8),
                                    squeeze=False, constrained_layout=True)
            for ax in axs.ravel()[len(phs):]:
                ax.set_visible(False)
            for ax, (h, it) in zip(axs.ravel(), phs):
                zr, dvr, u, v, cc, pm = cartes[nm][it]
                q = (zr if quoi == "z" else dvr)[np.ix_(jb, ib)]
                pcm = ax.pcolormesh(X, Y, q, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
                ax.contour(X, Y, wet[np.ix_(jb, ib)].astype(float), [0.5], colors="0.3", linewidths=.6)
                if quoi == "z":
                    uc, vc = sd.u2c(u)[np.ix_(jb, ib)], sd.v2c(v)[np.ix_(jb, ib)]
                    ax.quiver(X[::st], Y[::st], uc[::st, ::st], vc[::st, ::st], scale=12, width=.004)
                    if cc is not None:
                        ax.contour(X, Y, cc[np.ix_(jb, ib)].astype(float), [0.5], colors="k", linewidths=.9)
                        ax.contour(X, Y, pm[np.ix_(jb, ib)].astype(float), [0.5], colors="k", linewidths=.7, linestyles="--")
                ax.plot(xt / 1e3, yt / 1e3, "g+", ms=9); ax.plot(xm / 1e3, ym / 1e3, "k^", ms=5)
                ax.set_aspect(1); ax.set_xlim(box[0] / 1e3, box[1] / 1e3); ax.set_ylim(box[2] / 1e3, box[3] / 1e3)
                ax.set_title(f"{nm} — BM {h:+.1f} h", fontsize=9)
            fig.colorbar(pcm, ax=axs, label=lab + " ; + cible ; ▲ mouillage", shrink=.5)
            fig.savefig(os.path.join(od, f"fig_anse_{fn}_{nm}.png"), dpi=90); plt.close(fig)

    # --- series et verrouillage de phase
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    resume = {}
    for nm in noms:
        resume[nm] = []
        for c in cycles:
            r = [x for x in det[nm] if x["cycle"] == c]
            h = np.array([x["h"] for x in r]); o = np.argsort(h); h = h[o]
            gam = np.array([x["gam"] for x in r])[o]
            ls = "-" if c == dernier else ":"
            ax[0].plot(h, gam / 1e3, ls, label=f"{nm} cycle {c}")
            ax[1].plot(h, np.array([x["D"] for x in r])[o], ls)
            ax[2].plot(h, np.array([x["romax"] for x in r])[o], ls)
            ax[3].plot(h, np.array([x["div_rms"] for x in r])[o], ls)
            k = int(np.argmax(gam))
            pres = gam > 0.25 * gam[k]
            lo = k
            while lo > 0 and pres[lo - 1]:
                lo -= 1
            hi = k
            while hi < len(h) - 1 and pres[hi + 1]:
                hi += 1
            rk = r[o[k]]
            resume[nm].append(dict(cycle=c, h=h[k], duree=h[hi] - h[lo] if gam[k] > 0 else 0.0, **{x: rk[x] for x in
                              ("gam", "D", "gam5", "D5", "romax", "ro90", "V", "x", "y", "largeur", "frac", "gam_anti", "x_anti", "y_anti", "umax")}))
    ax[0].set_ylabel("Γ cyclone (10³ m²/s)"); ax[1].set_ylabel("diamètre équivalent (m)")
    ax[2].set_ylabel("ζ/f max du cœur"); ax[3].set_ylabel("divergence rms / f (fenêtre)")
    ax[3].set_xlabel("heure par rapport à la basse mer au mouillage (h)")
    for x in ax:
        x.axvline(0, color="k", lw=.6, ls=":")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(od, "fig_anse_series.png"), dpi=100); plt.close(fig)

    with open(os.path.join(od, "anse_detection.csv"), "w", newline="") as fh:
        cols = ["run", "cycle", "t", "h", "gam", "D", "gam5", "D5", "romax", "ro90", "V", "x", "y", "largeur", "frac",
                "gam_anti", "x_anti", "y_anti", "div_rms", "umax"]
        w = csv.writer(fh); w.writerow(cols)
        for nm in noms:
            for x in det[nm]:
                w.writerow([nm] + [f"{x[k]:.6g}" if isinstance(x[k], float) else x[k] for k in cols[1:]])

    # --- mouillage virtuel (3D, dernier cycle)
    moor = None
    if pre3:
        it3 = np.array(sd.list_iters(runs[0], pre3)); t3 = it3 * g.dt
        s3 = np.floor(t3 / T_M2 + 1e-9).astype(int) == dernier
        it3, t3 = it3[s3], t3[s3]
        kk = np.where(g.maskC[:, jm, im])[0]
        zc_ = -g.RC[kk]
        avec_rho = "RHOAnoma" in info[pre3]["fields"]
        moor = {}
        for run, nm in zip(runs, noms):
            W, S, R = [], [], []
            for it in it3:
                base = os.path.join(run, f"{pre3}.{it:010d}")
                d, _ = sd.read_fields(base)
                W.append(d["WVEL"][kk, jm, im]); S.append(d["SALT"][kk, jm, im])
                if avec_rho:
                    R.append(d["RHOAnoma"][kk, jm, im])
            W, S = np.array(W).T, np.array(S).T
            p = dict(W=W, S=S)
            if avec_rho and kk.size > 2:
                N2 = np.clip(sd.G / g.rho0 * np.diff(np.array(R).T, axis=0) / np.diff(zc_)[:, None], 0, None)
                N2m = N2.mean(axis=1)
                p["zpyc"] = float(0.5 * (zc_[:-1] + zc_[1:])[np.argmax(N2m)])
                p["Nmax"] = float(np.sqrt(N2m.max()))
                p["c1"] = float((np.sqrt(N2m) * np.diff(zc_)).sum() / math.pi)
            moor[nm] = p
        h3 = heure_rel(t3, tbm[dernier]); o = np.argsort(h3)
        zf = np.r_[0.0, np.cumsum(g.DRF)][: kk.size + 1]
        zlim = min(60.0, zf[-1])
        fig, axs = plt.subplots(len(noms) + 1, 1, figsize=(10, 3 + 3 * len(noms)), sharex=True)
        se = heure_rel(te, tbm[dernier]); oe = np.argsort(se); ok = np.isfinite(eta_m[oe])
        axs[0].plot(se[oe][ok], eta_m[oe][ok], "k"); axs[0].set_ylabel("η au mouillage (m)")
        wl = max(float(np.nanpercentile(np.abs(np.concatenate([moor[nm]["W"].ravel() for nm in noms])), 99)), 1e-4)
        he = np.r_[h3[o][0] - (h3[o][1] - h3[o][0]) / 2, (h3[o][1:] + h3[o][:-1]) / 2, h3[o][-1] + (h3[o][-1] - h3[o][-2]) / 2]
        for ax, nm in zip(axs[1:], noms):
            pc = ax.pcolormesh(he, -zf, moor[nm]["W"][:, o], cmap="RdBu_r", vmin=-wl, vmax=wl, shading="flat")
            ax.contour(h3[o], -zc_, moor[nm]["S"][:, o], levels=np.arange(10, 34, 2.0), colors="k", linewidths=.5)
            ax.set_ylim(-zlim, 0); ax.set_ylabel("z (m)"); ax.set_title(f"{nm} — mouillage virtuel (H = {g.Depth[jm, im]:.0f} m)", fontsize=9)
            fig.colorbar(pc, ax=ax, label="w* (m/s) ; contours : S")
        for ax in axs:
            ax.axvline(0, color="k", lw=.6, ls=":")
        axs[-1].set_xlabel(f"heure par rapport à la basse mer (h) ; instantanés 3D toutes les {t3[1] - t3[0]:.0f} s")
        fig.tight_layout(); fig.savefig(os.path.join(od, "fig_anse_mouillage.png"), dpi=100); plt.close(fig)

    # --- resume
    ll = (lambda x, y: rep.lonlat(x, y)) if rep else None
    def pos(x, y):
        if not np.isfinite(x):
            return "—"
        s = f"x={x:.0f} y={y:.0f} m"
        if ll:
            lo, la = ll(x, y); s += f" ({la:.4f} N, {abs(lo):.4f} O)"
        return s
    lines = [f"Fenetre x {box[0]:.0f}-{box[1]:.0f}, y {box[2]:.0f}-{box[3]:.0f} m ; niveau {-g.RC[kmod]:.1f} m ; maille {dx:.0f} m ; f = {f0:.3e} s-1",
             f"Cible {pos(xt, yt)}, rayon de recherche {a.rayon_cible:.0f} m ; mouillage {pos(float(g.XC[jm, im]), float(g.YC[jm, im]))}, H = {g.Depth[jm, im]:.0f} m",
             f"Cycles analyses : {', '.join(str(c) for c in cycles)} (--skip {a.skip}) ; basse mer au mouillage : "
             + ", ".join(f"cycle {c} t = {tbm[c]:.0f} s (phase {tbm[c] % T_M2 / T_M2 * 360:.0f}°)" for c in cycles),
             "Observations (glace, mars) : cyclone ~48,200 N 69,882 O, D ~1 km (presque toute la largeur), zeta/f ~20-30, "
             "V 0,4-0,8 m/s, Gamma ~2e3 m2/s, autour de la basse mer"]
    for nm in noms:
        for r in resume[nm]:
            if r["gam"] <= 0:
                lines.append(f"{nm:10s} cycle {r['cycle']} : aucun coeur cyclonique a moins de {a.rayon_cible:.0f} m de la cible")
                continue
            lines.append(f"{nm:10s} cycle {r['cycle']} : Gamma max {r['gam'] / 1e3:.2f}e3 m2/s a BM {r['h']:+.1f} h, "
                         f"D {r['D']:.0f} m ({100 * r['frac']:.0f} % de la largeur {r['largeur']:.0f} m), zeta/f max {r['romax']:.1f} "
                         f"(90e centile {r['ro90']:.1f}), V ~ {r['V']:.2f} m/s, present {r['duree']:.1f} h (Gamma > 25 % du max), "
                         f"centre {pos(r['x'], r['y'])} ; |u| max fenetre {r['umax']:.2f} m/s")
            lines.append(f"{'':10s}   zone zeta > {a.seuil_patch:g} f autour du coeur : D {r['D5']:.0f} m, Gamma {r['gam5'] / 1e3:.2f}e3 m2/s")
            if r["gam_anti"] < 0:
                lines.append(f"{'':10s}   anticyclone le plus fort au meme instant : Gamma {r['gam_anti'] / 1e3:.2f}e3 m2/s, "
                             f"centre {pos(r['x_anti'], r['y_anti'])}")
        hs = [r["h"] for r in resume[nm] if r["gam"] > 0]
        if len(hs) > 1:
            lines.append(f"{nm:10s} verrouillage de phase : heure du max {min(hs):+.1f} a {max(hs):+.1f} h (ecart {max(hs) - min(hs):.1f} h)")
    if moor:
        for nm in noms:
            p = moor[nm]
            lines.append(f"{nm:10s} mouillage : w* rms {1e3 * np.sqrt(np.nanmean(p['W'] ** 2)):.2f} mm/s, |w*| max {1e3 * np.nanmax(np.abs(p['W'])):.1f} mm/s"
                         + (f" ; pycnocline {p['zpyc']:.1f} m (N max {p['Nmax']:.3f} s-1), c1 {p['c1']:.2f} m/s" if "c1" in p else ""))
    lines.append(f"Figures : {od}/fig_anse_vorticite_*.png, fig_anse_divergence_*.png, fig_anse_series.png, fig_anse_mouillage.png ; "
                 f"detection : anse_detection.csv")
    open(os.path.join(od, "anse.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
