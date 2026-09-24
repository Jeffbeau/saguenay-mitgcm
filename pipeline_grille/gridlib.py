"""
gridlib.py — Outils communs : projection, grilles, agrégation 10 m -> grille, E/S MITgcm.
"""
import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

import config as C

_TO_UTM = Transformer.from_crs(C.CRS_SRC, C.CRS_MODEL, always_xy=True)
_TO_GEO = Transformer.from_crs(C.CRS_MODEL, C.CRS_SRC, always_xy=True)


def to_utm(lon, lat):
    return _TO_UTM.transform(lon, lat)


def to_geo(x, y):
    return _TO_GEO.transform(x, y)


# ---------------------------------------------------------------------------
# Grille verticale
# ---------------------------------------------------------------------------
def vertical_grid(hmax=None):
    """NSURF couches de DZ_SURF puis progression géométrique ; r trouvé pour sum = hmax."""
    hmax = hmax or C.HMAX
    nstr = C.NR - C.NSURF
    target = hmax - C.NSURF * C.DZ_SURF

    def total(r):
        return C.DZ_SURF * sum(r ** (k + 1) for k in range(nstr))

    lo, hi = 1.0 + 1e-9, 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if total(mid) < target else (lo, mid)
    r = 0.5 * (lo + hi)
    dz = np.r_[np.full(C.NSURF, C.DZ_SURF), C.DZ_SURF * r ** (np.arange(nstr) + 1)]
    dz = np.round(dz, 3)
    dz[-1] += hmax - dz.sum()                      # total exact
    ratio = dz[1:] / dz[:-1]
    assert ratio.max() <= C.R_MAX_ALLOWED + 1e-6, f"dz ratio {ratio.max():.3f} trop grand"
    return dz, r


def rf_from_dz(dz):
    return np.r_[0.0, -np.cumsum(dz)]              # faces, négatives vers le bas


# ---------------------------------------------------------------------------
# Grilles horizontales
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self, name, x0, y0, dx, nx, ny):
        self.name, self.x0, self.y0, self.dx, self.nx, self.ny = name, x0, y0, dx, nx, ny
        self.xg = x0 + dx * np.arange(nx + 1)       # faces
        self.yg = y0 + dx * np.arange(ny + 1)
        self.xc = 0.5 * (self.xg[1:] + self.xg[:-1])
        self.yc = 0.5 * (self.yg[1:] + self.yg[:-1])
        X, Y = np.meshgrid(self.xc, self.yc)
        self.lonc, self.latc = to_geo(X, Y)

    @property
    def x1(self):
        return self.xg[-1]

    @property
    def y1(self):
        return self.yg[-1]

    def __repr__(self):
        return (f"Grid({self.name}: dx={self.dx:g} m, {self.nx}x{self.ny}, "
                f"x=[{self.x0:.0f},{self.x1:.0f}] y=[{self.y0:.0f},{self.y1:.0f}])")


def snap_in(a, b, step, origin=0.0):
    return (origin + np.ceil((a - origin) / step) * step,
            origin + np.floor((b - origin) / step) * step)


def make_grid(name, x0, x1, y0, y1, dx, align=None):
    """Bornes arrondies vers l'intérieur sur la grille SNAP, ou sur les faces d'un parent
    (align = (x0p, y0p, dxp)) : obligatoire quand dx parent != SNAP (profil mini)."""
    ox, oy, snap = (0.0, 0.0, C.SNAP) if align is None else align
    x0, x1 = snap_in(x0, x1, snap, ox)
    y0, y1 = snap_in(y0, y1, snap, oy)
    nx = int(round((x1 - x0) / dx)); ny = int(round((y1 - y0) / dx))
    # Nx, Ny multiples de PAD_MULT*(snap/dx) pour garder les bornes sur la grille snap
    m = C.PAD_MULT * int(round(snap / dx)) if dx < snap else C.PAD_MULT
    nx -= nx % m; ny -= ny % m
    return Grid(name, x0, y0, dx, nx, ny)


# ---------------------------------------------------------------------------
# Agrégation des pixels NONNA (≈7.4 x 11.1 m) dans les cellules du modèle
# Chaque pixel est sur-échantillonné SUB x SUB pour lisser le comptage.
# ---------------------------------------------------------------------------
def bin_to_grid(g, H10, water, covered, lon, lat, sub=2, chunk=200):
    nc = g.nx * g.ny
    s_h = np.zeros(nc); n_w = np.zeros(nc); n_all = np.zeros(nc); n_cov = np.zeros(nc)
    dlon = lon[1] - lon[0]; dlat = lat[1] - lat[0]
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for j0 in range(0, len(lat), chunk):
        la = lat[j0:j0 + chunk]
        for oy in offs:
            for ox in offs:
                LON, LAT = np.meshgrid(lon + ox * dlon, la + oy * dlat)
                x, y = to_utm(LON, LAT)
                i = np.floor((x - g.x0) / g.dx).astype(np.int64)
                j = np.floor((y - g.y0) / g.dx).astype(np.int64)
                ok = (i >= 0) & (i < g.nx) & (j >= 0) & (j < g.ny)
                idx = (j * g.nx + i)[ok]
                w = water[j0:j0 + chunk][ok]; cv = covered[j0:j0 + chunk][ok]
                h = H10[j0:j0 + chunk][ok]
                n_all += np.bincount(idx, minlength=nc)
                n_cov += np.bincount(idx, weights=cv, minlength=nc)
                n_w += np.bincount(idx, weights=w, minlength=nc)
                s_h += np.bincount(idx, weights=np.where(w, h, 0.0), minlength=nc)
    shp = (g.ny, g.nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        Hm = (s_h / n_w).reshape(shp)
        wf = (n_w / n_cov).reshape(shp)
        cf = (n_cov / n_all).reshape(shp)
    cf[n_all.reshape(shp) == 0] = 0.0
    return Hm, np.nan_to_num(wf), cf


def sample_to_grid(g, H, water, covered, lon, lat, sub=None):
    """Produit grossier (NONNA-100) : échantillonnage bilinéaire sur sub x sub points par
    cellule. H est d'abord prolongé sous la terre par plus proche voisin (pas de contamination
    côtière), le masque d'eau est interpolé puis seuillé à 0,5 point par point."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    sub = sub or max(2, int(round(g.dx / 25.0)))
    idx = ndi.distance_transform_edt(~water, return_distances=False, return_indices=True)
    Hx = np.nan_to_num(H[tuple(idx)], nan=0.0)
    la, lo = lat, lon
    if la[0] > la[-1]:
        la = la[::-1]; Hx = Hx[::-1]; water = water[::-1]; covered = covered[::-1]
    fH = RGI((la, lo), Hx, bounds_error=False, fill_value=np.nan)
    fW = RGI((la, lo), water.astype(float), bounds_error=False, fill_value=0.0)
    fC = RGI((la, lo), covered.astype(float), method="nearest", bounds_error=False, fill_value=0.0)
    offs = (np.arange(sub) + 0.5) / sub
    sH = np.zeros((g.ny, g.nx)); nW = np.zeros((g.ny, g.nx)); nC = np.zeros((g.ny, g.nx))
    for oy in offs:
        for ox in offs:
            X, Y = np.meshgrid(g.xg[:-1] + ox * g.dx, g.yg[:-1] + oy * g.dx)
            LO, LA = to_geo(X, Y)
            pts = np.stack([LA.ravel(), LO.ravel()], -1)
            w = (fW(pts) >= 0.5).reshape(X.shape)
            h = fH(pts).reshape(X.shape)
            c = fC(pts).reshape(X.shape) > 0.5
            w &= c & np.isfinite(h)
            sH += np.where(w, h, 0.0); nW += w; nC += c
    with np.errstate(invalid="ignore", divide="ignore"):
        Hm = sH / nW
        wf = np.where(nC > 0, nW / nC, 0.0)
    return Hm, wf, nC / sub ** 2


# ---------------------------------------------------------------------------
# Opérations de masque
# ---------------------------------------------------------------------------
N4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def land_neighbours(wet):
    """Nombre de voisins terre (4-conn) ; l'extérieur du domaine compte comme mouillé
    (une cellule de frontière ouverte ne doit pas être éliminée à cause du bord)."""
    p = np.pad(wet, 1, constant_values=True)
    return ndi.convolve((~p).astype(int), N4, mode="constant")[1:-1, 1:-1]


def keep_connected(wet, seeds):
    """Garde les composantes (4-conn) qui touchent `seeds` ; si seeds vide -> la plus grande."""
    lab, n = ndi.label(wet, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if n <= 1:
        return wet, 0
    keep_ids = np.unique(lab[seeds & wet]) if seeds is not None and seeds.any() else []
    keep_ids = [k for k in keep_ids if k > 0]
    if not keep_ids:
        sizes = ndi.sum(wet, lab, range(1, n + 1))
        keep_ids = [int(np.argmax(sizes)) + 1]
    out = np.isin(lab, keep_ids)
    return out, int(wet.sum() - out.sum())


def edge_mask(shape):
    e = np.zeros(shape, bool)
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = True
    return e


# ---------------------------------------------------------------------------
# Arrondi hFac identique à MITgcm (ini_masks_etc.F, cas sans z* au repos)
# ---------------------------------------------------------------------------
def effective_depth(H, dz, hFacMin=C.HFAC_MIN, hFacMinDr=C.HFAC_MIN_DR):
    """Renvoie la profondeur (positive) que MITgcm utilisera réellement, et hFacC."""
    rf = rf_from_dz(dz)                       # 0, -dz1, ...
    Hn = -np.asarray(H, float)                # R_low (négatif)
    nz = len(dz)
    hfac = np.zeros((nz,) + Hn.shape)
    for k in range(nz):
        hmin_k = min(max(hFacMin, hFacMinDr / dz[k]), 1.0)
        h = (rf[k] - Hn) / dz[k]              # fraction de la cellule k au-dessus du fond
        h = np.clip(h, 0.0, 1.0)
        h = np.where(h < hmin_k, np.where(h < 0.5 * hmin_k, 0.0, hmin_k), h)
        hfac[k] = h
    Heff = np.tensordot(dz, hfac, axes=(0, 0))
    return Heff, hfac


# ---------------------------------------------------------------------------
# Entrées/sorties MITgcm
# ---------------------------------------------------------------------------
def write_bin(path, a, prec=64):
    dt = ">f8" if prec == 64 else ">f4"
    np.asarray(a, dtype=dt).tofile(path)       # a[ny, nx] C-order == Fortran (nx, ny)


def read_bin(path, shape, prec=64):
    dt = ">f8" if prec == 64 else ">f4"
    return np.fromfile(path, dtype=dt).reshape(shape)


def fortran_list(vals, per_line=5, fmt="{:.3f}"):
    s = [fmt.format(v) + "," for v in vals]
    lines = [" ".join(s[i:i + per_line]) for i in range(0, len(s), per_line)]
    return "\n        ".join(lines)
